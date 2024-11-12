import os
import json
import shutil
import random
import sys

from tqdm import tqdm
from PIL import Image

random.seed(4399)

def split_list_into_equal_chunks(lst, num_chunks):
    """
    Split a list into a specified number of equal chunks.

    Args:
        lst (list): The list to be split.
        num_chunks (int): The number of chunks to split the list into.

    Returns:
        list: A list of chunks.
    """
    avg_chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks

    start = 0
    chunks = []
    for i in range(num_chunks):
        chunk_size = avg_chunk_size + 1 if i < remainder else avg_chunk_size
        chunks.append(lst[start:start + chunk_size])
        start += chunk_size
    return chunks

def resize_single_picture_json(labelme_path):
    """
    Resize an image and update its corresponding JSON file with new coordinates.

    Args:
        labelme_path (str): The path to the JSON file.
    """
    with open(labelme_path) as json_file:
        data = json.load(json_file)

    old_width, old_height = data['imageWidth'], data['imageHeight']
    new_width, new_height = 2560, 2560
    scale_x, scale_y = new_width / old_width, new_height / old_height

    image = Image.open(f'{os.path.splitext(labelme_path)[0]}.jpg')
    image = image.resize((new_width, new_height))

    for shape in data['shapes']:
        shape_type = shape['shape_type']
        if shape_type == 'point':
            x, y = shape['points'][0]
            x, y = int(x * scale_x), int(y * scale_y)
            shape['points'][0] = [x, y]
        elif shape_type == 'rectangle':
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
            x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
            shape['points'][0], shape['points'][1] = [x1, y1], [x2, y2]

    image.save(f'{os.path.splitext(labelme_path)[0]}.jpg')
    data['imagePath'] = f'{os.path.splitext(labelme_path)[0]}.jpg'
    data['imageWidth'], data['imageHeight'] = new_width, new_height

    with open(labelme_path, 'w') as json_file:
        json.dump(data, json_file)

def convert_poly_to_rect(coordinate_list):
    """
    Convert polygon coordinates to rectangle coordinates.

    Args:
        coordinate_list (list): List of coordinates.

    Returns:
        tuple: A tuple containing the rectangle coordinates and a flag indicating if the conversion is valid.
    """
    X = [int(coordinate_list[2 * i]) for i in range(len(coordinate_list) // 2)]
    Y = [int(coordinate_list[2 * i + 1]) for i in range(len(coordinate_list) // 2)]

    Xmax = max(X)
    Xmin = min(X)
    Ymax = max(Y)
    Ymin = min(Y)
    flag = False
    if (Xmax - Xmin) == 0 or (Ymax - Ymin) == 0:
        flag = True
    return [Xmin, Ymin, Xmax - Xmin, Ymax - Ymin], flag

def convert_labelme_json_to_txt(files, out_txt_path, kind):
    """
    Convert LabelMe JSON files to YOLO format TXT files.

    Args:
        files (list): List of file names without extensions.
        out_txt_path (str): Output directory path.
        kind (str): Type of dataset (train, val, test).
    """
    global bbox_id

    json_list = [file + '.json' for file in files]
    for json_path in tqdm(json_list, desc=f"Processing {kind}: {len(json_list)}"):
        shutil.copy(f'{os.path.splitext(json_path)[0]}.jpg', os.path.join(out_txt_path, 'images', kind))

        with open(json_path, "r") as f_json:
            json_data = json.load(f_json)

        infos = json_data['shapes']
        if not infos:
            continue

        img_w = json_data['imageWidth']
        img_h = json_data['imageHeight']
        image_name = json_data['imagePath']
        txt_name = os.path.basename(json_path).split('.')[0] + '.txt'
        txt_path = os.path.join(os.path.join(out_txt_path, 'labels', kind), txt_name)

        with open(txt_path, 'w') as f:
            for label in infos:
                points = label['points']
                if len(points) < 2:
                    continue

                if len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                elif len(points) < 4:
                    continue

                segmentation = []
                for p in points:
                    segmentation.extend([int(p[0]), int(p[1])])

                bbox, flag = convert_poly_to_rect(segmentation)
                x1, y1, w, h = bbox

                if flag:
                    continue

                x_center = x1 + w / 2
                y_center = y1 + h / 2
                norm_x = x_center / img_w
                norm_y = y_center / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                family = label['label']
                if family not in bbox_class:
                    bbox_class[family] = bbox_id
                    bbox_id += 1

                obj_cls = bbox_class[family]
                line = [obj_cls, norm_x, norm_y, norm_w, norm_h]
                line = [str(ll) for ll in line]
                line = ' '.join(line) + '\n'
                f.write(line)

bbox_id = 0
bbox_class = {}

if __name__ == "__main__":
    init_dataset = r'../init_dataset/species'
    dataset_path = r'../labelme_output_images/species'
    output_path = r'../yolo_output_images/species'

    print("Dataset path:", dataset_path)
    print("Output path:", output_path)

    # Generate k-fold cross-validation data
    for k in range(10):
        output_path_k = os.path.join(output_path, str(k))

        # Delete all files and folders in the target output folder recursively
        if os.path.exists(output_path_k):
            shutil.rmtree(output_path_k)
        os.makedirs(output_path_k)

        for subdir in ['images', 'labels']:
            for kind in ['train', 'val', 'test']:
                os.makedirs(os.path.join(output_path_k, subdir, kind))

        # Select training and validation sets
        os.chdir(dataset_path)

        json_files = [file for file in os.listdir() if file.endswith('.json')]
        files_without_ext = [file.split('.')[0] for file in json_files]

        frac = 0.1
        fold_num = int(len(files_without_ext) * frac)
        k_files = split_list_into_equal_chunks(files_without_ext, 10)

        test_files = k_files[k]
        val_files = k_files[(k + 1) % 10]
        train_files = [file for i, files in enumerate(k_files) if i not in [k, (k + 1) % 10] for file in files]

        print(f"This is {k} fold, Total: {len(files_without_ext)}")
        convert_labelme_json_to_txt(train_files, output_path_k, 'train')
        convert_labelme_json_to_txt(val_files, output_path_k, 'val')
        convert_labelme_json_to_txt(test_files, output_path_k, 'test')

        print({v: k for k, v in bbox_class.items()})
        print(f"Successful: {k}")