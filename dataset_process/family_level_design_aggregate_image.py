import os
import json
import random
import shutil
import copy

from PIL import Image, ImageDraw
from tqdm import tqdm

def extract_chinese_directories(path):
    """
    Extract directories containing Chinese characters from the given path.

    Args:
        path (str): The file path.

    Returns:
        list: A list of directories containing Chinese characters.
    """
    directories = path.split(os.path.sep)
    chinese_directories = [directory for directory in directories if
                           any('\u4e00' <= char <= '\u9fff' for char in directory)]
    return chinese_directories

def is_overlap(position1, size1, position2, size2):
    """
    Check if two rectangles overlap.

    Args:
        position1 (tuple): The position of the first rectangle (x1, y1).
        size1 (tuple): The size of the first rectangle (w1, h1).
        position2 (tuple): The position of the second rectangle (x2, y2).
        size2 (tuple): The size of the second rectangle (w2, h2).

    Returns:
        bool: True if the rectangles overlap, False otherwise.
    """
    x1, y1 = position1
    w1, h1 = size1
    x2, y2 = position2
    w2, h2 = size2

    if x1 + w1 < x2 or x2 + w2 < x1:
        return False
    if y1 + h1 < y2 or y2 + h2 < y1:
        return False

    return True

def main():
    random.seed(4399)

    # Set the initdataset folder path
    initdataset_path = r"../init_dataset/family"

    # Get all subdirectories in the initdataset directory
    familys_folders = [f.path for f in os.scandir(initdataset_path) if f.is_dir()]

    # Get all images in each subdirectory
    famliy_list = []
    for family_folder in familys_folders:
        family = []
        for root, dirs, files in os.walk(family_folder):
            for file in files:
                if file.lower().endswith(".jpg"):
                    family.append(os.path.join(root, file))
        famliy_list.append(family)

    dataset = []
    for num in range(2, 7):
        tmp_family_lists = copy.deepcopy(famliy_list)
        # Randomly select x sublists
        selected_families = random.sample(tmp_family_lists, num)

        # Randomly select and remove one element from each list as the image to be synthesized
        jpg_lists = []
        for _ in range(226):
            jpg_list = []
            for lst in selected_families:
                if lst:
                    element = lst.pop(random.randrange(len(lst)))
                    jpg_list.append(element)
            jpg_lists.append(jpg_list)
        dataset.append(jpg_lists)

    # Background image path and output path
    background_path = '../4640_3480.jpg'
    output_path = 'result.jpg'

    # Rectangle color and width
    rectangle_color = 'red'
    rectangle_width = 3
    image_height = 3480
    image_width = 4640

    # Create a copy of the background image
    background_image = Image.open(background_path).copy()
    background_data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": [],
        "imagePath": "background.jpg",
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # Store the positions of already pasted annotation boxes
    occupied_positions = []

    # Recursively delete all files and folders in the target output folder
    shutil.rmtree('../labelme_output_images/family', ignore_errors=True)
    if not os.path.exists('../labelme_output_images'):
        os.makedirs('../labelme_output_images/family')

    # Synthesize images
    gradient = 2
    for jpg_lists in dataset:
        new_picture_id = 0
        save_dir = os.path.join('../labelme_output_images/family', f'family_{gradient}_{len(jpg_lists)}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        gradient += 1

        for jpg_list in tqdm(jpg_lists, desc=f"Processing the {gradient - 2} dataset images"):
            dict_areas = {}
            # Sort images by annotation box size from large to small
            for jpg in jpg_list:
                annotation_path = f'{os.path.splitext(jpg)[0]}.json'
                with open(annotation_path, 'r') as f:
                    data = json.load(f)
                for annotation in data['shapes']:
                    if annotation['shape_type'] != 'rectangle':
                        continue
                    x1, y1 = annotation['points'][0]
                    x2, y2 = annotation['points'][1]
                    area = abs(x2 - x1) * abs(y2 - y1)
                    dict_areas[jpg] = area

            jpg_list = sorted(dict_areas, key=dict_areas.get, reverse=True)

            for jpg in jpg_list:
                image_path = jpg
                image = Image.open(image_path)
                draw = ImageDraw.Draw(image)

                annotation_path = f'{os.path.splitext(jpg)[0]}.json'
                with open(annotation_path, 'r') as f:
                    data = json.load(f)

                for annotation in data['shapes']:
                    if annotation['shape_type'] != 'rectangle':
                        continue
                    x1, y1 = annotation['points'][0]
                    x2, y2 = annotation['points'][1]

                    annotation['label'] = extract_chinese_directories(jpg)[0].split()[1]
                    cropped_image = image.crop((x1, y1, x2, y2))

                    position_found = False
                    trys = 1000
                    while not position_found:
                        trys -= 1
                        if trys == 0:
                            print("ERROR: Too many jpgs to create!")
                            exit(0)
                        if len(jpg_lists) <= 2:
                            random_x = random.randint(0, background_image.width - cropped_image.width)
                            random_y = random.randint(0, background_image.height - cropped_image.height)
                        else:
                            if trys <= 997:
                                random_x = random.randint(0, background_image.width - cropped_image.width)
                                random_y = random.randint(0, background_image.height - cropped_image.height)
                            elif trys <= 998:
                                random_x = random.randint(0, background_image.width - cropped_image.width)
                                random_y = random.randint(0, 100)
                            else:
                                random_x = random.randint(0, 100)
                                random_y = random.randint(0, background_image.height - cropped_image.height)

                        new_position = (random_x, random_y)

                        overlap = False
                        for occupied_position in occupied_positions:
                            if is_overlap(new_position, cropped_image.size, occupied_position[0], occupied_position[1]):
                                overlap = True
                                break

                        if not overlap:
                            occupied_positions.append((new_position, cropped_image.size))
                            position_found = True

                            draw.rectangle((new_position[0], new_position[1], new_position[0] + cropped_image.width,
                                            new_position[1] + cropped_image.height),
                                           width=rectangle_width, outline=rectangle_color)

                            background_image.paste(cropped_image, new_position)

                            difference = [new_position[0] - x1, new_position[1] - y1]
                            for annotation in data['shapes']:
                                if annotation['shape_type'] == 'rectangle':
                                    new_x1 = new_position[0]
                                    new_y1 = new_position[1]
                                    cur_x2, cur_y2 = annotation['points'][1]
                                    new_x2 = cur_x2 + difference[0]
                                    new_y2 = cur_y2 + difference[1]
                                    background_annotation = annotation
                                    background_annotation['points'] = [[new_x1, new_y1], [new_x2, new_y2]]
                                    background_data['shapes'].append(background_annotation)
                                elif annotation['shape_type'] == 'point':
                                    new_x1 = annotation['points'][0][0] + difference[0]
                                    new_y1 = annotation['points'][0][1] + difference[1]
                                    background_annotation = annotation
                                    background_annotation['points'] = [[new_x1, new_y1]]
                                    background_data['shapes'].append(background_annotation)

            background_data['imagePath'] = f'{new_picture_id}.jpg'
            with open(os.path.join(save_dir, f'{new_picture_id}.json'), 'w') as f:
                json.dump(background_data, f)

            background_image.save(os.path.join(save_dir, f'{new_picture_id}.jpg'))
            new_picture_id += 1

            background_image = Image.open(background_path).copy()
            background_data = {
                "version": "5.4.1",
                "flags": {},
                "shapes": [],
                "imagePath": "background.jpg",
                "imageData": None,
                "imageHeight": image_height,
                "imageWidth": image_width
            }
            occupied_positions.clear()

if __name__ == "__main__":
    main()