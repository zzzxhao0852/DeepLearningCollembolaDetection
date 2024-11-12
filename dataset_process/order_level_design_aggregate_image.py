from PIL import Image, ImageDraw, ImageFilter
import os
import random
import json
import shutil
from tqdm import tqdm
import time
import copy

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

random.seed(4396)

# Set the initdataset folder path
initdataset_path = r"F:\StorageFile\Code\zhuangxiaohao\YoloAICollembolaPose\init_dataset\family"

# Get all family directories in the initdataset directory
families_folders = [f.path for f in os.scandir(initdataset_path) if f.is_dir()]

# Get all images in each order directory under each family directory
# familylists structure: familylists[ family[ order[ images ] ] ]
familylists = []
for families_folder in families_folders:
    family = []
    # Get all order directories under the family directory
    order_folders = [f.path for f in os.scandir(families_folder) if f.is_dir()]
    for order_folder in order_folders:
        order = []
        for root, dirs, files in os.walk(order_folder):
            for file in files:
                if file.lower().endswith(".jpg"):
                    order.append(os.path.join(root, file))
        family.append(order)
    familylists.append(family)

# For each family, randomly select and remove one element from each order to create the dataset
# For each family, select num orders to create a dataset with the same family but different orders
# Store the images to be synthesized in the dataset
dataset = []
# Loop through the family lists and select num order lists
for num in range(2, 11):
    tmp_familylists = copy.deepcopy(familylists)
    # Iterate through each family's order lists
    for tmp_family in tmp_familylists:
        jpg_lists = []
        # Skip the family if the number of orders is less than num
        if len(tmp_family) < num:
            continue
        # Randomly select x order lists
        selected_orders = random.sample(tmp_family, num)
        for _ in range(100):
            jpg_list = []
            for order in selected_orders:
                # Randomly select and remove an image from the current order list
                element = order.pop(random.randrange(len(order)))
                jpg_list.append(element)
            jpg_lists.append(jpg_list)
        # Add the generated image lists to the dataset
        print(len(jpg_lists))
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
if not os.path.exists('../labelme_output_images/family'):
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
                annotation['label'] = jpg.split('\\')[-4]
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

                cropped_image = image.crop((x1, y1, x2, y2))

                position_found = False
                trys = 1000
                while not position_found:
                    trys -= 1
                    if trys == 0:
                        print("ERROR: Too many jpgs to create!")
                        print(jpg_list)
                        print(len(jpg_list))
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