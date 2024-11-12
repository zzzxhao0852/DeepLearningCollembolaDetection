from PIL import Image, ImageDraw, ImageFilter
import os
import random
import json
import shutil
from tqdm import tqdm

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

random.seed(4399)

# Set the initdataset folder path
initdataset_path = "../init_dataset/species"

# Synthesis mode: random or design
# random: Randomly shuffle all images and paste them onto the background image until no more positions are available.
# design: Specify the number of families, genera, species, and individuals in a synthesized image, and paste each folder's images onto the background image.

# Recursively delete all files and folders in the target output folder
for root, dirs, files in os.walk(r"serialize_dataset", topdown=False):
    for file_name in tqdm(files, desc="Deleting all files in the target output folder"):
        file_path = os.path.join(root, file_name)
        os.remove(file_path)

# Collect all JPG files in the initdataset folder
jpg_files = []
for root, dirs, files in os.walk(initdataset_path):
    for file in files:
        if file.lower().endswith(".jpg"):
            jpg_files.append(os.path.join(root, file))

# Randomly shuffle the JPG file list
random.shuffle(jpg_files)

# Move corresponding JPG and JSON files to a new folder
for i, jpg_file in enumerate(tqdm(jpg_files, desc="Moving corresponding JPG and JSON files to a new folder")):
    json_file = os.path.splitext(jpg_file)[0] + ".json"
    shutil.copy(jpg_file, f"F:/StorageFile/Code/zhuangxiaohao/YoloAICollembolaPose/serialize_dataset/{i}.jpg")
    shutil.copy(json_file, f"F:/StorageFile/Code/zhuangxiaohao/YoloAICollembolaPose/serialize_dataset/{i}.json")

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
new_picture_id = 0
random.seed(1080)

# Recursively delete all files and folders in the target output folder
for root, dirs, files in os.walk("../labelme_output_images/species", topdown=False):
    for file_name in tqdm(files, desc="Deleting all files in the target output folder"):
        file_path = os.path.join(root, file_name)
        os.remove(file_path)

# Process all images in the serialize_dataset folder
for filename in tqdm(os.listdir('../serialize_dataset'), desc="Processing images"):
    if not filename.endswith('.jpg'):
        continue

    # Open the current image and create a drawable object
    image_path = os.path.join('../serialize_dataset', filename)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Open the corresponding annotation file
    annotation_path = os.path.join('../serialize_dataset', f'{os.path.splitext(filename)[0]}.json')
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Process all annotations in the current image
    for annotation in data['shapes']:
        if annotation['shape_type'] != 'rectangle':
            continue
        x1, y1 = annotation['points'][0]
        x2, y2 = annotation['points'][1]

        # Crop the annotation box and calculate its position
        cropped_image = image.crop((x1, y1, x2, y2))

        # Randomly generate a new position until a non-overlapping position is found
        position_found = False
        trys = 1000
        while not position_found:
            trys -= 1
            if trys == 0:
                background_data['imagePath'] = str(new_picture_id) + '.jpg'
                with open(os.path.join('../labelme_output_images/species', f'{os.path.splitext(str(new_picture_id))[0]}.json'), 'w') as f:
                    json.dump(background_data, f)

                # Save the current image with annotations
                background_image.save(os.path.join('../labelme_output_images/species', str(new_picture_id) + '.jpg'))
                new_picture_id += 1

                # Create a new background image and JSON data
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
                trys = 1000

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

            # Check if the new position overlaps with any existing annotation boxes
            overlap = False
            for occupied_position in occupied_positions:
                if is_overlap(new_position, cropped_image.size, occupied_position[0], occupied_position[1]):
                    overlap = True
                    break

            if not overlap:
                occupied_positions.append((new_position, cropped_image.size))
                position_found = True

                # Draw the annotation box and paste the cropped image onto the background image
                draw.rectangle((new_position[0], new_position[1], new_position[0] + cropped_image.width,
                                new_position[1] + cropped_image.height),
                               width=rectangle_width, outline=rectangle_color)

                background_image.paste(cropped_image, new_position)

                # Update the JSON data after pasting
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

# Save the final image with annotations
background_data['imagePath'] = str(new_picture_id) + '.jpg'
with open(os.path.join('../labelme_output_images/species', f'{os.path.splitext(str(new_picture_id))[0]}.json'), 'w') as f:
    json.dump(background_data, f)

background_image.save(os.path.join('../labelme_output_images/species', str(new_picture_id) + '.jpg'))
print(str(new_picture_id) + '.jpg')