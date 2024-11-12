import os
import random
import shutil

from tqdm import tqdm

# Recursively delete all files and folders in the target output folder
for root, dirs, files in os.walk(r"serialize_dataset", topdown=False):
    for file_name in tqdm(files, desc="Deleting all files in the target output folder"):
        file_path = os.path.join(root, file_name)
        os.remove(file_path)

# Set the initdataset folder path
initdataset_path = "../init_dataset"
random.seed(4399)

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
    shutil.copy(jpg_file, f"../serialize_dataset/{i}.jpg")
    shutil.copy(json_file, f"../serialize_dataset/{i}.json")