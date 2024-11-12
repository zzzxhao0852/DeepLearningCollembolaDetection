From Labeled Dataset to Family-Genus-Species Community Datasets

1 After labeling the dataset, first run check_json.py to check for any errors in the annotations.

2 Copy the organized data to init_dataset (or remember to back it up).

3.1 Random Synthesis Mode (Species Community Datasets):

Run species_level_random_aggregate_image.py to randomly paste Collembola from a single image onto a background image and update the JSON file accordingly. The synthesized community images are output to labelme_output_images.

3.2 Design Synthesis Mode (Families Community Datasets, Genera Community Datasets):

Run family_level_design_aggregate_image.py or order_level_design_aggregate_image.py.

4 Modify the dataset_path and output_path in the labelme_output_images folder, then run labelme2yolo (labelme2yolo_detect.py or labelme2yolo_pose.py) to convert LabelMe-formatted JSON annotation files to YOLO-formatted TXT annotation files, outputting them to yolo_output_images.

5 You need to modify the YAML file in the labelme_output_images folder to match the output folder path, names, and bbox_class with the labelme2yolo.py file's Train/val/test path.

6 Modify the project and data parameters, then run the train.py.

7 Modify the model path in val.py. If the names in the YAML file do not match, also modify the names.