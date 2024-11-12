import os
import time
import torch
from ultralytics import YOLO
import yaml
import shutil

if __name__ == '__main__':
    # List of model names to train
    model_list = ['yolov8n', 'yolov8s', 'yolov8m']

    for model_name in model_list:
        for k in range(10):
            # Skip the first two folds for yolov8m
            if model_name == 'yolov8m' and k < 2:
                continue

            print(f"{model_name} - species: {k} dataset training:")

            # Load the YAML configuration file
            yaml_path = 'yaml/detect_different_species.yaml'
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)

            # Modify the path in the YAML configuration
            data['path'] = os.path.join(
                '../yolo_output_images/species', str(k))

            # Write the modified data back to the YAML file
            with open(yaml_path, 'w') as file:
                yaml.dump(data, file)

            # Load the YOLO model
            model = YOLO(f'{model_name}.pt')

            # Define the training run directory
            run_dir = os.path.join(
                'runs/detect', model_name, f'51kinds_4640x3480_different_species{k}',
                '2batch50epochs2560imgsz2dims50close_mosaic4points')

            # Remove the existing run directory if it exists
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)

            # Train the model
            results = model.train(
                project=os.path.join('runs/detect', model_name, f'51kinds_4640x3480_different_species{k}'),
                name='2batch50epochs2560imgsz2dims50close_mosaic4points',
                save_period=10, batch=2, imgsz=2560, close_mosaic=50,
                data=yaml_path, seed=4399, epochs=50,
                device=0, pretrained=True, optimizer='auto', scale=0.0, dropout=0.2, workers=1)

            # Free up memory
            del model
            torch.cuda.empty_cache()
            time.sleep(5)