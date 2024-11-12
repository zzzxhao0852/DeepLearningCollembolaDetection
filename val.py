from ultralytics import YOLO
import time
import torch
import yaml
import os.path
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # List of model names to evaluate
    model_list = ['yolov8n', 'yolov8s', 'yolov8m']

    for model_name in model_list:
        for k in range(6, 7):
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

            # Load the custom model
            model_path = os.path.join(
                'runs/detect', model_name, f'51kinds_4640x3480_different_species{k}',
                '2batch50epochs2560imgsz2dims50close_mosaic4points/weights/best.pt')
            model = YOLO(model_path)

            # Evaluate the model's performance on the test set
            # Note: Do not set --save-hybrid to True, as it will write true and predicted values to the annotation file *.txt,
            # causing P R mAP@.5 to become extremely high.
            metrics = model.val(
                data=yaml_path,  # Dataset configuration
                batch=2,  # Batch size
                split='test',  # Split to evaluate (test set)
                imgsz=2560,  # Image size for evaluation
                max_det=30,  # Maximum number of detections per image
                seed=4399,  # Random seed for reproducibility
                plots=True,  # Generate plots for evaluation
                workers=1  # Number of worker threads for data loading
            )

            # Extract precision values
            p = metrics.box.p
            print(f"{model_name}: {k}")
            print(p)
            print(len(p.tolist()))
            print(metrics.box.mp)

            # Append mean precision to the precision array
            p = np.append(p, metrics.box.mp)
            p = p.tolist()

            # Create a DataFrame
            df = pd.DataFrame(p, columns=['Value'])

            # Save the DataFrame to an Excel file
            output_path = os.path.join('runs/val/species', model_name, f'{1}_{k}_results.xlsx')
            df.to_excel(output_path, index=False)

            # Free up memory
            del model
            torch.cuda.empty_cache()
            time.sleep(5)