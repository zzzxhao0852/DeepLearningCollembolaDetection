from ultralytics import YOLO


if __name__ == '__main__':
    # Load a custom model
    model = YOLO(r'runs/cls/crop_224x224/train4/weights/best.pt')

    # Evaluate the model's performance on the test set
    metrics = model.val(
        data='cls_dataset(cls_crop)',  # Dataset configuration
        batch=2,  # Batch size
        split='test',  # Split to evaluate (test set)
        imgsz=224,  # Image size for evaluation
        max_det=30,  # Maximum number of detections per image
        seed=9999,  # Random seed for reproducibility
        plots=True,  # Generate plots for evaluation
        workers=1  # Number of worker threads for data loading
    )

    # Print the evaluation metrics
    print(metrics)