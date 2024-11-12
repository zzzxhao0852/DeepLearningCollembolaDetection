from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLO model with the pre-trained weights
    model = YOLO("yolov8m-cls.pt")

    # Train the model with the specified parameters
    result = model.train(
        project='runs/cls',  # Project directory
        name='crop_2560X2560',  # Experiment name
        data='cls_dataset(cls_crop)',  # Dataset configuration
        epochs=100,  # Number of training epochs
        batch=2,  # Batch size
        seed=4399,  # Random seed for reproducibility
        single_cls=False,  # Whether to treat all classes as a single class
        imgsz=224,  # Image size for training
        save_period=10,  # Save checkpoint every 10 epochs
        close_mosaic=80,  # Close mosaic augmentation after 80% of epochs
        pretrained=True,  # Use pre-trained weights
        optimizer='auto',  # Automatic optimizer selection
        scale=0.0  # Scale augmentation (set to 0.0 to disable)
    )