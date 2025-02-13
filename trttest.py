from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolo11m.pt')

# Export the model to TensorRT format
model.export(
    format='engine', 
    device=0, 
    imgsz=(640,640), 
    dynamic=True, 
    batch=1,
    half=True,
)  # creates 'yolov8m.engine'