from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11l.pt")

# Export the model to TensorRT
model.export(format="engine",device="cuda:0",imgsz=(1280,1280),half=True,dynamic=True,batch=1)  # creates 'yolo11n.engine'

# Load the exported TensorRT model
# trt_model = YOLO("yolo11l.engine")

# # Run inference with confidence threshold and class filtering
# results = trt_model("RangelinePhelpsNB.jpg", imgsz=(1280, 1280), conf=0.5, classes=[2, 5, 7])  # Adjust conf & class as needed

# # Save the processed image with detection results
# for i, result in enumerate(results):
#     result.save(filename=f"output_{i}.jpg")  # Saves image with bounding boxes

# print("Processed image saved with bounding boxes.")