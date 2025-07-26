from ultralytics import YOLO

# load a model
model = YOLO("./ultralytics/cfg/models/v8/yolov8-seg.yaml")
model = YOLO("./yolov8n-seg.pt")

# Train the model
model.train(data="./ultralytics/cfg/datasets/coco128-seg.yaml", epochs=30, imgsz=640)
