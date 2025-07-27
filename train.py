from ultralytics import YOLO

#load a model
model = YOLO("./ultralytics/cfg/models/11/yolo11.yaml")
model = YOLO("./yolo11n.pt")

#Train the model
model.train(data='./mycoco.yaml',epochs=30,imgsz=640)