from ultralytics import YOLO

model = YOLO("./BeltDetection.pt")

model.export(format="onnx")