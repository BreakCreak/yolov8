from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolov8n-pose.pt")

# Define path to the image file
source = "inference.jpg"

# Run inference on the source
results = model(source)  # list of Results objects

for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")