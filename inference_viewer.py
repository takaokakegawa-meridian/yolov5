import torch
from PIL import Image
import numpy as np
from pathlib import Path

# Load the YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp2/weights/best.pt')

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model.to(device)

# Set the model to evaluation mode
model.eval()

# Path to the directory containing the images
image_dir = Path('../datasets/test/images')

# Iterate over the images in the directory
for image_path in image_dir.glob('*.jpg'):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to a format compatible with YOLOv5
    image = np.array(image)

    # Perform inference
    results = model(image)

    # Get the detected object labels, bounding box coordinates, and confidence scores
    labels = results.xyxy[0][:, -1].numpy()
    boxes = results.xyxy[0][:, :-1].numpy()
    scores = results.xyxy[0][:, 4].numpy()

    # Print the detected objects and their bounding boxes
    print(f"Image: {image_path}")
    for label, box, score in zip(labels, boxes, scores):
        print('Label:', label)
        print('Bounding Box:', box)     # xmin, ymin, xmax, ymax
        print('Confidence Score:', score)
        print('---')
    print(type(box))