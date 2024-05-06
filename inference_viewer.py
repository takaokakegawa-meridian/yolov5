import torch
from PIL import Image
import numpy as np
from pathlib import Path
import cv2 as cv
import os

# Load the YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp2/weights/best.pt')

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Move the model to the device
model.to(device)
# Set the model to evaluation mode
model.eval()

# # Path to the directory containing the images
# image_dir = Path('../datasets/test/images')
# count = 1
# with torch.no_grad():
#     # Iterate over the images in the directory
#     for image_path in image_dir.glob('*.jpg'):
#         # Load the image
#         image = Image.open(image_path)

#         # Convert the image to a format compatible with YOLOv5
#         image = np.array(image)

#         # Perform inference
#         results = model(image)
#         print(f"image shape: {image.shape}")
#         print(f"image dtype: {image.dtype}")
#         # cv.imshow('',image)
#         # Get the detected object labels, bounding box coordinates, and confidence scores
#         labels = results.xyxy[0][:, -1].numpy()
#         boxes = results.xyxy[0][:, :-1].numpy()
#         scores = results.xyxy[0][:, 4].numpy()

#         # # Print the detected objects and their bounding boxes
#         # print(f"Image: {image_path}")
#         for label, box, score in zip(labels, boxes, scores):
#             xB = int(box[2])
#             xA = int(box[0])
#             yB = int(box[3])
#             yA = int(box[1])
#             if int(label) == 0:
#                 cv.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 1)
#             else:
#                 cv.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 1)
#         cv.imwrite(os.path.join(r"C:\Users\takao\Desktop\YoloV8 Data\test_images", f"test_{count}.jpg"), image)
#         count += 1

def evaluate(arr):
    assert arr.shape == (460,460, 3), "incompatible shape"
    with torch.no_grad():
        results = model(arr)
    labels = results.xyxy[0][:, -1].numpy()
    boxes = results.xyxy[0][:, :-1].numpy()
    scores = results.xyxy[0][:, 4].numpy()
    return labels, boxes, scores

