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
predictions = model("my_image.png")

# Set the model to evaluation mode
model.eval()
