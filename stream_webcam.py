# # import the opencv library 
# import cv2 as cv
# import torch
# # Load the YOLOv5s model
# # model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp2/weights/best.pt')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# # Set the device to GPU if available, otherwise use CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Move the model to the device
# model.to(device)
# # Set the model to evaluation mode
# model.eval()


# # define a video capture object 
# vid = cv.VideoCapture(1) 
# with torch.no_grad():
#     while True: 
#         # Capture the video frame 
#         # by frame 
#         ret, frame = vid.read() 
#         print(frame.shape)
#         # # Perform inference
#         # results = model(frame)
#         # # Get the detected object labels, bounding box coordinates, and confidence scores
#         # labels = results.xyxy[0][:, -1].numpy()
#         # boxes = results.xyxy[0][:, :-1].numpy()
#         # scores = results.xyxy[0][:, 4].numpy()
#         # for label, box, score in zip(labels, boxes, scores):
#         #     xB = int(box[2])
#         #     xA = int(box[0])
#         #     yB = int(box[3])
#         #     yA = int(box[1])
#         #     cv.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

#         # Display the resulting frame 
#         cv.imshow('frame', frame)
        
#         # the 'q' button is set as the python
#         # quitting button you may use any 
#         # desired button of your choice 
#         if cv.waitKey(1) & 0xFF == ord('q'): 
#             break

# # After the loop release the cap object 
# vid.release() 
# # Destroy all the windows 
# cv.destroyAllWindows() 

import cv2
import threading

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

# Create two threads as follows
thread1 = camThread("Camera 1", 1)
thread2 = camThread("Camera 2", 2)
thread1.start()
thread2.start()

