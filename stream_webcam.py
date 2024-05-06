import cv2 as cv
import numpy as np
import os

def video_to_frames(path_output_dir):
    count = 0
    cam = cv.VideoCapture(1)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cropped_frame = np.array([array[::-1] for array in frame[10:,30:-30]])
        cv.imwrite(os.path.join(path_output_dir, f"face_{count}.png"), cropped_frame)
        cv.imshow("", cropped_frame)
        rval, frame = cam.read()
        count += 1
        key = cv.waitKey(1)
        if key == ord("q"):  # exit on ESC
            break
    cv.destroyAllWindows()

video_to_frames(r"C:\Users\takao\Desktop\YoloV8 Data\face_images\webcam_imgs")