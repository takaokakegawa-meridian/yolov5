import cv2 as cv
import numpy as np
import os

def video_to_frames(path_output_dir):
    count = 0
    cam = cv.VideoCapture(0)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cropped_frame = np.array([array[::-1] for array in frame[:,90:-90]])
        # cropped_frame = np.array([array[::-1] for array in frame])
        print(f"cropped frame shape: {cropped_frame.shape}")
        # cv.imwrite(os.path.join(path_output_dir, f"face_{count}.png"), cropped_frame)
        cv.imshow("", cropped_frame)
        rval, frame = cam.read()
        count += 1
        key = cv.waitKey(1)
        if key == ord("q"):  # exit on ESC
            break
    cv.destroyAllWindows()

video_to_frames("")