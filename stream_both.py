
import cv2 as cv
import sys
import os
import signal
import time
import logging
import numpy as np
from senxorplus.stark import STARKFilter
from preprocessing import preprocess
from senxor.utils import data_to_frame, remap,\
                         cv_render, RollingAverageFilter,\
                         connect_senxor
import threading
from inference_viewer import evaluate

# This will enable mi48 logging debug messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

global mi48

def signal_handler(sig, frame):
    """Ensure clean exit in case of SIGINT or SIGTERM"""
    logger.info("Exiting due to SIGINT or SIGTERM")
    mi48.stop()
    cv.destroyAllWindows()
    logger.info("Done.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
mi48, connected_port, port_names = connect_senxor()
ncols, nrows = mi48.fpa_shape
# print out camera info
logger.info('Camera info:')
logger.info(mi48.camera_info)
mi48.regwrite(0xB4, 0x03)
mi48.regwrite(0xD0, 0x00)   # disable temporal filter
mi48.regwrite(0x30, 0x00)   # disable median filter
mi48.regwrite(0x25, 0x00)
mi48.set_sens_factor(100)
mi48.set_offset_corr(0.0)
time.sleep(1)
with_header = True
mi48.start(stream=True, with_header=with_header)
minav = RollingAverageFilter(N=25)
maxav = RollingAverageFilter(N=16)

minav2 = RollingAverageFilter(N=25)
maxav2 = RollingAverageFilter(N=16)

stark_par = {'sigmoid': 'sigmoid',
             'lm_atype': 'ra',
             'lm_ks': (3,3),
             'lm_ad': 9,
             'alpha': 2.0,
             'beta': 2.0,}
frame_filter = STARKFilter(stark_par) 

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    if camID == 'photo':
        cv.namedWindow(previewName)
        cam = cv.VideoCapture(1)
        if cam.isOpened():  # try to get the first frame
            rval, frame = cam.read()
        else:
            rval = False

        while rval:
            cropped_frame = np.array([array[::-1] for array in frame[10:,30:-30]])
            cv.imshow(previewName, evaluate(cropped_frame))
            rval, frame = cam.read()
            key = cv.waitKey(1)
            if key == ord("q"):  # exit on ESC
                break
        cv.destroyWindow(previewName)
    else:
        while True:
            data, _ = mi48.read()
            if data is None:
                logger.critical('NONE data received instead of GFRA')
                mi48.stop()
                sys.exit(1)

            frame = data_to_frame(data, (ncols, nrows), hflip=False)[15:-17,14:-59];
            min_temp1= minav(np.median(np.sort(frame.flatten())[:16]))
            max_temp1= maxav(np.median(np.sort(frame.flatten())[-5:]))
            frame = np.clip(frame, min_temp1, max_temp1)
            frame = frame_filter(frame)
            # min_temp2 = minav2(np.median(np.sort(frame.flatten())[:9]))
            max_temp2= maxav2(np.median(np.sort(frame.flatten())[-5:]))
            frame = np.clip(frame, min_temp1, max_temp2)

            cv_render(remap(frame),
                      resize=(frame.shape[1]*3,frame.shape[0]*3),
                      colormap='inferno')
            # cv.imshow('', preprocess(frame))
            key = cv.waitKey(1)  # & 0xFF
            if key == ord("q"):
                break
        mi48.stop()
        cv.destroyAllWindows()

# Create two threads as follows
thread1 = camThread("Photo WebCam", 'photo')
thread2 = camThread("Thermal Camera", 'thermal')
thread1.start()
thread2.start()

