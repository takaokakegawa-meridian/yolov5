# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2020. All rights reserved.
#
import sys
import os
import signal
import time
import logging
import serial
import numpy as np
# import pandas as pd
from senxorplus.stark import STARKFilter
from homography import homographic_blend
import cv2 as cv

# from senxor.mi48 import MI48, format_header, format_framestats
from senxor.utils import data_to_frame, remap,\
                         cv_render, RollingAverageFilter,\
                         connect_senxor

# This will enable mi48 logging debug messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))


# Make the a global variable and use it as an instance of the mi48.
# This allows it to be used directly in a signal_handler.
global mi48
sr = cv.dnn_superres.DnnSuperResImpl_create()
sr.readModel(r"ESPCN_x4.pb")
sr.setModel("espcn",4)
def upsample_display(frame):
    img = cv_render(remap(frame),
              resize=(frame.shape[1],frame.shape[0]),
              colormap='rainbow2',
              display=False)
    return sr.upsample(img)
# define a signal handler to ensure clean closure upon CTRL+C
# or kill from terminal
def signal_handler(sig, frame):
    """Ensure clean exit in case of SIGINT or SIGTERM"""
    logger.info("Exiting due to SIGINT or SIGTERM")
    mi48.stop()
    cv.destroyAllWindows()
    logger.info("Done.")
    sys.exit(0)

# Define the signals that should be handled to ensure clean exit
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Make an instance of the MI48, attaching USB for 
# both control and data interface.
# can try connect_senxor(src='/dev/ttyS3') or similar if default cannot be found
mi48, connected_port, port_names = connect_senxor()
ncols, nrows = mi48.fpa_shape

# print out camera info
logger.info('Camera info:')
logger.info(mi48.camera_info)

# set desired FPS
mi48.regwrite(0xB4, 0x03)

# see if filtering is available in MI48 and set it up
mi48.regwrite(0xD0, 0x00)   # disable temporal filter
#mi48.regwrite(0x20, 0x00)   # disable STARK filter
mi48.regwrite(0x30, 0x00)   # disable median filter
mi48.regwrite(0x25, 0x00)   # disable MMS
#mi48.disable_filter(f1=True, f2=True, f3=True)
#mi48.set_filter_1(85)
#mi48.enable_filter(f1=True, f2=False, f3=False, f3_ks_5=False)


mi48.set_sens_factor(95)
mi48.set_offset_corr(1.5)
mi48.set_emissivity(97)

# initiate continuous frame acquisition
time.sleep(1)
with_header = True
mi48.start(stream=True, with_header=with_header)

# set cv_filter parameters
#par = {'blur_ks':3, 'd':5, 'sigmaColor': 27, 'sigmaSpace': 27}

minav = RollingAverageFilter(N=15)
maxav = RollingAverageFilter(N=8)

minav2 = RollingAverageFilter(N=15)
maxav2 = RollingAverageFilter(N=8)

stark_par = {'sigmoid': 'sigmoid',
             'lm_atype': 'ra',
             'lm_ks': (3,3),
            #  'lm_ad': 9,
            'lm_ad': 6,
             'alpha': 2.0,
             'beta': 2.0,}
frame_filter = STARKFilter(stark_par)
# with torch.no_grad():
data, header = mi48.read()
cam = cv.VideoCapture(0)

if (data is None) or not (cam.isOpened()):
        logger.critical('NONE data received instead of GFRA')
        mi48.stop()
        cv.destroyAllWindows()
        sys.exit(1)
else:
    while True:
        data, header = mi48.read()
        rval, raw_webcam = cam.read()
        webcam_img = np.array([array[::-1] for array in raw_webcam[:,90:-90]])
        # min/max stabilization
        # clip before and after applying STARK filter
        frame = data_to_frame(data, (ncols, nrows), hflip=False)[:,:-45]
        min_temp1= minav(np.median(np.sort(frame.flatten())[:16]))
        max_temp1= maxav(np.median(np.sort(frame.flatten())[-5:]))
        frame = np.clip(frame, min_temp1, max_temp1)
        frame = frame_filter(frame)
        min_temp2 = minav2(np.median(np.sort(frame.flatten())[:9]))
        max_temp2= maxav2(np.median(np.sort(frame.flatten())[-5:]))
        frame = np.clip(frame, min_temp1, max_temp2)

        frange = frame.max() - frame.min()
        print(f"frame shape: {frame.shape}")
        print(f'{data.min():1f}: {min_temp1:.1f} ({min_temp2:.1f}), {data.max():.1f}: {max_temp1:.1f} ({max_temp2:.1f})')
        
        thermal_img = cv_render(remap(frame),
                resize=(frame.shape[1]*4,frame.shape[0]*4),
                colormap='rainbow2',
                display=False)
        
        overlay_img = homographic_blend(thermal_img, webcam_img)

        cv.namedWindow("Thermal Image")
        cv.namedWindow("Webcam Image")
        cv.namedWindow("Overlaid Image")
        cv.imshow("Thermal Image",thermal_img)
        cv.imshow("Webcam Image",webcam_img)
        cv.imshow("Overlaid Image",overlay_img)

        key = cv.waitKey(1)  # & 0xFF
        if key == ord("q"):
            break

# stop capture and quit
mi48.stop()
cv.destroyAllWindows()

