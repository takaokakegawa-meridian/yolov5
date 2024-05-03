import pandas as pd
# from senxor.utils import remap
import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import os

clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(5,5))
sr = cv.dnn_superres.DnnSuperResImpl_create()
sr.readModel(r"ESPCN_x4.pb")
sr.setModel("espcn",4)

def preprocess(frame):
    frame = frame[:,45:]
    # frame_resized = cv.resize(frame, dsize=(frame.shape[1]*5, frame.shape[0]*5),
    #                           interpolation=cv.INTER_NEAREST_EXACT)
    normalized_array = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    gray_image_norm = clahe.apply(normalized_array)
    gray_image_norm = cv.cvtColor(gray_image_norm,cv.COLOR_GRAY2BGR)
    result = sr.upsample(sr.upsample(gray_image_norm))
    result = cv.resize(result, dsize=(frame.shape[1]*4, frame.shape[0]*4),
                        interpolation=cv.INTER_AREA)

    return result
