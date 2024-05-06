import pandas as pd
# from senxor.utils import remap
import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import os

clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(5,5))
sr = cv.dnn_superres.DnnSuperResImpl_create()
sr.readModel(r"ESPCN_x3.pb")
sr.setModel("espcn",3)

def preprocess(frame):
    frame = frame[:,:-45]
    normalized_array = cv.normalize(frame.astype('float32'), None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    gray_image_norm = cv.cvtColor(normalized_array,cv.COLOR_GRAY2BGR)
    result = sr.upsample(gray_image_norm)
    # result = cv.resize(result, dsize=(460, 460),
    #                     interpolation=cv.INTER_AREA)
    imC = cv.applyColorMap(result, cv.COLORMAP_INFERNO)

    # return result
    return imC
