import cv2 as cv

clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(5,5))
sr = cv.dnn_superres.DnnSuperResImpl_create()
sr.readModel(r"ESPCN_x3.pb")
sr.setModel("espcn",3)

def preprocess(frame):
    # frame = frame[:,:-45]
    normalized_array = cv.normalize(frame.astype('float32'), None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    gray_image_norm = cv.cvtColor(normalized_array,cv.COLOR_GRAY2BGR)
    result = sr.upsample(gray_image_norm)
    return result
