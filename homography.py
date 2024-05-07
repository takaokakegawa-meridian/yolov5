import cv2 as cv
import os
import argparse
import numpy as np

local_thermal_root = r"C:\Users\takao\Desktop\YoloV8 Data\face_images\thermal_imgs"
local_webcam_root = r"C:\Users\takao\Desktop\YoloV8 Data\face_images\webcam_imgs"
local_homography_root = r"C:\Users\takao\Desktop\YoloV8 Data\face_images\homography_imgs"

thermal_pts = np.float32([[166,208],[284,201],[227,262],[229,322]]) # thermal landmark coords 
webcam_pts = np.float32([[166,216],[298,218],[228,299],[226,375]])

def naive_blend(image1, image2, alpha=0.5):
    assert alpha <= 1.0 and alpha >= 0., "alpha must be between 0 and 1"
    super_imposed_img = cv.addWeighted(image1, alpha, image2, 1-alpha, 0)
    return super_imposed_img

def homographic_blend(thermal_img, webcam_img, thermal_pts, webcam_pts):
    rows,cols, _ = webcam_img.shape
    M = cv.getPerspectiveTransform(thermal_pts,webcam_pts)    
    dst = cv.warpPerspective(thermal_img, M, (cols, rows))
    overlay = cv.addWeighted(webcam_img, 0.3, dst, 0.7, 0)
    return overlay

def main(thermal_root, webcam_root):
    for f in os.listdir(thermal_root):
        idx = int(f.split(".")[0].split("_")[1])
        thermal_img = cv.imread(os.path.join(thermal_root, f"sampt_{idx}.png"))
        webcam_img = cv.imread(os.path.join(webcam_root, f"samp_{idx}.png"))
        overlay = homographic_blend(thermal_img, webcam_img, thermal_pts, webcam_pts)
        # overlay = homographic_blend(thermal_img, webcam_img, thermal_pts, webcam_pts)


    # cv.namedWindow("Overlaid Image")
    # cv.imshow("Overlaid Image",overlay)
        cv.imwrite(os.path.join(local_homography_root,
                                        f"homography_{idx}.png"),
                        overlay)
        print(f"processed and saved img idx {idx}")
    # cv.waitKey(0)
    # # Close all windows
    # cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-thermal', '--thermal-root', default=local_thermal_root, type=str,
                        dest='thermal_root', help='thermal image root directory')
    parser.add_argument('-webcam', '--webcam-root', default=local_webcam_root, type=str,
                        dest='webcam_root', help='webcam image root directory')
    args = parser.parse_args()
    main(args.thermal_root, args.webcam_root)