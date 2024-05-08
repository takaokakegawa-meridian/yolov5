import matplotlib.pyplot as plt
import cv2 as cv
import argparse
import os
import keyboard

local_thermal_root = r"C:\Users\takao\Desktop\YoloV8 Data\face_images\facial_overlaid_imgs\thermal_imgs"
local_webcam_root = r"C:\Users\takao\Desktop\YoloV8 Data\face_images\facial_overlaid_imgs\webcam_imgs"

def view_display(path1, path2):
    fig, ax = plt.subplots(figsize=(12,6),ncols=2)
    img1, img2 = cv.imread(path1), cv.imread(path2)
    img1, img2 = cv.cvtColor(img1, cv.COLOR_BGR2RGB), cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    idx = path1.split("_")[-1].split(".")[0]
    ax[0].set_title(f"index: {idx}")
    ax[1].set_title(f"index: {idx}")
    plt.show()

def main(thermal_root, webcam_root, idx):
    view_display(os.path.join(thermal_root,f"thermal_{idx}.png"),
                 os.path.join(webcam_root,f"sampt_{idx}.png")
                 )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-thermal', '--thermal-root', default=local_thermal_root, type=str,
                        dest='thermal_root', help='thermal image root directory')
    parser.add_argument('-webcam', '--webcam-root', default=local_webcam_root, type=str,
                        dest='webcam_root', help='webcam image root directory')
    # parser.add_argument('-idx', '--index', default=10, type=int,
    #                     dest='idx', help='index image')
    args = parser.parse_args()
    for i in range(110,114):
        print(f"viewing image index {i} now:")
        main(args.thermal_root, args.webcam_root, i)