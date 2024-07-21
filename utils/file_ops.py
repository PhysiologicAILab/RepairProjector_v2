import os
from pathlib import Path
import cv2

def list_images(datapath):

    damage_path = Path(os.path.join(datapath, "damages"))
    repair_path = Path(os.path.join(datapath, "repairs"))

    pattern = ['*.jpg', '*.png', '*.jpeg', '*.JPG']

    repair_images = []
    for ptn in pattern:
        tmp_list = list(repair_path.glob(pattern=ptn))
        for im in tmp_list:
            repair_images.append(im)

    damage_images = []
    for ptn in pattern:
        tmp_list = list(damage_path.glob(pattern=ptn))
        for im in tmp_list:
            damage_images.append(im)

    print(damage_images)

    return repair_images, damage_images
    

def load_image(im_path):
    img = cv2.imread(im_path)
    return img
