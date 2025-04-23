from PIL import Image
import os
import shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src_folder', required=True, type=str)
parser.add_argument('-d', '--dst_folder', required=True, type=str)

opt, _ = parser.parse_known_args()
if not os.path.isdir(opt.src_folder):
    raise FileNotFoundError(f"Directory does not exist: {opt.src_folder}")
src_folder = opt.src_folder
dst_folder = opt.dst_folder
os.makedirs(dst_folder, exist_ok=True)

picked_img_num = 0
for filename in os.listdir(src_folder):
    if filename.lower().endswith(("jpg", "jpeg", "png")):
        img_path = os.path.join(src_folder, filename)
        img = Image.open(img_path)
        img = np.array(img)

        try:
            if img.shape[2] == 3 and img.shape[0] >= 256 and img.shape[0] <= 512 and img.shape[1] >= 256 and img.shape[1] <= 512:
                shutil.copy(img_path, dst_folder)
                picked_img_num = picked_img_num + 1
        except:
            continue

print(f"{picked_img_num} color images with proper size are picked!")
