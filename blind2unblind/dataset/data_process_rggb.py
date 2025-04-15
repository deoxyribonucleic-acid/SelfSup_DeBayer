import os
from glob import glob
import numpy as np
from scipy.io import savemat
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm

data_dir = "./SIDD_Medium_Raw/Data/"
path_all_noisy = sorted(glob(os.path.join(data_dir, '**/*NOISY*.MAT'), recursive=True))
print('Number of noisy images: {:d}'.format(len(path_all_noisy)))

save_folder = "../data/train/SIDD_Medium_Raw_noisy_sub512_rggb/"
if os.path.exists(save_folder):
    os.system(f"rm -r {save_folder}")
os.makedirs(save_folder)

# Bayer pattern mapping
pattern_map = {
    'GP': 'BGGR',
    'IP': 'RGGB',
    'S6': 'GRBG',
    'N6': 'BGGR',
    'G4': 'BGGR'
}

offset_map = {
    'rggb': (0, 0),
    'grbg': (0, 1),
    'gbrg': (1, 0),
    'bggr': (1, 1)
}

crop_size = 512
step = 256

def get_camera_code(path):
    for key in pattern_map.keys():
        if key in path:
            return key
    return None

def to_rggb(im, pattern):
    dx, dy = offset_map.get(pattern.lower(), (0, 0))
    return im[dx:, dy:]

def process_image(idx):
    path = path_all_noisy[idx]
    cam_code = get_camera_code(path)
    if cam_code is None:
        print(f"Unknown camera pattern in path: {path}")
        return

    pattern = pattern_map[cam_code]
    img_name, ext = os.path.splitext(os.path.basename(path))
    with h5py.File(path, 'r') as mat:
        im = np.array(mat['x'])

    im = to_rggb(im, pattern)
    h, w = im.shape
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > 0:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > 0:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped = im[x:x + crop_size, y:y + crop_size]
            cropped = np.ascontiguousarray(cropped)
            save_path = os.path.join(save_folder, f"{img_name}_s{index:03d}.mat")
            savemat(save_path, {"x": cropped})

Parallel(n_jobs=10)(delayed(process_image)(i) for i in tqdm(range(len(path_all_noisy))))
