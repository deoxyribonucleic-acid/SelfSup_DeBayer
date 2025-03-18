from math import log10, sqrt
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def PSNR(original, noise):
    mse = np.mean((original - noise) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(original, noise):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    noise_gray = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
    (ssim_score, dif) = ssim(original_gray, noise_gray, full=True)
    return ssim_score
