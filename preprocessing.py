import numpy as np
import torch


def bayer_filter_tensor(img):
    # img is RGB, BCHW
    bayer = img.detach().clone()
    
    bayer[:, 0::2, 0::2, 0::2] = 0 # Green pixels - set the blue and the red planes to zero (and keep the green)
    bayer[:, 1:, 0::2, 1::2] = 0   # Red pixels - set the blue and the green planes to zero (and keep the red)
    bayer[:, :2, 1::2, 0::2] = 0   # Blue pixels - set the red and the green planes to zero (and keep the blue)
    bayer[:, 0::2, 1::2, 1::2] = 0 # Green pixels - set the blue and the red planes to zero (and keep the green)

    return bayer


def bayer_filter_ndarray(img):
    # img is RGB, HWC
    bayer = img.copy()
    
    bayer[0::2, 0::2, 0::2] = 0 # Green pixels - set the blue and the red planes to zero (and keep the green)
    bayer[0::2, 1::2, 1:] = 0   # Red pixels - set the blue and the green planes to zero (and keep the red)
    bayer[1::2, 0::2, :2] = 0   # Blue pixels - set the red and the green planes to zero (and keep the blue)
    bayer[1::2, 1::2, 0::2] = 0 # Green pixels - set the blue and the red planes to zero (and keep the green)

    return bayer


def interp_tensor(bayer):
    # img is RGB, BCHW
    interp_f32 = bayer.detach().clone()
    
    # Interpolate blue, last col and first row is all 0
    interp_f32[:, 2, 1::2, 1:-1:2] = (interp_f32[:, 2, 1::2, 0:-2:2] + interp_f32[:, 2, 1::2, 2::2]) / 2 # row
    interp_f32[:, 2, 1::2, -1] = interp_f32[:, 2, 1::2, -2] # last col
    interp_f32[:, 2, 2::2, :] = (interp_f32[:, 2, 1:-1:2, :] + interp_f32[:, 2, 3::2, :]) / 2 # col
    interp_f32[:, 2, 0, :] = interp_f32[:, 2, 1, :] # first row

    # Interpolate red, first col and last row is all 0
    interp_f32[:, 0, 0::2, 2::2] = (interp_f32[:, 0, 0::2, 1:-1:2] + interp_f32[:, 0, 0::2, 3::2]) / 2 # row
    interp_f32[:, 0, 0::2, 0] = interp_f32[:, 0, 0::2, 1] # first col
    interp_f32[:, 0, 1:-1:2, :] = (interp_f32[:, 0, 0:-2:2, :] + interp_f32[:, 0, 2::2, :]) / 2 # col
    interp_f32[:, 0, -1, :] = interp_f32[:, 0, -2, :] # last row

    # Interpolate green
    interp_f32[:, 1, 0, -1] = (interp_f32[:, 1, 0, -2] + interp_f32[:, 1, 1, -1]) / 2 # top right
    interp_f32[:, 1, -1, 0] = (interp_f32[:, 1, -2, 0] + interp_f32[:, 1, -1, 1]) / 2 # bottom left

    interp_f32[:, 1, 1:-1:2, 0] = (interp_f32[:, 1, 0:-2:2, 0] + interp_f32[:, 1, 2::2, 0] + 
                                   interp_f32[:, 1, 1:-1:2, 1]) / 3 # first col
    interp_f32[:, 1, 2::2, -1] = (interp_f32[:, 1, 1:-1:2, -1] + interp_f32[:, 1, 3::2, -1] + 
                                  interp_f32[:, 1, 2::2, -2]) / 3 # last col
    interp_f32[:, 1, 0, 1:-1:2] = (interp_f32[:, 1, 0, 0:-2:2] + interp_f32[:, 1, 0, 2::2] + 
                                   interp_f32[:, 1, 1, 1:-1:2]) / 3 # first row
    interp_f32[:, 1, -1, 2::2] = (interp_f32[:, 1, -1, 1:-1:2] + interp_f32[:, 1, -1, 3::2] + 
                                  interp_f32[:, 1, -2, 2::2]) / 3 # last row

    interp_f32[:, 1, 1:-1:2, 2::2] = (interp_f32[:, 1, 1:-1:2, 1:-1:2] + interp_f32[:, 1, 1:-1:2, 3::2] + 
                                      interp_f32[:, 1, 0:-2:2, 2::2] + interp_f32[:, 1, 2::2, 2::2]) / 4
    interp_f32[:, 1, 2::2, 1:-1:2] = (interp_f32[:, 1, 2::2, 0:-2:2] + interp_f32[:, 1, 2::2, 2::2] + 
                                      interp_f32[:, 1, 1:-1:2, 1:-1:2] + interp_f32[:, 1, 3::2, 1:-1:2]) / 4

    return interp_f32


def interp_ndarray(bayer):
    # img is RGB, HWC
    interp_f32 = bayer.copy()
    
    # Interpolate blue, last col and first row is all 0
    interp_f32[1::2, 1:-1:2, 2] = (interp_f32[1::2, 0:-2:2, 2] + interp_f32[1::2, 2::2, 2]) / 2 # row
    interp_f32[1::2, -1, 2] = interp_f32[1::2, -2, 2] # last col
    interp_f32[2::2, :, 2] = (interp_f32[1:-1:2, :, 2] + interp_f32[3::2, :, 2]) / 2 # col
    interp_f32[0, :, 2] = interp_f32[1, :, 2] # first row

    # Interpolate red, first col and last row is all 0
    interp_f32[0::2, 2::2, 0] = (interp_f32[0::2, 1:-1:2, 0] + interp_f32[0::2, 3::2, 0]) / 2 # row
    interp_f32[0::2, 0, 0] = interp_f32[0::2, 1, 0] # first col
    interp_f32[1:-1:2, :, 0] = (interp_f32[0:-2:2, :, 0] + interp_f32[2::2, :, 0]) / 2 # col
    interp_f32[-1, :, 0] = interp_f32[-2, :, 0] # last row

    # Interpolate green
    interp_f32[0, -1, 1] = (interp_f32[0, -2, 1] + interp_f32[1, -1, 1]) / 2 # top right
    interp_f32[-1, 0, 1] = (interp_f32[-2, 0, 1] + interp_f32[-1, 1, 1]) / 2 # bottom left

    interp_f32[1:-1:2, 0, 1] = (interp_f32[0:-2:2, 0, 1] + interp_f32[2::2, 0, 1] + 
                                interp_f32[1:-1:2, 1, 1]) / 3 # first col
    interp_f32[2::2, -1, 1] = (interp_f32[1:-1:2, -1, 1] + interp_f32[3::2, -1, 1] + 
                               interp_f32[2::2, -2, 1]) / 3 # last col
    interp_f32[0, 1:-1:2, 1] = (interp_f32[0, 0:-2:2, 1] + interp_f32[0, 2::2, 1] + 
                                interp_f32[1, 1:-1:2, 1]) / 3 # first row
    interp_f32[-1, 2::2, 1] = (interp_f32[-1, 1:-1:2, 1] + interp_f32[-1, 3::2, 1] + 
                               interp_f32[-2, 2::2, 1]) / 3 # last row

    interp_f32[1:-1:2, 2::2, 1] = (interp_f32[1:-1:2, 1:-1:2, 1] + interp_f32[1:-1:2, 3::2, 1] + 
                                   interp_f32[0:-2:2, 2::2, 1] + interp_f32[2::2, 2::2, 1]) / 4
    interp_f32[2::2, 1:-1:2, 1] = (interp_f32[2::2, 0:-2:2, 1] + interp_f32[2::2, 2::2, 1] + 
                                   interp_f32[1:-1:2, 1:-1:2, 1] + interp_f32[3::2, 1:-1:2, 1]) / 4

    return interp_f32
