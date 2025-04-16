import numpy as np
import torch


def bayer_filter(img, pattern="grbg"):
    # Determine if the input is a tensor or ndarray
    is_tensor = torch.is_tensor(img)
    bayer = img.clone() if is_tensor else img.copy()

    # Rotate the image to GRBG equivalent for uniform processing
    if pattern == "rggb":
        bayer = np.rot90(bayer, k=1, axes=(-2, -1)) if not is_tensor else torch.rot90(bayer, k=1, dims=(-2, -1))
    elif pattern == "bggr":
        bayer = np.rot90(bayer, k=2, axes=(-2, -1)) if not is_tensor else torch.rot90(bayer, k=2, dims=(-2, -1))
    elif pattern == "gbrg":
        bayer = np.rot90(bayer, k=3, axes=(-2, -1)) if not is_tensor else torch.rot90(bayer, k=3, dims=(-2, -1))

    # Apply GRBG Bayer filter
    if is_tensor:
        bayer[:, 0, 0::2, 0::2] = 0  # Green pixels
        bayer[:, 2, 0::2, 1::2] = 0  # Red pixels
        bayer[:, 1, 1::2, 0::2] = 0  # Blue pixels
        bayer[:, 0, 1::2, 1::2] = 0  # Green pixels
    else:
        bayer[0::2, 0::2, 0] = 0  # Green pixels
        bayer[0::2, 1::2, 2] = 0  # Red pixels
        bayer[1::2, 0::2, 1] = 0  # Blue pixels
        bayer[1::2, 1::2, 0] = 0  # Green pixels

    # Rotate back to the original Bayer pattern
    if pattern == "rggb":
        bayer = np.rot90(bayer, k=3, axes=(-2, -1)) if not is_tensor else torch.rot90(bayer, k=3, dims=(-2, -1))
    elif pattern == "bggr":
        bayer = np.rot90(bayer, k=2, axes=(-2, -1)) if not is_tensor else torch.rot90(bayer, k=2, dims=(-2, -1))
    elif pattern == "gbrg":
        bayer = np.rot90(bayer, k=1, axes=(-2, -1)) if not is_tensor else torch.rot90(bayer, k=1, dims=(-2, -1))

    return bayer

def interp_bayer(bayer, pattern="grbg"):
    # Rotate the Bayer pattern to GRBG equivalent for uniform processing
    if pattern == "rggb":
        bayer = np.rot90(bayer, k=1, axes=(-2, -1)) if not torch.is_tensor(bayer) else torch.rot90(bayer, k=1, dims=(-2, -1))
    elif pattern == "bggr":
        bayer = np.rot90(bayer, k=2, axes=(-2, -1)) if not torch.is_tensor(bayer) else torch.rot90(bayer, k=2, dims=(-2, -1))
    elif pattern == "gbrg":
        bayer = np.rot90(bayer, k=3, axes=(-2, -1)) if not torch.is_tensor(bayer) else torch.rot90(bayer, k=3, dims=(-2, -1))

    # Perform interpolation assuming GRBG pattern
    interp_f32 = bayer.clone() if torch.is_tensor(bayer) else bayer.copy()

    # Interpolate blue, last col and first row is all 0
    if torch.is_tensor(bayer):
        interp_f32[:, 2, 1::2, 1:-1:2] = (interp_f32[:, 2, 1::2, 0:-2:2] + interp_f32[:, 2, 1::2, 2::2]) / 2  # row
        interp_f32[:, 2, 1::2, -1] = interp_f32[:, 2, 1::2, -2]  # last col
        interp_f32[:, 2, 2::2, :] = (interp_f32[:, 2, 1:-1:2, :] + interp_f32[:, 2, 3::2, :]) / 2  # col
        interp_f32[:, 2, 0, :] = interp_f32[:, 2, 1, :]  # first row
    else:
        interp_f32[1::2, 1:-1:2, 2] = (interp_f32[1::2, 0:-2:2, 2] + interp_f32[1::2, 2::2, 2]) / 2  # row
        interp_f32[1::2, -1, 2] = interp_f32[1::2, -2, 2]  # last col
        interp_f32[2::2, :, 2] = (interp_f32[1:-1:2, :, 2] + interp_f32[3::2, :, 2]) / 2  # col
        interp_f32[0, :, 2] = interp_f32[1, :, 2]  # first row

    # Interpolate red, first col and last row is all 0
    if torch.is_tensor(bayer):
        interp_f32[:, 0, 0::2, 2::2] = (interp_f32[:, 0, 0::2, 1:-1:2] + interp_f32[:, 0, 0::2, 3::2]) / 2  # row
        interp_f32[:, 0, 0::2, 0] = interp_f32[:, 0, 0::2, 1]  # first col
        interp_f32[:, 0, 1:-1:2, :] = (interp_f32[:, 0, 0:-2:2, :] + interp_f32[:, 0, 2::2, :]) / 2  # col
        interp_f32[:, 0, -1, :] = interp_f32[:, 0, -2, :]  # last row
    else:
        interp_f32[0::2, 2::2, 0] = (interp_f32[0::2, 1:-1:2, 0] + interp_f32[0::2, 3::2, 0]) / 2  # row
        interp_f32[0::2, 0, 0] = interp_f32[0::2, 1, 0]  # first col
        interp_f32[1:-1:2, :, 0] = (interp_f32[0:-2:2, :, 0] + interp_f32[2::2, :, 0]) / 2  # col
        interp_f32[-1, :, 0] = interp_f32[-2, :, 0]  # last row

    # Interpolate green
    if torch.is_tensor(bayer):
        interp_f32[:, 1, 0, -1] = (interp_f32[:, 1, 0, -2] + interp_f32[:, 1, 1, -1]) / 2  # top right
        interp_f32[:, 1, -1, 0] = (interp_f32[:, 1, -2, 0] + interp_f32[:, 1, -1, 1]) / 2  # bottom left

        interp_f32[:, 1, 1:-1:2, 0] = (interp_f32[:, 1, 0:-2:2, 0] + interp_f32[:, 1, 2::2, 0] +
                                       interp_f32[:, 1, 1:-1:2, 1]) / 3  # first col
        interp_f32[:, 1, 2::2, -1] = (interp_f32[:, 1, 1:-1:2, -1] + interp_f32[:, 1, 3::2, -1] +
                                      interp_f32[:, 1, 2::2, -2]) / 3  # last col
        interp_f32[:, 1, 0, 1:-1:2] = (interp_f32[:, 1, 0, 0:-2:2] + interp_f32[:, 1, 0, 2::2] +
                                       interp_f32[:, 1, 1, 1:-1:2]) / 3  # first row
        interp_f32[:, 1, -1, 2::2] = (interp_f32[:, 1, -1, 1:-1:2] + interp_f32[:, 1, -1, 3::2] +
                                      interp_f32[:, 1, -2, 2::2]) / 3  # last row

        interp_f32[:, 1, 1:-1:2, 2::2] = (interp_f32[:, 1, 1:-1:2, 1:-1:2] + interp_f32[:, 1, 1:-1:2, 3::2] +
                                          interp_f32[:, 1, 0:-2:2, 2::2] + interp_f32[:, 1, 2::2, 2::2]) / 4
        interp_f32[:, 1, 2::2, 1:-1:2] = (interp_f32[:, 1, 2::2, 0:-2:2] + interp_f32[:, 1, 2::2, 2::2] +
                                          interp_f32[:, 1, 1:-1:2, 1:-1:2] + interp_f32[:, 1, 3::2, 1:-1:2]) / 4
    else:
        interp_f32[0, -1, 1] = (interp_f32[0, -2, 1] + interp_f32[1, -1, 1]) / 2  # top right
        interp_f32[-1, 0, 1] = (interp_f32[-2, 0, 1] + interp_f32[-1, 1, 1]) / 2  # bottom left

        interp_f32[1:-1:2, 0, 1] = (interp_f32[0:-2:2, 0, 1] + interp_f32[2::2, 0, 1] +
                                    interp_f32[1:-1:2, 1, 1]) / 3  # first col
        interp_f32[2::2, -1, 1] = (interp_f32[1:-1:2, -1, 1] + interp_f32[3::2, -1, 1] +
                                   interp_f32[2::2, -2, 1]) / 3  # last col
        interp_f32[0, 1:-1:2, 1] = (interp_f32[0, 0:-2:2, 1] + interp_f32[0, 2::2, 1] +
                                    interp_f32[1, 1:-1:2, 1]) / 3  # first row
        interp_f32[-1, 2::2, 1] = (interp_f32[-1, 1:-1:2, 1] + interp_f32[-1, 3::2, 1] +
                                   interp_f32[-2, 2::2, 1]) / 3  # last row

        interp_f32[1:-1:2, 2::2, 1] = (interp_f32[1:-1:2, 1:-1:2, 1] + interp_f32[1:-1:2, 3::2, 1] +
                                       interp_f32[0:-2:2, 2::2, 1] + interp_f32[2::2, 2::2, 1]) / 4
        interp_f32[2::2, 1:-1:2, 1] = (interp_f32[2::2, 0:-2:2, 1] + interp_f32[2::2, 2::2, 1] +
                                       interp_f32[1:-1:2, 1:-1:2, 1] + interp_f32[3::2, 1:-1:2, 1]) / 4

    # Rotate back to the original Bayer pattern
    if pattern == "rggb":
        interp_f32 = np.rot90(interp_f32, k=3, axes=(-2, -1)) if not torch.is_tensor(interp_f32) else torch.rot90(interp_f32, k=3, dims=(-2, -1))
    elif pattern == "bggr":
        interp_f32 = np.rot90(interp_f32, k=2, axes=(-2, -1)) if not torch.is_tensor(interp_f32) else torch.rot90(interp_f32, k=2, dims=(-2, -1))
    elif pattern == "gbrg":
        interp_f32 = np.rot90(interp_f32, k=1, axes=(-2, -1)) if not torch.is_tensor(interp_f32) else torch.rot90(interp_f32, k=1, dims=(-2, -1))

    return interp_f32
