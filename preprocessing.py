import cv2
import numpy as np

def bayer_filter(img_path):
    img = cv2.imread(img_path)
    (height, width) = img.shape[:2]
    (B,G,R) = cv2.split(img)

    bayer = np.empty((height, width), np.uint8)

    # Bayer Pattern
    # G R
    # B G
    bayer[0::2, 0::2] = G[0::2, 0::2] # top left
    bayer[0::2, 1::2] = R[0::2, 1::2] # top right
    bayer[1::2, 0::2] = B[1::2, 0::2] # bottom left
    bayer[1::2, 1::2] = G[1::2, 1::2] # bottom right

    bayer = cv2.cvtColor(bayer, cv2.COLOR_GRAY2BGR)  # Convert from Grayscale to BGR (r=g=b for each pixel).

    # Channel0 - B, Channel1 - G, Channel2 - R
    bayer[0::2, 0::2, 0::2] = 0 # Green pixels - set the blue and the red planes to zero (and keep the green)
    bayer[0::2, 1::2, :2] = 0   # Red pixels - set the blue and the green planes to zero (and keep the red)
    bayer[1::2, 0::2, 1:] = 0   # Blue pixels - set the red and the green planes to zero (and keep the blue)
    bayer[1::2, 1::2, 0::2] = 0 # Green pixels - set the blue and the red planes to zero (and keep the green)

    return bayer

def plot_channel(img):
    cv2.imshow('original', img)
    cv2.imshow('b', img[:, :, 0])
    cv2.imshow('g', img[:, :, 1])
    cv2.imshow('r', img[:, :, 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def interp(bayer):
    # plot_channel(bayer)
    interp_8bit = np.copy(np.array(bayer, dtype=np.uint8))
    interp_10bit = (interp_8bit.astype(np.uint16) * 4)
    (B,G,R) = cv2.split(interp_10bit)

    # Interpolate blue, last col and first row is all 0
    B = np.pad(B, 1)
    B[2::2, 2::2] = (np.array(B[2::2, 1:-1:2]) + np.array(B[2::2, 3::2])) / 2 # row
    B[2::2, -2] = B[2::2, -3] # last col
    B[1:-1:2, 1:-1] = (np.array(B[0:-2:2, 1:-1]) + np.array(B[2::2, 1:-1])) / 2 # col
    B[1, 1:-1] = B[2, 1:-1] # first row
    B = B[1:-1, 1:-1]

    # Interpolate red, first col and last row is all 0
    R = np.pad(R, 1)
    R[1:-1:2, 1:-1:2] = (np.array(R[1:-1:2, 0:-2:2]) + np.array(R[1:-1:2, 2::2])) / 2 # row
    R[1:-1:2, 1] = R[1:-1:2, 2] # first col
    R[2::2, 1:-1] = (np.array(R[1:-1:2, 1:-1]) + np.array(R[3::2, 1:-1])) / 2 # col
    R[-2, 1:-1] = R[-3, 1:-1] # last row
    R = R[1:-1, 1:-1]

    # Interpolate green
    G[0][-1] = (G[0][-2] + G[1][-1]) / 2
    G[-1][0] = (G[-2][0] + G[-1][1]) / 2

    G[1:-1:2, 0] = (np.array(G[0:-2:2, 0]) + np.array(G[2::2, 0]) + 
                    np.array(G[1:-1:2, 1])) / 3
    G[2::2, -1] = (np.array(G[1:-1:2, -1]) + np.array(G[3::2, -1]) + 
                   np.array(G[2::2, -2])) / 3
    G[0, 1:-1:2] = (np.array(G[0, 0:-2:2]) + np.array(G[0, 2::2]) + 
                   np.array(G[1, 1:-1:2])) / 3
    G[-1, 2::2] = (np.array(G[-1, 1:-1:2]) + np.array(G[-1, 3::2]) + 
                   np.array(G[-2, 2::2])) / 3

    G[1:-1:2, 2::2] = (np.array(G[1:-1:2, 1:-1:2]) + np.array(G[1:-1:2, 3::2]) + 
                       np.array(G[0:-2:2, 2::2]) + np.array(G[2::2, 2::2])) / 4
    G[2::2, 1:-1:2] = (np.array(G[2::2, 0:-2:2]) + np.array(G[2::2, 2::2]) + 
                       np.array(G[1:-1:2, 1:-1:2]) + np.array(G[3::2, 1:-1:2])) / 4

    interp_10bit[:, :, 0] = B
    interp_10bit[:, :, 1] = G
    interp_10bit[:, :, 2] = R
    interp_8bit = (interp_10bit // 4).astype(np.uint8)

    # plot_channel(interp_8bit)
    return interp_8bit
