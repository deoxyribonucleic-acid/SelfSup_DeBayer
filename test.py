import cv2
import preprocessing as pp
import metrics as mt

path = 'data/validation/Kodak24/kodim01.png'

img = cv2.imread(path)

bayer = pp.bayer_filter(img)

rec = pp.interp(bayer)

psnr = mt.PSNR(img, rec)
print(f"PSNR value is {psnr} dB")
ssim = mt.SSIM(img, rec)
print(f"SSIM value is {ssim}")

cv2.imshow('original', img)
cv2.imshow('bayer filter', bayer)
cv2.imshow('reconstruction', rec)
cv2.waitKey(0)
cv2.destroyAllWindows()
