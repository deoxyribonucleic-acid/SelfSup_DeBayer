import cv2
import preprocessing as pp
from metrics import *
import os
import pandas as pd

path = 'data/validation/Kodak24/'

images = os.listdir(path)

results = pd.DataFrame(columns=['Image', 'PSNR', 'SSIM'])
for im_path in images:
    img = cv2.imread(os.path.join(path, im_path), cv2.IMREAD_COLOR)

    print('Processing image: ', im_path)

    bayer = pp.bayer_filter(img)

    rec = pp.interp(bayer)

    psnr = calculate_psnr(img, rec)
    ssim = calculate_ssim(img, rec)

    results = pd.concat([results, pd.DataFrame([{'Image': im_path, 'PSNR': psnr, 'SSIM': ssim}])], ignore_index=True)

results = pd.concat([results, pd.DataFrame([{'Image': 'Mean', 'PSNR': results['PSNR'].mean(), 'SSIM': results['SSIM'].mean()}])], ignore_index=True)
results.to_csv('results.csv', index=False)
print(results)


# cv2.imshow('original', img)
# cv2.imshow('bayer filter', bayer)
# cv2.imshow('reconstruction', rec)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
