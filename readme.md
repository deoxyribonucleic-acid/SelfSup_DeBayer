# Deep Image Demosaicing via Self-Supervised Denoiser
[Ruijie Chen](https://deoxyribonucleic-acid.github.io/), [Haotian Hao](), [Xianchao Wang](), [Shufeng Yin]()

Project for UMich ECE 556 Image Processing WN 2025

## Credit
This project borrows heavily on [Blind2Unblind](https://github.com/zejinwang/Blind2Unblind).

## Get Started
The model is built with Python 3.8.5, running on Linux.

Please note that the working directory should be : **./ours**

- For preparing ImageNet Validation Dataset, please run the command
  ```shell
  python pick_img.py -s /your_dataset_dir -d /dst_dir
  ```
- For installing the required packages, please run the command
  ```shell
  pip install -r requirements.txt
  ```

## Pre-trained Models
Download the pre-trained models: [Google Drive](https://drive.google.com/drive/folders/1-aOvdfhDLeLomSSEyLt9dJ3Ynb8c7Ixa?usp=drive_link)

Please note that the pre-trained models should be placed in the folder: **./ours/pretrained_models**

```yaml
# For HQ + Denoise + Restore
./pretrained_models/hq_denoise_restore.pth
# For HQ + Denoise
./pretrained_models/hq_denoise.pth
# For Bilinear + Denoise
./pretrained_models/bilinear_denoise.pth
```

## Train
- For high-quality interpolation method
  ```shell
  python train_ours.py --noisetype hq --pvr --data_dir /your_dataset_dir --val_dirs /your_valset_dirs --save_model_path ../experiments/results --log_name hq_denoise_restore --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0
  ```
- For bilinear interpolation method
  ```shell
  python train_ours.py --noisetype bilinear --data_dir /your_dataset_dir --val_dirs /your_valset_dirs --save_model_path ../experiments/results --log_name bilinear_denoise --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0
  ```
- For ablation study on pixel-wise value restoration, please do not include the argument `--pvr`

## Test
- For high-quality interpolation method
  ```shell
  python test_ours.py --noisetype hq --pvr --checkpoint ./pretrained_models/hq_denoise_restore.pth --test_dirs /your_testset_dirs --save_test_path ./test --log_name hq_denoise_restore
  ```
- For bilinear interpolation method
  ```shell
  python test_ours.py --noisetype bilinear --checkpoint ./pretrained_models/bilinear_denoise.pth --test_dirs /your_testset_dirs --save_test_path ./test --log_name bilinear_denoise
  ```
