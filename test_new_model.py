import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from train_new import UNet, get_SIDD_validation, calculate_psnr, calculate_ssim, BayerPatternShifter, bilinear_demosaic
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_SIDD_validation(dataset_dir):
    val_data_dict = loadmat(
        os.path.join(dataset_dir, "ValidationNoisyBlocksRaw.mat"))
    val_data_noisy = val_data_dict['ValidationNoisyBlocksRaw']
    val_data_dict = loadmat(
        os.path.join(dataset_dir, 'ValidationGtBlocksSrgb.mat'))
    val_data_gt = val_data_dict['ValidationGtBlocksSrgb']
    # print(val_data_gt.shape)
    num_img, num_block, _, _, _ = val_data_gt.shape
    return num_img, num_block, val_data_noisy, val_data_gt


def test(model_path, data_dir, out_dir):
    model = UNet(in_channels=4, out_channels=3, wf=48).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    num_img, num_block, val_noisy, val_gt = get_SIDD_validation(data_dir)
    os.makedirs(out_dir, exist_ok=True)

    psnr_list, ssim_list = [], []

    for idx in tqdm(range(num_img), desc="Testing", unit="image"):
        for idy in tqdm(range(num_block), desc="Processing blocks", unit="block"):
            gt = val_gt[idx, idy] / 255.0  # [H, W, 3]
            # print(gt.shape)
            noisy = val_noisy[idx, idy][:, :, np.newaxis]  # [H, W, 1]

            transformer = transforms.Compose([transforms.ToTensor()])
            noisy_tensor = transformer(noisy).unsqueeze(0).to(device)
            noisy_tensor = BayerPatternShifter.bayer_1ch_to_4ch(noisy_tensor)

            with torch.no_grad():
                pred_rgb = model(noisy_tensor)

            pred_rgb = pred_rgb.permute(0, 2, 3, 1).cpu().clamp(0, 1).numpy().squeeze(0)
            pred255 = np.clip(pred_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

            psnr = calculate_psnr(gt.astype(np.float32), pred_rgb.astype(np.float32), 1.0)
            ssim = calculate_ssim(gt * 255.0, pred_rgb * 255.0)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            save_path = os.path.join(out_dir, f"val_{idx:03d}_{idy:03d}.png")
            Image.fromarray(pred255).save(save_path)

    print(f"Average PSNR: {np.mean(psnr_list):.4f}, SSIM: {np.mean(ssim_list):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to validation data folder')
    parser.add_argument('--out_dir', type=str, default='./test_results', help='Directory to save predictions')
    args = parser.parse_args()

    # test(args.checkpoint, args.test_dir, args.out_dir)

    test_im = Image.open('./data/validation/koidim01.png')
    shifter = BayerPatternShifter()
    test_im = test_im.convert('RGB')
    test_im = np.array(test_im)
    bayer_f = shifter.remosaic(test_im, 'RGGB', out_channel=4)
    simple_demosaic = bilinear_demosaic(bayer_f)
    
