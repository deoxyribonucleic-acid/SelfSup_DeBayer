import os

from preprocessing import bayer_filter, interp
import cv2
import metrics as mt

import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from dataset.image_dataset import ImageDataset

from blind2unblind.model.arch_unet import UNet

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Bayer filter interpolation')
    parser.add_argument('--clean_input', '-i', type=str, required=False, help='Path to clean image',default='data/validation/')
    parser.add_argument('--noisy_input', '-n', type=str, required=False, help='Path to noisy image, if not provided, the clean image will be bayar filtered and interpolated')
    parser.add_argument('--output', '-o', type=str, required=False, help='Path to output image', default='output/')

    parser.add_argument('--model', '-m', type=str, required=False, default='model.pth', help='Path to model')

    parser.add_argument('--batch_size', '-b', type=int, required=False, default=1, help='Batch size for dataloader')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Force CPU usage, else GPU will be used if available')
    args = parser.parse_args()

    # get the paths
    assert args.clean_input is not None or args.noisy_input is not None, 'Either clean or noisy input must be provided'
    input = args.clean_input if args.noisy_input is None else args.noisy_input
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    # get the device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    device = torch.device('mps' if torch.cuda.is_available() and not args.force_cpu else 'cpu')

    # create a dataloader
    dataset = ImageDataset(input, is_clean= args.noisy_input is None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # load the model
    model = UNet()
    if args.model is not None:
        model.load_state_dict(torch.load(args.model))
    
    model.to(device)

    for sample in tqdm(dataloader):
        # move the sample to the device
        sample = sample.to(device)

        # forward pass
        output = model(sample)

        # save the output
        cv2.imwrite(os.path.join(args.output, 'output.png'), output)

if __name__ == '__main__':
    main()