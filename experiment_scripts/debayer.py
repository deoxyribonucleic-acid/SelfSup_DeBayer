import os

from preprocessing import bayer_filter, interp
import cv2
import numpy as np

import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from dataset.image_dataset import ImageDataset

from blind2unblind.model.arch_unet import UNet

from tqdm import tqdm

from PIL import Image

def get_generator(device):
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device=device)
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

class Masker(object):
    def __init__(self, width=4, mode='interpolate', mask_type='all'):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n,self.width**2,c,h,w), device=img.device)
        masks = torch.zeros((n,self.width**2,1,h,w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:,i,...] = x
            masks[:,i,...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks
    
def generate_mask(img, width=4, mask_type='random'):
    # This function generates random masks with shape (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask = torch.zeros(size=(n * h // width * w // width * width**2, ),
                       dtype=torch.int64,
                       device=img.device)
    idx_list = torch.arange(
        0, width**2, 1, dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(size=(n * h // width * w // width, ),
                         dtype=torch.int64,
                         device=img.device)

    if mask_type == 'random':
        torch.randint(low=0,
                      high=len(idx_list),
                      size=(n * h // width * w // width, ),
                      device=img.device,
                      generator=get_generator(device=img.device),
                      out=rd_idx)
    elif mask_type == 'batch':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(n, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(h // width * w // width)
    elif mask_type == 'all':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(1, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(n * h // width * w // width)
    elif 'fix' in mask_type:
        index = mask_type.split('_')[-1]
        index = torch.from_numpy(np.array(index).astype(
            np.int64)).type(torch.int64)
        rd_idx = index.repeat(n * h // width * w // width).to(img.device)

    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // width * w // width * width**2,
                                step=width**2,
                                dtype=torch.int64,
                                device=img.device)

    mask[rd_pair_idx] = 1

    mask = depth_to_space(mask.type_as(img).view(
        n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width).type(torch.int64)

    return mask

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)

def depth_to_space(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


def interpolate_mask(tensor, mask, mask_inv):
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1)

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv



def main():
    parser = argparse.ArgumentParser(description='Bayer filter interpolation')
    parser.add_argument('--clean_input', '-i', type=str, required=False, help='Path to clean image',default='data/validation/')
    parser.add_argument('--noisy_input', '-n', type=str, required=False, help='Path to noisy image, if not provided, the clean image will be bayar filtered and interpolated')
    parser.add_argument('--output', '-o', type=str, required=False, help='Path to output image', default='output/')

    parser.add_argument('--model', '-m', type=str, required=False, default='model.pth', help='Path to model')
    parser.add_argument('--beta', '-b', type=float, required=False, default=0.5, help='Beta value for the loss function')

    parser.add_argument('--batch_size', '-bs', type=int, required=False, default=1, help='Batch size for dataloader')
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

    beta = args.beta

    # load the model
    model = UNet()
    if args.model is not None:
        model.load_state_dict(torch.load(args.model))

    # Masker
    masker = Masker(width=4, mode='interpolate', mask_type='all')
    
    model.to(device)

    for sample, name in tqdm(dataloader):
        # move the sample to the device
        sample = sample.to(device)

        H = sample.shape[-2]
        W = sample.shape[-1]

        # forward pass
        with torch.no_grad():
                n, c, h, w = sample.shape
                net_input, mask = masker.train(sample)
                noisy_output = (model(net_input)*mask).view(n,-1,c,h,w).sum(dim=1)
                exp_output = model(sample)

        pred_dn = noisy_output[:, :, :H, :W]
        pred_exp = exp_output[:, :, :H, :W]
        pred_mid = (pred_dn + beta*pred_exp) / (1 + beta)

        pred_dn = pred_dn.permute(0, 2, 3, 1)
        pred_exp = pred_exp.permute(0, 2, 3, 1)
        pred_mid = pred_mid.permute(0, 2, 3, 1)

        pred_dn = pred_dn.cpu().data.clamp(0, 1).numpy().squeeze(0)
        pred_exp = pred_exp.cpu().data.clamp(0, 1).numpy().squeeze(0)
        pred_mid = pred_mid.cpu().data.clamp(0, 1).numpy().squeeze(0)

        pred255_dn = np.clip(pred_dn * 255.0 + 0.5, 0,
                                255).astype(np.uint8)
        pred255_exp = np.clip(pred_exp * 255.0 + 0.5, 0,
                                255).astype(np.uint8)
        pred255_mid = np.clip(pred_mid * 255.0 + 0.5, 0,
                                255).astype(np.uint8)      

        # save the output
        for fname, img in zip(name, pred255_mid):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(args.output, fname), img)
            print(f'Saved {fname} to {args.output}')

if __name__ == '__main__':
    main()