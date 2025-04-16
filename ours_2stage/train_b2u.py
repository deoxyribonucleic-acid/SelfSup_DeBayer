from __future__ import division
import os
import logging
import time
import glob
import random
import datetime
import argparse
import numpy as np
from scipy.io import loadmat, savemat
from tqdm import tqdm  # Ensure tqdm is imported
from torch.utils.tensorboard import SummaryWriter

# Ours
import preprocessing as pp

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

from arch_unet import UNet
import utils as util
from collections import OrderedDict

parser = argparse.ArgumentParser()
# Ours
parser.add_argument("--noisetype", type=str, default="gauss25", choices=['gauss25', 'gauss5_50', 'poisson30', 'poisson5_50', 'bayer'])
parser.add_argument('--resume', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--data_dir', type=str,
                    default='./data/train/Imagenet_val')
parser.add_argument('--val_dirs', type=str, default='./data/validation')
parser.add_argument('--save_model_path', type=str,
                    default='../experiments/results')
parser.add_argument('--log_name', type=str,
                    default='b2u_unet_gauss25_112rf20')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--w_decay', type=float, default=1e-8)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--patchsize', type=int, default=128)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=2.0)
parser.add_argument("--increase_ratio", type=float, default=20.0)
parser.add_argument("--mask_type", type=str, default='random')
parser.add_argument("--warmup_epoch", type=int, default=100)
parser.add_argument("--remosaic_mode", type=str, default='single', choices=['single', 'random'])
parser.add_argument("--num_remosaic", type=int, default=2, help="Number of re-mosaic iterations in the second stage")

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0

# Set CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log device information
if torch.cuda.is_available():
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")

# Configure logging
opt.save_path = os.path.join(opt.save_model_path, opt.log_name, systime)
os.makedirs(opt.save_path, exist_ok=True)
util.setup_logger(
    "train",
    opt.save_path,
    "train_" + opt.log_name,
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("train")


def save_network(network, epoch, name):
    save_path = os.path.join(opt.save_path, 'models')
    os.makedirs(save_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_path = os.path.join(save_path, model_name)
    if isinstance(network, nn.DataParallel) or isinstance(
        network, nn.parallel.DistributedDataParallel
    ):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)
    logger.info('Checkpoint saved to {}'.format(save_path))


def load_network(load_path, network, strict=True):
    assert load_path is not None
    logger.info("Loading model from [{:s}] ...".format(load_path))
    if isinstance(network, nn.DataParallel) or isinstance(
        network, nn.parallel.DistributedDataParallel
    ):
        network = network.module
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith("module."):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)
    return network

def save_state(epoch, optimizer, scheduler):
    """Saves training state during training, which will be used for resuming"""
    save_path = os.path.join(opt.save_path, 'training_states')
    os.makedirs(save_path, exist_ok=True)
    state = {"epoch": epoch, "scheduler": scheduler.state_dict(), 
                                            "optimizer": optimizer.state_dict()}
    # Ours
    save_filename = "{:03d}.state".format(epoch)
    save_path = os.path.join(save_path, save_filename)
    torch.save(state, save_path)

def resume_state(load_path, optimizer, scheduler):
    """Resume the optimizers and schedulers for training"""
    resume_state = torch.load(load_path)
    epoch = resume_state["epoch"]
    resume_optimizer = resume_state["optimizer"]
    resume_scheduler = resume_state["scheduler"]
    optimizer.load_state_dict(resume_optimizer)
    scheduler.load_state_dict(resume_scheduler)
    return epoch, optimizer, scheduler

def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise(object):
    def __init__(self, style, bayer_pattern="grbg"):
        """
        Initialize the noise adder with a specific noise style and Bayer pattern.
        :param style: Noise style (e.g., gauss_fix, gauss_range, poisson_fix, poisson_range, bayer)
        :param bayer_pattern: Bayer pattern (e.g., rggb, grbg, bggr, gbrg)
        """
        print(f"Initializing AugmentNoise with style: {style}")
        self.bayer_pattern = bayer_pattern  # Default Bayer pattern is "grbg"
        self.style = style

        if style == "gauss_fix":
            self.params = [25.0 / 255.0]  # Default fixed Gaussian noise std
        elif style == "gauss_range":
            self.params = [5.0 / 255.0, 50.0 / 255.0]  # Default Gaussian noise range
        elif style == "poisson_fix":
            self.params = [30.0]  # Default fixed Poisson noise lambda
        elif style == "poisson_range":
            self.params = [5.0, 50.0]  # Default Poisson noise range
        elif style == "bayer":
            self.params = []  # No additional parameters for Bayer noise
        else:
            raise ValueError(f"Unsupported noise style: {style}")

    def add_train_noise(self, x, bayer_pattern=None):
        """
        Add noise to the input image during training.
        :param x: Input image tensor
        :param bayer_pattern: Bayer pattern to use (default is self.bayer_pattern)
        :return: Noisy image
        """
        if bayer_pattern is None:
            bayer_pattern = self.bayer_pattern

        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "bayer":
            bayer = pp.bayer_filter(x, pattern=bayer_pattern)
            return pp.interp_bayer(bayer, pattern=bayer_pattern)

    def add_valid_noise(self, x, bayer_pattern=None):
        """
        Add noise to the input image during validation.
        :param x: Input image numpy array
        :param bayer_pattern: Bayer pattern to use (default is self.bayer_pattern)
        :return: Noisy image
        """
        if bayer_pattern is None:
            bayer_pattern = self.bayer_pattern

        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "bayer":
            bayer = pp.bayer_filter(x, pattern=bayer_pattern)
            return pp.interp_bayer(bayer, pattern=bayer_pattern)


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def depth_to_space(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


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


def interpolate_mask(tensor, mask, mask_inv):
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1)

    return filtered.view_as(tensor) * mask + tensor * mask_inv


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
        tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
        masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=256):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # random crop
        H = im.shape[0]
        W = im.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]
        # np.ndarray to torch.tensor
        transformer = transforms.Compose([transforms.ToTensor()])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns)


class DataLoader_SIDD_Medium_Raw(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_SIDD_Medium_Raw, self).__init__()
        self.data_dir = data_dir
        # get images path
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = loadmat(fn)["x"]
        # random crop
        H, W = im.shape
        CSize = 256
        rnd_h = np.random.randint(0, max(0, H - CSize))
        rnd_w = np.random.randint(0, max(0, W - CSize))
        im = im[rnd_h : rnd_h + CSize, rnd_w : rnd_w + CSize]
        im = im[np.newaxis, :, :]
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns)


def get_SIDD_validation(dataset_dir):
    val_data_dict = loadmat(
        os.path.join(dataset_dir, "ValidationNoisyBlocksRaw.mat"))
    val_data_noisy = val_data_dict['ValidationNoisyBlocksRaw']
    val_data_dict = loadmat(
        os.path.join(dataset_dir, 'ValidationGtBlocksRaw.mat'))
    val_data_gt = val_data_dict['ValidationGtBlocksRaw']
    num_img, num_block, _, _ = val_data_gt.shape
    return num_img, num_block, val_data_noisy, val_data_gt


def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_bsd300(dataset_dir):
    fns = []
    # Ours
    fns.extend(glob.glob(os.path.join(dataset_dir, "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # Ours
        if opt.noisetype == "bayer":
            im = im[:-1, :-1, :] # Crop images to even rows and cols
        images.append(im)
    return images


def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # Ours
        if opt.noisetype == "bayer":
            (H, W) = im.shape[:2]
            im = im[:H-(H%2), :W-(W%2), :] # Crop images to even rows and cols
        images.append(im)
    return images


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref, data_range=255.0):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(data_range**2 / np.mean(np.square(diff)))
    return psnr


def train_one_epoch(network, optimizer, scheduler, masker, noise_adder, TrainingLoader, epoch, opt, logger, device, writer):
    """Train the network for one epoch."""
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        logger.info(f"LearningRate of Epoch {epoch} = {current_lr}")
        writer.add_scalar("Learning Rate/Stage1", current_lr, epoch)

    network.train()
    progress_bar = tqdm(enumerate(TrainingLoader), total=len(TrainingLoader), desc=f"Epoch {epoch} | Loss: N/A", unit="batch")
    for iteration, clean in progress_bar:
        st = time.time()
        clean = clean / 255.0
        clean = clean.to(device)  # Move data to the selected device
        noisy = noise_adder.add_train_noise(clean)

        optimizer.zero_grad()

        net_input, mask = masker.train(noisy)
        net_input = net_input.to(device)  # Move data to the selected device
        noisy_output = network(net_input)

        n, c, h, w = noisy.shape
        noisy_output = (noisy_output * mask).view(n, -1, c, h, w).sum(dim=1)
        diff = noisy_output - noisy

        with torch.no_grad():
            exp_output = network(noisy)
        exp_diff = exp_output - noisy

        Lambda = epoch / opt.n_epoch
        if Lambda <= Thread1:
            beta = Lambda2
        elif Thread1 <= Lambda <= Thread2:
            beta = Lambda2 + (Lambda - Thread1) * (increase_ratio - Lambda2) / (Thread2 - Thread1)
        else:
            beta = increase_ratio
        alpha = Lambda1

        revisible = diff + beta * exp_diff
        loss_reg = alpha * torch.mean(diff**2)
        loss_rev = torch.mean(revisible**2)
        loss_all = loss_reg + loss_rev

        loss_all.backward()
        optimizer.step()

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Stage1/Total", loss_all.item(), epoch * len(TrainingLoader) + iteration)
        writer.add_scalar("Loss/Stage1/Regularization", loss_reg.item(), epoch * len(TrainingLoader) + iteration)
        writer.add_scalar("Loss/Stage1/Reversible", loss_rev.item(), epoch * len(TrainingLoader) + iteration)

        # Update progress bar description with the current loss
        progress_bar.set_description(f"Epoch {epoch} | Loss: {loss_all.item():.6f}")

    else:
        # Log training progress in the console at the end of the epoch
        logger.info(
            f"Epoch [{epoch}] Iteration [{iteration}/{len(TrainingLoader)}]: "
            f"diff={torch.mean(diff**2).item():.6f}, exp_diff={torch.mean(exp_diff**2).item():.6f}, "
            f"Loss_Reg={loss_reg.item():.6f}, Lambda={Lambda}, Loss_Rev={loss_rev.item():.6f}, "
            f"Loss_All={loss_all.item():.6f}, Time={time.time() - st:.4f}"
        )

    # Step the scheduler
    scheduler.step()


def train_second_stage(network, optimizer, scheduler, masker, noise_adder, TrainingLoader, epoch, opt, logger, device, writer):
    """Train the network for the second stage."""
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    logger.info(f"LearningRate of Epoch {epoch} = {current_lr}")
    writer.add_scalar("Learning Rate/Stage2", current_lr, epoch)

    network.train()
    progress_bar = tqdm(enumerate(TrainingLoader), total=len(TrainingLoader), desc=f"Epoch {epoch} | Loss: N/A", unit="batch")
    for iteration, I in progress_bar:
        optimizer.zero_grad()

        # Convert input to device
        I = I.to(device)  # [N, C, H, W]
        I = I / 255.0  # Normalize input to [0, 1]

        # Standard GRGB Pattern random filter retrieval
        bayer_grbg = pp.bayer_filter(I, pattern='grbg')

        # Initially set pred_rgb to the input image
        pred_rgb = I

        # Repeat re-mosaic process for a specified number of times
        num_remosaic = opt.num_remosaic  # Number of re-mosaic iterations
        remosaiced_bayer = None
        loss = 0

        # Mask out the boundary pixels
        valid_mask = torch.ones_like(bayer_grbg)
        valid_mask[:, :, :1, :] = 0  # Top boundary
        valid_mask[:, :, -1:, :] = 0  # Bottom boundary
        valid_mask[:, :, :, :1] = 0  # Left boundary
        valid_mask[:, :, :, -1:] = 0  # Right boundary

        
        for _ in range(num_remosaic):
            # Select re-mosaic pattern
            if opt.remosaic_mode == 'single':
                pattern = 'rggb'  # Fixed pattern, change as needed
            elif opt.remosaic_mode == 'random':
                pattern = random.choice(['grbg','rggb', 'bggr', 'gbrg'])  # Random pattern

            # Remosaic the predicted RGB output
            remosaiced_bayer = pp.bayer_filter(pred_rgb, pattern=pattern)

            # Interpolate the remosaiced Bayer image
            # with torch.no_grad():
            remosaiced_bayer = pp.interp_bayer(remosaiced_bayer, pattern=pattern)

            # Use no_grad for exp_output, following stage 1
            with torch.no_grad():
                exp_output = network(remosaiced_bayer)

            net_input, mask = masker.train(remosaiced_bayer)

            n, c, h, w = remosaiced_bayer.shape
            noisy_output = network(net_input)
            noisy_output = (noisy_output * mask).view(n, -1, c, h, w).sum(dim=1)

            # Calculate beta based on the epoch
            Lambda = epoch / opt.n_epoch
            if Lambda <= Thread1:
                beta = Lambda2
            elif Thread1 <= Lambda <= Thread2:
                beta = Lambda2 + (Lambda - Thread1) * (increase_ratio - Lambda2) / (Thread2 - Thread1)
            else:
                beta = increase_ratio

            pred_dn = noisy_output[:, :, :h, :w]
            pred_exp = exp_output.clone()[:, :, :h, :w]

            pred_rgb = (pred_dn + beta * pred_exp) / (1 + beta)

            # print(pred_rgb.requires_grad)

            # Finally re-mosaic the predicted RGB output
            remosaiced_bayer = pp.bayer_filter(pred_rgb, pattern='grbg')

            # Compute loss between original and remosaiced Bayer images
            # loss = F.mse_loss(I, pred_rgb)
            loss += F.mse_loss(remosaiced_bayer * valid_mask, bayer_grbg * valid_mask)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Stage2", loss.item(), epoch * len(TrainingLoader) + iteration)

        # Update progress bar description with the current loss
        progress_bar.set_description(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    else:
        # Log training progress in the console at the end of the epoch
        logger.info(
            f"Epoch [{epoch}] Iteration [{iteration}/{len(TrainingLoader)}]: Loss: {loss.item():.6f}"
        )

    # Step the scheduler
    scheduler.step()


def validate(network, masker, valid_dict, epoch, opt, systime, logger, device, noise_adder):
    """Validate the network on the validation datasets."""
    network.eval()
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    validation_path = os.path.join(save_model_path, "validation")
    os.makedirs(validation_path, exist_ok=True)
    np.random.seed(101)
    valid_repeat_times = {"Kodak24": 10, "BSD300": 3, "Set14": 20}

    logger.info("Validation started for epoch {}".format(epoch))

    # Calculate beta based on the epoch
    Lambda = epoch / opt.n_epoch
    if Lambda <= Thread1:
        beta = Lambda2
    elif Thread1 <= Lambda <= Thread2:
        beta = Lambda2 + (Lambda - Thread1) * (increase_ratio - Lambda2) / (Thread2 - Thread1)
    else:
        beta = increase_ratio

    for valid_name, valid_images in valid_dict.items():
        if opt.noisetype == "bayer":
            valid_repeat_times = {"Kodak24": 1, "BSD300": 1, "Set14": 1}
            avg_psnr_interp = []
            avg_ssim_interp = []
        avg_psnr_dn = []
        avg_ssim_dn = []
        avg_psnr_exp = []
        avg_ssim_exp = []
        avg_psnr_mid = []
        avg_ssim_mid = []
        save_dir = os.path.join(validation_path, valid_name)
        os.makedirs(save_dir, exist_ok=True)
        repeat_times = valid_repeat_times[valid_name]
        for i in range(repeat_times):
            for idx, im in enumerate(valid_images):
                origin255 = im.copy()
                origin255 = origin255.astype(np.uint8)
                im = np.array(im, dtype=np.float32) / 255.0
                noisy_im = pp.interp_bayer(pp.bayer_filter(im)) if opt.noisetype == "bayer" else noise_adder.add_valid_noise(im)

                # Padding to square
                H, W = noisy_im.shape[:2]
                val_size = (max(H, W) + 31) // 32 * 32
                noisy_im = np.pad(noisy_im, [[0, val_size - H], [0, val_size - W], [0, 0]], 'reflect')
                transformer = transforms.Compose([transforms.ToTensor()])
                noisy_im = transformer(noisy_im)
                noisy_im = torch.unsqueeze(noisy_im, 0).to(device)  # Move data to the selected device

                with torch.no_grad():
                    n, c, h, w = noisy_im.shape
                    net_input, mask = masker.train(noisy_im)
                    net_input = net_input.to(device)  # Move data to the selected device
                    noisy_output = (network(net_input) * mask).view(n, -1, c, h, w).sum(dim=1)
                    dn_output = noisy_output.detach().clone()
                    del net_input, mask, noisy_output
                    torch.cuda.empty_cache()
                    exp_output = network(noisy_im)

                pred_dn = dn_output[:, :, :H, :W]
                pred_exp = exp_output.detach().clone()[:, :, :H, :W]
                pred_mid = (pred_dn + beta * pred_exp) / (1 + beta)

                del exp_output
                torch.cuda.empty_cache()

                pred_dn = pred_dn.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
                pred_exp = pred_exp.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
                pred_mid = pred_mid.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)

                pred255_dn = np.clip(pred_dn * 255.0 + 0.5, 0, 255).astype(np.uint8)
                pred255_exp = np.clip(pred_exp * 255.0 + 0.5, 0, 255).astype(np.uint8)
                pred255_mid = np.clip(pred_mid * 255.0 + 0.5, 0, 255).astype(np.uint8)

                psnr_dn = calculate_psnr(origin255.astype(np.float32), pred255_dn.astype(np.float32))
                avg_psnr_dn.append(psnr_dn)
                ssim_dn = calculate_ssim(origin255.astype(np.float32), pred255_dn.astype(np.float32))
                avg_ssim_dn.append(ssim_dn)

                psnr_exp = calculate_psnr(origin255.astype(np.float32), pred255_exp.astype(np.float32))
                avg_psnr_exp.append(psnr_exp)
                ssim_exp = calculate_ssim(origin255.astype(np.float32), pred255_exp.astype(np.float32))
                avg_ssim_exp.append(ssim_exp)

                psnr_mid = calculate_psnr(origin255.astype(np.float32), pred255_mid.astype(np.float32))
                avg_psnr_mid.append(psnr_mid)
                ssim_mid = calculate_ssim(origin255.astype(np.float32), pred255_mid.astype(np.float32))
                avg_ssim_mid.append(ssim_mid)

        avg_psnr_dn = np.mean(avg_psnr_dn)
        avg_ssim_dn = np.mean(avg_ssim_dn)
        avg_psnr_exp = np.mean(avg_psnr_exp)
        avg_ssim_exp = np.mean(avg_ssim_exp)
        avg_psnr_mid = np.mean(avg_psnr_mid)
        avg_ssim_mid = np.mean(avg_ssim_mid)

        log_path = os.path.join(validation_path, f"A_log_{valid_name}.csv")
        with open(log_path, "a") as f:
            f.writelines(f"epoch:{epoch},dn:{avg_psnr_dn:.6f}/{avg_ssim_dn:.6f},"
                         f"exp:{avg_psnr_exp:.6f}/{avg_ssim_exp:.6f},"
                         f"mid:{avg_psnr_mid:.6f}/{avg_ssim_mid:.6f}\n")
            
        logger.info(f"Validation for {valid_name} at epoch {epoch}: "
                    f"DN PSNR: {avg_psnr_dn:.6f}, DN SSIM: {avg_ssim_dn:.6f}, "
                    f"EXP PSNR: {avg_psnr_exp:.6f}, EXP SSIM: {avg_ssim_exp:.6f}, "
                    f"MID PSNR: {avg_psnr_mid:.6f}, MID SSIM: {avg_ssim_mid:.6f}")


def main():
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(opt.save_path, "tensorboard_logs"))

    # Training Set
    TrainingDataset = DataLoader_Imagenet_val(opt.data_dir, patch=opt.patchsize)
    TrainingLoader = DataLoader(dataset=TrainingDataset,
                                num_workers=0,
                                batch_size=opt.batchsize,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)

    # Validation Set
    Kodak_dir = os.path.join(opt.val_dirs, "Kodak24")
    BSD300_dir = os.path.join(opt.val_dirs, "BSD300")
    Set14_dir = os.path.join(opt.val_dirs, "Set14")
    valid_dict = {
        "Kodak24": validation_kodak(Kodak_dir),
        "BSD300": validation_bsd300(BSD300_dir),
        "Set14": validation_Set14(Set14_dir)
    }

    # Noise adder
    noise_adder = AugmentNoise(style=opt.noisetype)

    # Masker
    masker = Masker(width=4, mode='interpolate', mask_type='all')

    # Network
    network = UNet(in_channels=opt.n_channel,
                    out_channels=opt.n_channel,
                    wf=opt.n_feature)
    if opt.parallel and torch.cuda.device_count() > 1:
        network = torch.nn.DataParallel(network)
    network = network.to(device)

    # Training scheme
    num_epoch = opt.n_epoch
    ratio = num_epoch / 100
    optimizer = optim.Adam(network.parameters(), lr=opt.lr,
                        weight_decay=opt.w_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                        milestones=[
                                            int(20 * ratio) - 1,
                                            int(40 * ratio) - 1,
                                            int(60 * ratio) - 1,
                                            int(80 * ratio) - 1
                                        ],
                                        gamma=opt.gamma)
    print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

    # Resume and load pre-trained model
    epoch_init = 1
    if opt.resume is not None:
        epoch_init, optimizer, scheduler = resume_state(opt.resume, optimizer, scheduler)
    if opt.checkpoint is not None:
        network = load_network(opt.checkpoint, network, strict=True)

    # Temp
    if opt.checkpoint is not None:
        epoch_init = 60
        for i in range(1, epoch_init):
            scheduler.step() # Although Torch gives a Warning about usage of scheduler.step() before optimizer.step(), we need to do this to get the correct learning rate
            new_lr = scheduler.get_lr()[0]
        logger.info('----------------------------------------------------')
        logger.info("==> Resuming Training with learning rate:{}".format(new_lr))
        logger.info('----------------------------------------------------')

    print('init finish')

    if opt.noisetype in ['gauss25', 'poisson30']:
        global Thread1, Thread2
        Thread1 = 0.8
        Thread2 = 1.0
    else:
        Thread1 = 0.4
        Thread2 = 1.0

    global Lambda1, Lambda2, increase_ratio
    Lambda1 = opt.Lambda1
    Lambda2 = opt.Lambda2
    increase_ratio = opt.increase_ratio

    # First stage training
    for epoch in range(epoch_init, opt.warmup_epoch + 1):
        train_one_epoch(network, optimizer, scheduler, masker, noise_adder, TrainingLoader, epoch, opt, logger, device, writer)
        if epoch % opt.n_snapshot == 0 or epoch == opt.warmup_epoch:
            save_network(network, epoch, "model_stage1")
            save_state(epoch, optimizer, scheduler)
            validate(network, masker, valid_dict, epoch, opt, systime, logger, device, noise_adder)

    # Adjust parameters for the second stage
    logger.info("Starting second stage training...")
    opt.lr = opt.lr / 10  # Example: Reduce learning rate for fine-tuning
    optimizer = optim.Adam(network.parameters(), lr=opt.lr, weight_decay=opt.w_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                        milestones=[
                                            int(20 * ratio) - 1,
                                            int(40 * ratio) - 1,
                                            int(60 * ratio) - 1,
                                            int(80 * ratio) - 1
                                        ],
                                        gamma=opt.gamma)

    # Second stage training
    for epoch in range(max(epoch_init, opt.warmup_epoch + 1), opt.n_epoch + 1):
        train_second_stage(network, optimizer, scheduler, masker, noise_adder, TrainingLoader, epoch, opt, logger, device, writer)
        if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
            save_network(network, epoch, "model_stage2")
            save_state(epoch, optimizer, scheduler)
            validate(network, masker, valid_dict, epoch, opt, systime, logger, device, noise_adder)


if __name__ == "__main__":
    main()
