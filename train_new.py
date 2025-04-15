from __future__ import division
import os
import random
import logging
import glob
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat, savemat

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from blind2unblind.model.arch_unet import UNet
import blind2unblind.model.utils as util
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--data_dir', type=str, default='./data/train/SIDD_Medium_Raw_noisy_sub512')
parser.add_argument('--val_dirs', type=str, default='./data/validation/SIDD_Validation_Blocks')
parser.add_argument('--save_model_path', type=str, default='../experiments/results')
parser.add_argument('--log_name', type=str, default='b2u_new_mask')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--in_channel', type=int, default=4)
parser.add_argument('--out_channel', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--w_decay', type=float, default=1e-8)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--use_mask', action='store_true', help='use edge mask or not')
parser.add_argument('--remosaic_mode', type=str, default='random,', choices=['single', 'random'], help='remosaic pattern')
parser.add_argument('--warmup_epoch', type=int, default=20)

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
torch.set_num_threads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config loggers. Before it, the log will not work
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
    save_filename = "{}.state".format(epoch)
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
    logger.info("Resume training from epoch {}".format(epoch))
    return epoch, optimizer, scheduler

def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))

class BayerPatternShifter:

    def __init__(self, pattern='RGGB'):
        self.pattern = pattern
        self.offset_map = {
            'RGGB': (0, 0),
            'GRBG': (0, 1),
            'GBRG': (1, 0),
            'BGGR': (1, 1)
        }

    def shift_bayer(self, bayer_img, target_pattern):
        
        dx, dy = self.offset_map[target_pattern]
        return bayer_img[..., dx:, dy:]

    def inverse_shift(self, shifted_img, target_pattern):

        dx, dy = self.offset_map[target_pattern]
        B, H, W = shifted_img.shape
        out = torch.zeros((B, H + dx, W + dy), device=shifted_img.device)
        out[..., dx:, dy:] = shifted_img
        return out

    def remosaic(self, rgb_img, target_pattern, out_channel=1):
        """
        Converts an RGB image into a Bayer image (implements P_i).
        Input: rgb_img: (3, H, W)
        Output: bayer_img: (1, H, W)
        """
        R = rgb_img[0]
        G = rgb_img[1]
        B = rgb_img[2]
        H, W = R.shape
        bayer = torch.zeros((out_channel, H, W), device=rgb_img.device)

        dx, dy = self.offset_map[target_pattern]

        if out_channel == 1:
            bayer[0, dx::2, dy::2] = R[dx::2, dy::2]
            bayer[0, dx::2, 1-dy::2] = G[dx::2, 1-dy::2]
            bayer[0, 1-dx::2, dy::2] = G[1-dx::2, dy::2]
            bayer[0, 1-dx::2, 1-dy::2] = B[1-dx::2, 1-dy::2]

        elif out_channel == 4:
            bayer[0, dx::2, dy::2] = R[dx::2, dy::2]
            bayer[1, dx::2, 1-dy::2] = G[dx::2, 1-dy::2]
            bayer[2, 1-dx::2, dy::2] = G[1-dx::2, dy::2]
            bayer[3, 1-dx::2, 1-dy::2] = B[1-dx::2, 1-dy::2]

        else:
            raise ValueError("out_channel must be 1 or 4")

        return bayer
    
    @staticmethod
    def bayer_1ch_to_4ch(x):

        B, _, H, W = x.shape
        out = torch.zeros(B, 4, H, W, device=x.device)
        out[:, 0, 0::2, 0::2] = x[:, 0, 0::2, 0::2]
        out[:, 1, 0::2, 1::2] = x[:, 0, 0::2, 1::2]
        out[:, 2, 1::2, 0::2] = x[:, 0, 1::2, 0::2]
        out[:, 3, 1::2, 1::2] = x[:, 0, 1::2, 1::2]

        return out

def bilinear_demosaic_rggb(bayer):
    B, H, W = bayer.shape
    print(bayer.shape)
    
    # 初始化 RGB 通道
    R = torch.zeros_like(bayer)
    G = torch.zeros_like(bayer)
    B_ = torch.zeros_like(bayer)

    # 从 Bayer pattern 中提取通道分布
    R[:, 0::2, 0::2] = bayer[:, 0::2, 0::2]
    G[:, 0::2, 1::2] = bayer[:, 0::2, 1::2]
    G[:, 1::2, 0::2] = bayer[:, 1::2, 0::2]
    B_[:, 1::2, 1::2] = bayer[:, 1::2, 1::2]

    # 插值空位置
    R = F.interpolate(R.unsqueeze(1), scale_factor=1, mode='bilinear', align_corners=False)
    G = F.interpolate(G.unsqueeze(1), scale_factor=1, mode='bilinear', align_corners=False)
    B_ = F.interpolate(B_.unsqueeze(1), scale_factor=1, mode='bilinear', align_corners=False)

    # 拼接成 RGB
    rgb = torch.cat([R, G, B_], dim=1)  # [B, 3, H, W]
    return rgb

# def space_to_depth(x, block_size):
#     n, c, h, w = x.size()
#     unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
#     return unfolded_x.view(n, c * block_size**2, h // block_size,
#                            w // block_size)

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


def train(network, optimizer, scheduler, TrainingLoader, epoch_init, num_epoch, opt):
    """Training loop for the model."""
    for epoch in range(epoch_init, num_epoch + 1):
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

        network.train()
        for iteration, I in tqdm(enumerate(TrainingLoader), total=len(TrainingLoader)):
            optimizer.zero_grad()
            shifter = BayerPatternShifter()

            I = I.to(device)  # [N,1,H,W] 
            I_4ch = shifter.bayer_1ch_to_4ch(I)  # [N,4,H,W] 
            pred_rgb = network(I_4ch)

            # warm up stage: learn to remosaic
            if epoch <= opt.warmup_epoch:
                target_rgb = bilinear_demosaic_rggb(I) # I must in RGGB format, 1 channel
                loss = F.mse_loss(pred_rgb, target_rgb)
            else:
               # Self-supervised remosaic stage
                if opt.remosaic_mode == 'single':
                    pat = 'GRBG'
                elif opt.remosaic_mode == 'random':
                    pat = random.choice(['GRBG', 'GBRG', 'BGGR'])

                I_i = [shifter.remosaic(rgb, pat, 4) for rgb in pred_rgb] # remosaic model output
                I_i = torch.stack(I_i)

                # re-predict
                I_D_tilde = network(I_i)
                
                # back to RGGB
                I_hat = [shifter.remosaic(rgb, 'RGGB', 1) for rgb in I_D_tilde] # size [N,1,H,W], dense Bayer F
                I_hat = torch.stack(I_hat)
                
                if opt.use_mask:
                    valid_mask = torch.ones_like(I)
                    valid_mask[..., :1, :] = 0
                    valid_mask[..., -1:, :] = 0
                    valid_mask[..., :, :1] = 0
                    valid_mask[..., :, -1:] = 0
                # loss
                if valid_mask is not None:
                    loss = F.mse_loss(I_hat * valid_mask, I * valid_mask)
                else:
                    loss = F.mse_loss(I_hat, I)

            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch or epoch == opt.warmup_epoch:
            save_network(network, epoch, "model")
            save_state(epoch, optimizer, scheduler)
            # print log
            logger.info("===> Train Epoch[{}]: Loss: {:.6f}".format(epoch, loss.item()))

            # validate(network, valid_dict, opt, systime, epoch)


def validate(network, valid_dict, opt, systime, epoch):
    """Validation loop with RGB prediction and RGB ground truth."""
    network.eval()
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    validation_path = os.path.join(save_model_path, "validation_rgb")
    os.makedirs(validation_path, exist_ok=True)
    np.random.seed(101)

    for valid_name, valid_data in valid_dict.items():
        avg_psnr = []
        avg_ssim = []
        save_dir = os.path.join(validation_path, valid_name)
        os.makedirs(save_dir, exist_ok=True)
        num_img, num_block, val_noisy, val_gt = valid_data

        for idx in range(num_img):
            for idy in range(num_block):
                gt = val_gt[idx, idy] / 255.0  # shape [H, W, 3]
                noisy = val_noisy[idx, idy][:, :, np.newaxis]  # shape [H, W, 1]

                transformer = transforms.Compose([transforms.ToTensor()])
                noisy_tensor = transformer(noisy).unsqueeze(0).to(device)  # [1, 1, H, W]
                # noisy_tensor = space_to_depth(noisy_tensor, 2)  # [1, 4, H/2, W/2]
                noisy_tensor = BayerPatternShifter.bayer_1ch_to_4ch(noisy_tensor)  # [1, 4, H, W]

                with torch.no_grad():
                    pred_rgb = network(noisy_tensor)  # [1, 3, H, W]

                pred_rgb = pred_rgb.permute(0, 2, 3, 1).cpu().clamp(0, 1).numpy().squeeze(0)
                pred255 = np.clip(pred_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

                psnr = calculate_psnr(gt.astype(np.float32), pred_rgb.astype(np.float32), 1.0)
                ssim = calculate_ssim(gt * 255.0, pred_rgb * 255.0)
                avg_psnr.append(psnr)
                avg_ssim.append(ssim)

                save_path = os.path.join(save_dir, f"{valid_name}_{idx:03d}-{idy:03d}-{epoch}_rgb.png")
                Image.fromarray(pred255).save(save_path)

        avg_psnr = np.mean(avg_psnr)
        avg_ssim = np.mean(avg_ssim)

        # print log
        logger.info("===> Validation Epoch[{}]: {} PSNR: {:.6f} dB, SSIM: {:.6f}".format(
            epoch, valid_name, avg_psnr, avg_ssim))

if __name__ == "__main__":
    # Training Set
    TrainingDataset = DataLoader_SIDD_Medium_Raw(opt.data_dir)
    TrainingLoader = DataLoader(dataset=TrainingDataset,
                                num_workers=8,
                                batch_size=opt.batchsize,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)

    # Validation Set
    valid_dict = {
        "SIDD_Val": get_SIDD_validation(opt.val_dirs),
        # "Kodak24": validation_kodak(opt.val_dirs),
    }

    # Network
    network = UNet(in_channels=opt.in_channel,
                    out_channels=opt.out_channel,
                    wf=opt.n_feature)
    if opt.parallel:
        network = torch.nn.DataParallel(network)
    network = network.to(device)

    # Training scheme
    num_epoch = opt.n_epoch
    ratio = num_epoch / 100
    optimizer = optim.Adam(network.parameters(), lr=opt.lr, weight_decay=opt.w_decay)
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

    print('init finish')

    # Training
    train(network, optimizer, scheduler, TrainingLoader, epoch_init, num_epoch, opt)
