from torch.utils.data import DataLoader, Dataset
from preprocessing import bayer_filter, interp
import os
import  cv2

from torchvision.transforms import ToTensor

class ImageDataset(Dataset):
    def __init__(self, input, is_clean=True):
        self.input = input
        self.is_clean = is_clean
        # list all pngs in the input directory
        if os.path.isdir(input):
            self.input = [os.path.join(input, f) for f in os.listdir(input) if f.endswith('.png')]

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.input[idx], cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        name = os.path.basename(self.input[idx])
        # convert to [0,1]
        img = img / 255.0
        if self.is_clean:
            return ToTensor()(interp(bayer_filter(img))), name
        else:
            return ToTensor()(img), name