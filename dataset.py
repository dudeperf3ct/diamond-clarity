import os
import numpy as np
from PIL import Image
from random import shuffle
import torch
from torch.utils.data import Dataset


class ImageClassifierDataset(Dataset):
    """Image Classifier dataset"""

    def __init__(self, img_paths: list, lbls: list = None, root_dir: str = None, transform=None):
        """
        Args:
            img_paths (list) : list containing path of images
            lbls (list) : list containing labels corresponding to images
            root_dir (str) : Parent path for reading images
            transform (callabe, Optional): Transforms to be applied
        """
        self.img_filepath = img_paths
        self.lbls = lbls
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_filepath)

    def __getitem__(self, idx):
        images = []
        for i in range(6):
            if self.root_dir is not None:
                img_filename = os.path.join(self.root_dir, self.img_filepath[idx]+f'_{i}.jpg')
            else:
                img_filename = self.img_filepath[idx]
            img = np.array(Image.open(img_filename).convert("RGB"))
            if self.transform is not None:
                img = self.transform(image=img)["image"]
            images.append(img)
        #shuffle(images)
        if self.lbls is not None:
            # N,C​,D,H,W
            return torch.stack(images, 0).permute(1,0,2,3), self.lbls[idx]
        else:
            return images
