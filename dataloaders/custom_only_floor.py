from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import random
import torch
import cv2
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms


class CustomFloorDataset(BaseDataSet):

    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.custom2_palette
        super(CustomFloorDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ["training", "validation"]:
            self.image_dir = os.path.join(self.root, self.split)
            self.label_dir = os.path.join(self.root, self.split)
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]
        else:
            raise ValueError(f"Invalid split name {self.split}")

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '_seg.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path).convert('L'), dtype=np.int32)
        return image, label, image_id

    def _val_augmentation(self, image, label):
        h, w, _ = image.shape
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (
            int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        return image, label


class CustomFloor(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False,
                 shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False):
        self.MEAN = [0.48897059, 0.46548275, 0.4294]
        self.STD = [0.22861765, 0.22948039, 0.24054667]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = CustomFloorDataset(**kwargs)
        super(CustomFloor, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
