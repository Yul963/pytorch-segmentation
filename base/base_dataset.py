import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, size=400, augment=False, val=False,
                 flip=False, rotate=False, blur=False, return_id=False):
        self.root = root
        self.split = split
        self.augment = augment
        self.transform_image = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.transform_label = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        if self.augment:
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self._set_files()
        self.return_id = return_id

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)

        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)

        return image, label

        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        if self.augment:
            image, label = self._augmentation(image, label)

        # label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        label = Image.fromarray(np.array(label, dtype=np.int32))
        image = Image.fromarray(np.uint8(image))
        # print(label.size, image.size)
        if self.return_id:
            return self.transform_image(image), self.transform_label(label).long(), image_id
        return self.transform_image(image), self.transform_label(label).long()

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

