import collections
import os.path as osp
import cv2

import numpy as np
import PIL.Image
import torch
from torch.utils import data


class CVPPPDataset(data.Dataset):

    class_names = np.array([
        'background',   #0
        'leaf',         #1
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        # CVPPP training and validation datasets
        dataset_dir = osp.join(self.root, 'CVPPP')
        self.files = collections.defaultdict(list)
        if split in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, f'CVPPP_{split}', '%s_img.txt' % split)
            for img in open(imgsets_file):
                img = img.strip()
                img_file = osp.join(dataset_dir, f'CVPPP_{split}', img)
                lbl = img.replace('rgb','fg')
                lbl_file = osp.join(dataset_dir, f'CVPPP_{split}', lbl)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        img = img[:,:,:3]

        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int64)
        lbl[lbl == 255] = 1

        """Resizing the input image and corresponding ground truth label for training and validation"""
        img = cv2.resize(img, (410,410), interpolation = cv2.INTER_AREA)
        lbl = cv2.resize(lbl, (410,410), interpolation = cv2.INTER_NEAREST)

        img_transform, lbl_transform = self.transform(img, lbl)

        if self._transform:
            return img, lbl, img_transform, lbl_transform
        else:
            return img, lbl, img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]   # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]   # BGR -> RGB
        lbl = lbl.numpy()
        return img, lbl
