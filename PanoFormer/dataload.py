import os
import copy

from torch.utils.data import Dataset

from common import _load_paths, _load_image, _filename_separator
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
import random

class Dataload(Dataset):
    def __init__(self, txt_path, height=256, width=512,disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, test = False, is_training=False, transform = None, target_transform = None):

        self.max_depth_meters = 16.0
        self.w = width
        self.h = height

        self.is_training = is_training

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.test = test
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))#imgs.append((words[0], words[3]))
            self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        #print(fn)
        inputs = {}
        # with open('rgbpath.txt', "a") as f:
        #     f.write(str(fn) + '\n')
        #     f.close()
        # img = cv2.imread(fn)
        # img = cv2.resize(img, (512,256), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.imread('../data/'+fn)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(512, 256))


        
        # depth = cv2.imread(label, cv2.IMREAD_ANYDEPTH)
        # depth = cv2.resize(depth,(512,256),interpolation=cv2.INTER_CUBIC)

        gt_depth = cv2.imread('../data/'+label, cv2.IMREAD_ANYDEPTH)
        gt_depth = cv2.resize(gt_depth, dsize=(512, 256), interpolation=cv2.INTER_CUBIC)
        gt_depth = gt_depth.astype(np.float)/512#gt_depth = gt_depth.astype(np.float)
        gt_depth[gt_depth > self.max_depth_meters + 1] = self.max_depth_meters + 1

        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w//4)
        else:
            roll_idx = 0
            
            
        rgb = np.roll(rgb, roll_idx, 1)
        gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        mask = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                              & ~torch.isnan(inputs["gt_depth"]))
        mask1 = (rgb[0] != 0)
        mask1 = (mask1).float()
        inputs["val_mask"] = mask.float()*mask1
        return inputs

    def __len__(self):
        return len(self.imgs)
