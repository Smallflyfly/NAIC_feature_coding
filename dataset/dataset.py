#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/4 14:17 
"""
import os
import random

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

ROOT_PATH = 'data/train/train_feature'
TRAIN_LIST = 'data/train/train_list.txt'


class FeatureDataset(Dataset):
    def __init__(self, training=True):
        super(FeatureDataset, self).__init__()
        self.class_num = 15000
        self.training = training
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.all_images = []
        self.all_labels = []

        self.train_images = []
        self.train_labels = []

        self.val_images = []
        self.val_labels = []

        self.error_list = ['00005879.dat', '00005879.dat']
        self.get_all_data()

        self.get_train_val_data()

    def get_all_data(self):
        with open(TRAIN_LIST) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                file = line[0]
                if file in self.error_list:
                    continue
                num = int(line[1])
                self.all_images.append(os.path.join(ROOT_PATH, file))
                self.all_labels.append(num)

    def get_train_val_data(self):
        total = len(self.all_images)
        val_num = int(total * 0.1)
        val_list = self._generate_val_num(val_num, total)
        print('val list:', len(val_list))
        index = 0
        for image, label in zip(self.all_images, self.all_labels):
            if index in val_list:
                self.val_images.append(image)
                self.val_labels.append(label)
            else:
                self.train_images.append(image)
                self.train_labels.append(label)
            index += 1
        print("data process successfully")

    def __getitem__(self, index):
        if self.training:
            image = self.train_images[index]
            label = self.train_labels[index]
        else:
            image = self.val_images[index]
            label = self.val_labels[index]
        image = np.fromfile(image, dtype='<f4')[None, None]
        image = self.transforms(image)
        image = image[0][0]
        return label, image

    def __len__(self):
        if self.training:
            return len(self.train_images)
        else:
            return len(self.val_images)

    def _generate_val_num(self, val_num, total):
        val_list = []
        while val_num > 0:
            temp = random.randint(0, total-1)
            if temp not in val_list:
                val_list.append(temp)
                val_num -= 1
        return val_list