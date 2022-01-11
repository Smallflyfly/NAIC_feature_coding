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

# ROOT_PATH = 'data/train/train_feature'
ROOT_PATH = '/media/smallflyfly/DATA_MANAGER/AI/datasets/NAIC/train'
TRAIN_LIST = 'data/train/train_list.txt'


def _generate_val_num(val_num, total):
    val_list = []
    while val_num > 0:
        temp = random.randint(0, total - 1)
        if temp not in val_list:
            val_list.append(temp)
            val_num -= 1
    return val_list


class FeatureDataset(Dataset):
    def __init__(self, training=True, class_num=15000, dataset=None):
        super(FeatureDataset, self).__init__()
        self.class_num = class_num
        self.training = training
        self.dataset = dataset
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_images = []
        self.train_labels = []

        self.val_images = []
        self.val_labels = []

        self.get_train_val_data()

    def get_train_val_data(self):
        for data in self.dataset:
            image, label = data
            if self.training:
                self.train_images.append(image)
                self.train_labels.append(label)
            else:
                self.val_images.append(image)
                self.val_labels.append(label)

    def __getitem__(self, index):
        if self.training:
            image = self.train_images[index]
            label = self.train_labels[index]
        else:
            image = self.val_images[index]
            label = self.val_labels[index]
        # image = np.fromfile(image, dtype='<f4')[None, None]
        image = np.fromfile(image, dtype='<f4')[None, None]
        image = self.transforms(image)
        return image, label

    def __len__(self):
        if self.training:
            return len(self.train_images)
        else:
            return len(self.val_images)


def get_train_val_data(all_images, all_labels):
    train_set = []
    val_set = []
    total = len(all_images)
    val_num = int(total * 0.1)
    val_list = _generate_val_num(val_num, total)
    print('val list:', len(val_list))
    index = 0
    for image, label in zip(all_images, all_labels):
        if index in val_list:
            train_set.append((image, label))
        else:
            val_set.append((image, label))
        index += 1
    print("data process successfully")
    return train_set, val_set


def getDataset():
    error_list = ['00005879.dat', '00005879.dat']
    all_images = []
    all_labels = []
    with open(TRAIN_LIST) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            file = line[0]
            if file in error_list:
                continue
            num = int(line[1])
            all_images.append(os.path.join(os.path.join(ROOT_PATH, 'train_feature'), file))
            all_labels.append(num)
    train_set, val_set = get_train_val_data(all_images, all_labels)
    return train_set, val_set
