#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/4 14:17 
"""
import os
import numpy as np

from torch.utils.data import Dataset

ROOT_PATH = 'data/train/train_feature'
TRAIN_LIST = 'data/train/train_list.txt'


class FeatureDataset(Dataset):

    def __init__(self):
        super.__init__()
        self.class_num = 0
        self.train_images = []
        self.train_labels = []
        self.relabel_map = {}

        self.get_train_id()

    def get_train_id(self):
        label_nums = []
        count = 0
        with open(TRAIN_LIST) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                file = line[0]
                num = int(line[1])
                if num not in label_nums:
                    label_nums.append(num)
                    self.relabel_map[num] = count
                    self.train_images.append(os.path.join(ROOT_PATH, file))
                    self.train_labels.append(count)
                    count += 1
                else:
                    self.train_images.append(os.path.join(ROOT_PATH, file))
                    self.train_labels.append(self.relabel_map[num])

    def __getitem__(self, index):
        label = self.train_labels[index]
        image = self.train_images[index]
        image = np.fromfile(image, dtype='<f4')[None, None]

    def __len__(self):
        return