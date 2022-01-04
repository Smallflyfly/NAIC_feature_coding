#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/4 14:17 
"""
from torch.utils.data import Dataset

ROOT_PATH = 'F:\\AI\\dataset\\NAIC\\train\\'


class FeatureDataset(Dataset):

    def __init__(self):
        super.__init__()

    def __getitem__(self, index):
        pass

    def __len__(self):
        return