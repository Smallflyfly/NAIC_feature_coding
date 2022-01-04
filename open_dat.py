#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/4 14:30 
"""
import struct

import h5py
import scipy.io as scio


TEST_DAT_FILE = 'F:/AI/dataset/NAIC/train/train_feature/00012531.dat'

import numpy as np



def read_dat(path):
    img = np.fromfile(path, dtype='<f4')[None, None]
    print(img)

def demo():
    # plane = scio.loadmat(TEST_MAT_FILE)
    # plane = h5py.File(TEST_MAT_FILE, 'w')
    # with open(TEST_DAT_FILE, 'w', encoding='utf-8') as f:
    #     print(f.readlines())
        # print(lines)
    read_dat(TEST_DAT_FILE)

if __name__ == '__main__':
    demo()
