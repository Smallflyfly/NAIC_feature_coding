#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/4 14:30 
"""
import h5py
import scipy.io as scio


TEST_MAT_FILE = 'F:\\AI\\dataset\\NAIC\\train\\train_feature\\00012531.dat'


def demo():
    # plane = scio.loadmat(TEST_MAT_FILE)
    plane = h5py.File(TEST_MAT_FILE, 'w')
    print(plane)
    print(plane.keys())


if __name__ == '__main__':
    demo()
