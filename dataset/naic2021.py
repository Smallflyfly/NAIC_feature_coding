#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/7 14:34 
"""
import os

from utils.utils import check_dir_file


class NAICTrain:

    def __init__(self, root='data'):
        self.root = root
        self.dataset_dir = os.path.join(self.root, 'train')
        train_set, query_set, gallery_set = self.gen_samle_sets()

    def gen_sample_sets(self):
        train_dir = os.path.join(self.dataset_dir, 'train_feature')
        query_dir = os.path.join(self.dataset_dir, 'train_feature')
        gallery_dir = os.path.join(self.dataset_dir, 'train_feature')
        check_dir_file([train_dir, query_dir, gallery_dir])

        train_set = self.process_dir(train_dir, os.path.join(self.dataset_dir, 'sub_train_list.txt'))

    def process_dir(self, dir, file_list_path, pseudo_camid, is_train=True):
        check_dir_file(file_list_path)
        data_list = []
        with open(file_list_path, 'r') as f:
            for line in f:
                im_name, im_pid = line.strip().split()
                data_list.append((os.path.join(dir, im_name), int(im_pid)))

        return data_list