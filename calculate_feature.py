import json
import os#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/6 10:23 
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

def read_dat(path):
    return np.fromfile(path, dtype=np.float32)

with open('data/sub_a.json') as up:
    sub = json.load(up)

    test_query_path = glob.glob('./data/test_A/query_feature_A/*.dat')
    test_query_path = np.array(test_query_path)
    test_query = [read_dat(path) for path in tqdm(test_query_path)]
    test_query = np.vstack(test_query)

    test_gallery_path = np.array(glob.glob('./data/test_A/gallery_feature_A/*.dat'))
    test_gallery_path = np.array(test_gallery_path)
    test_gallery = [read_dat(path) for path in tqdm(test_gallery_path)]
    test_gallery = np.vstack(test_gallery)

    test_query = normalize(test_query)
    test_gallery = normalize(test_gallery)

    # with open('data/sub_a.json') as up:
    #     sub = json.load(up)

    total_idx = 0
    for idx in range(test_query.shape[0]//1000 + 1):
        idss = np.dot(test_query[idx*1000: (idx+1)*1000], test_gallery.T)
        for ids in idss:
            ids_path = test_gallery_path[ids.argsort()[::-1][:100]]
            sub_name = os.path.basename(test_query_path[total_idx])
            sub[sub_name] = [os.path.basename(x) for x in ids_path]
            total_idx += 1