#!/usr/bin/python
#-*- coding:utf-8 -*-
"""
@author:smallflyfly
@time: 2022/01/07
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
import json


def read_dat(path):
    return np.fromfile(path, dtype=np.float32)


with open('data/sub_a.json', 'w', encoding='utf8') as f:
    # up = open('data/sub_a.json', 'w+')
    # sub = json.load(f)

    test_query_path = glob.glob('/media/smallflyfly/DATA_MANAGER/AI/datasets/NAIC/test_A/query_feature_A/*.dat')
    test_query_path = np.array(test_query_path)
    test_query = [read_dat(path) for path in tqdm(test_query_path)]
    test_query = np.vstack(test_query)

    test_gallery_path = np.array(glob.glob('/media/smallflyfly/DATA_MANAGER/AI/datasets/NAIC/test_A/gallery_feature_A/*.dat'))
    test_gallery_path = np.array(test_gallery_path)
    test_gallery = [read_dat(path) for path in tqdm(test_gallery_path)]
    test_gallery = np.vstack(test_gallery)

    test_query = normalize(test_query)
    test_gallery = normalize(test_gallery)

    # with open('data/sub_a.json') as up:
    #     sub = json.load(up)

    total_idx = 0
    sub = {}
    for idx in range(test_query.shape[0]//1000 + 1):
        idss = np.dot(test_query[idx*1000: (idx+1)*1000], test_gallery.T)
        for ids in idss:
            ids_path = test_gallery_path[ids.argsort()[::-1][:100]]
            sub_name = os.path.basename(test_query_path[total_idx])
            sub[sub_name] = [os.path.basename(x) for x in ids_path]
            total_idx += 1
    f.write(json.dumps(sub, indent=2, sort_keys=False))

# up.close()