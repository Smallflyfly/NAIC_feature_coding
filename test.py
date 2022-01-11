#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/11 10:48 
"""
import argparse
import os

from torch.backends import cudnn
from torchvision import transforms

from model.mlp import MLP
from utils.utils import load_pretrained_weights
import numpy as np

parser = argparse.ArgumentParser(description="reid test")
parser.add_argument("-w", type=str, default='', help="weight")
args = parser.parse_args()

WEIGHT = args.w
TEST_PATH_ROOT = 'data/test_A'
transform = transforms.Compose([
    transforms.ToTensor()
])
NUM_CLASSES = 15000
depth = 5
cudnn.benchmark = True


def test(model):
    query_file_path = os.path.join(TEST_PATH_ROOT, 'query_feature_A')
    query_file_list = os.listdir(query_file_path)
    gallery_feature_path = os.path.join(TEST_PATH_ROOT, 'gallery_feature_A')
    gallery_file_list = os.listdir(gallery_feature_path)
    gallery_result_list = []
    for gallery_file in gallery_file_list:
        gallery_file = os.path.join(gallery_feature_path, gallery_file)
        image = np.fromfile(gallery_file, dtype='<f4')[None, None]
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.cuda()
        out = model(image)
        out = out.cpu().detach().numpy()
        assert out.shape[0] == 1, out.shape[1] == 2048
        gallery_result_list.append(out[0])
    gallery_result_list = np.vstack(gallery_result_list)
    print(gallery_result_list.shape)
    fang[-1]

    for query_file in query_file_list:
        query_file = os.path.join(query_file_path, query_file)
        image = np.fromfile(query_file, dtype='<f4')[None, None]
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.cuda()
        out = model(image)
        out = out.cpu().detach().numpy()
        assert out.shape[0] == 1, out.shape[1] == 2048
        print(out)
        fang[-1]

def load_model():
    model = MLP(num_classes=NUM_CLASSES, depth=depth, training=False)
    load_pretrained_weights(model, WEIGHT)
    model = model.cuda()
    model.eval()
    return model


if __name__ == '__main__':
    model = load_model()
    test(model)
