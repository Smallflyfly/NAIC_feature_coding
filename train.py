#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/4 14:31 
"""
import argparse

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset.dataset import FeatureDataset
from model.mlp import MLP

import tensorboardX as tb

from utils.utils import build_optimizer, build_scheduler

parser = argparse.ArgumentParser(description="river pollution level classify")
parser.add_argument("-b", type=int, default=16, help="train batch size")
parser.add_argument("-e", type=int, default=20, help="train epochs")

args = parser.parse_args()

BATCH_SIZE = args.b
EPOCHS = args.e

def train():
    dataset = FeatureDataset(class_num=15000)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader()
    model= MLP(num_classes=dataset.class_num)
    print(model)
    model = model.cuda()

    optimizer = build_optimizer(model, 'adam', lr=0.0005)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=EPOCHS)
    loss_func = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    for epoch in range(EPOCHS):
        model.train()
        for index, data in enumerate(train_loader):
            im, label = data
            im = im.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = model(im)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    train()