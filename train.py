#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/4 14:31 
"""
import argparse

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset.dataset import FeatureDataset, getDataset
from model.mlp import MLP

import tensorboardX as tb

from utils.utils import build_optimizer, build_scheduler

parser = argparse.ArgumentParser(description="river pollution level classify")
parser.add_argument("-b", type=int, default=16, help="train batch size")
parser.add_argument("-e", type=int, default=20, help="train epochs")

args = parser.parse_args()

BATCH_SIZE = args.b
EPOCHS = args.e
NUM_CLASSES = 15000

depth = 5


def train():
    train_sets, val_sets = getDataset()
    train_dataset = FeatureDataset(dataset=train_sets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = FeatureDataset(training=False, dataset=val_sets)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = MLP(num_classes=NUM_CLASSES, depth=depth)
    print(model)
    model = model.cuda()

    optimizer = build_optimizer(model, 'adam', lr=0.001)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=EPOCHS)
    loss_func = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    writer = tb.SummaryWriter()

    for epoch in range(EPOCHS):
        model.train()
        # model.eval()
        # if epoch > 0 and epoch % 10 == 0
        for index, data in enumerate(train_loader):
            im, label = data
            im = im.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = model(im)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            if index % 50 == 0:
                num_epoch = epoch * len(train_loader) + index
                print('Epoch: [{}/{}] [{}/{}]  loss = {:.6f}'.format(epoch + 1, EPOCHS, index + 1, len(train_loader),
                                                                     loss))
                writer.add_scalar('loss', loss, num_epoch)

        scheduler.step()
        if (epoch + 1) % 5 == 0 and epoch != EPOCHS-1:
            torch.save(model.state_dict(), 'weights/mlp_{}_{}.pth'.format(depth, epoch+1))

    writer.close()


if __name__ == '__main__':
    train()