#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/7 16:39 
"""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_channels = 2048
        self.out_feature = 2048
        self.depth = 3

        self.squeeze = Squeeze()
        self.mlp_blocks = nn.Sequential(
            *[MLPBlock(in_features=self.feat_channels, bn=True, act=nn.ReLU if idx != self.depth-1 else None) for idx in range(self.depth-1)],
            MLPBlock(in_features=self.feat_channels, out_features=self.out_feature, bn=True, act=None)
        )
        self.unsqueeze = Unsqueeze()

    def forward(self, x):
        x = self.squeeze(x)
        x = self.mlp_blocks(x)
        x = self.unsqueeze()


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        assert x.ndim == 4 and x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class Unsqueeze(nn.Module):
    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        assert x.ndim == 2
        return x[:, :, None, None]


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features=None, bn=True, act=nn.ReLU):
        super().__init__()
        out_features = out_features or in_features
        self.bn = nn.BatchNorm1d(out_features) if bn is True else None
        self.fc = nn.Linear(in_features, out_features, bias=self.bn is None)
        self.act = act() if act is not None else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x