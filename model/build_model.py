#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/7 16:08 
"""
from torchvision.models import resnet50

from model.mlp import MLP


def build_model(model = 'resnet50'):

    if model == 'resnent50':
        return resnet50()

    elif model == 'mlp':
        return MLP(depth=5)