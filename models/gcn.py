#! /usr/bin/env python
# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid


def get_data():
    dataset = Planetoid(root="../datasets", name="Cora")
    return dataset[0]
