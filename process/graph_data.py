#! /usr/bin/env python
# -*-coding:utf-8-*-
from torch_geometric.datasets import Planetoid


def get_cora():
    dataset = Planetoid(root="../datasets", name="Cora")
    return dataset
