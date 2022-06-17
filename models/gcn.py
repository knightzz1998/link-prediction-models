#! /usr/bin/env python
# -*-coding:utf-8-*-
from abc import ABC

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from process.graph_data import get_cora


class GCNConv(MessagePassing, ABC):

    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.liner = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        :param x: [node_num, feature_dim]
        :param edge_index: [2, edge_num]
        :return:
        """
        # 为邻接矩阵添加自环
        edge_index, edge_attr = add_self_loops(edge_index, num_nodes=x.size(0))

        # 线性转换节点特征矩阵
        # linear = y = xA^T + b
        # x[node_num, feature_dim] * w[in_channel, node_num]^T  + b
        x = self.liner(x)

        # 开始聚合操作
        # 所有的逻辑代码都在forward()里面，当我们调用propagate()函数之后，它将会在内部调用message()和update()。
        return self.propagate(edge_index, x=x, size=(x.size(0), x.size(0)))

    def message(self, x_j, edge_index, size):
        """
        消息传递函数
        :param x_j:
        :param edge_index:
        :return:
        """
        row, col = edge_index
        # 计算度
        deg = degree(row, num_nodes=size[0], dtype=x_j.dtype)

    def update(self, aggr_out):
        """
        更新函数
        :param aggr_out: aggr_out has shape [N, out_channels]
        :return:
        """
        return aggr_out


if __name__ == '__main__':
    dataset = get_cora()
    data = dataset[0]
    conv = GCNConv(dataset.num_features, 16)
    out = conv(data.x, data.edge_index)
    print(out.shape)