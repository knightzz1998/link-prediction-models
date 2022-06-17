#! /usr/bin/env python
# -*-coding:utf-8-*-

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F
import argparse


class GCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        """
        初始化
        :param in_channels: num_features
        :param out_channels: num_classes
        """
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        out = self.conv1(x, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.conv2(out, edge_index)
        return F.sigmoid(out)


def process():
    dataset = Planetoid(root="../datasets", name="Cora")
    data = dataset[0]
    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    return dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--hidden_channels', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num_features', type=int, default=1433, help='节点特征数量')
    parser.add_argument('--num_classes', type=int, default=7, help='节点类别数量')
    args = parser.parse_args()
    return args


def train(data, args, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test(data, model):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        test_acc = correct / data.test_mask.sum().item()
        return test_acc


def main(args):
    dataset = Planetoid(root="../datasets", name="Cora")
    data = dataset[0]
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # = train_test_split_edges(data)
    # 输出维度可以自定义, 如果设置为2, 就是二分类, 如果是节点的类别数量, 就是对节点多分类
    model = GCN(args.num_features, hidden_channels=args.hidden_channels, out_channels=args.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(args.epochs):
        loss = train(data, args, model, criterion, optimizer)
        accuracy = test(data, model)
        print('Epoch: {:03d}, Loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch, loss, accuracy))


if __name__ == '__main__':
    args = parse_args()
    main(args)
