#! /usr/bin/env python
# -*-coding:utf-8-*-

class Walker():
    def __init__(self, graph, p=1, q=1):
        self.graph = graph
        self.p = p
        self.q = q
        self.walks = []

    def deepwalk(self, start_node, walk_length):
        """
        随机游走
        :param start_node: 开始节点的位置
        :param walk_length: 游走的长度
        :return:
        """
        pass