import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("models/")
from mlp import MLP

class GIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps,
                 graph_pooling_type, neighbor_pooling_type, device, residual_type=None):
        '''
            num_layers: 神经网络的层数（包括输入层）
            num_mlp_layers: MLP 的层数（不包括输入层）
            input_dim: 输入特征的维度
            hidden_dim: 所有层的隐藏单元维度
            output_dim: 预测的类别数
            final_dropout: 最后一层的 dropout 比率
            learn_eps: 如果为 True，则学习 epsilon 以区分中心节点和邻居节点；如果为 False，则将邻居和中心节点一起聚合
            neighbor_pooling_type: 聚合邻居的方式（mean, average, 或 max）
            graph_pooling_type: 聚合整个图中节点的方式（mean, average）
            device: 使用的设备（CPU 或 GPU）
        '''

        super(GIN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        ### MLP 列表
        self.mlps = torch.nn.ModuleList()

        ### 应用于 MLP 输出的 BatchNorm 列表（最终预测线性层的输入）
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 将不同层的隐藏表示映射到预测分数的线性函数
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

    def __preprocess_neighbors_maxpool(self, batch_graph):
        ### 创建连接图中的邻居列表

        # 计算当前小批次中图的最大邻居数
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + graph.num_nodes)
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                # 为邻居索引添加偏移值
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                # 填充，假定虚拟数据存储在 -1 中
                pad.extend([-1] * (max_deg - len(pad)))

                # 如果 learn_eps 为 False，则在 maxpooling 中添加中心节点，即将中心节点和邻居节点一起聚合。
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ### 创建块对角稀疏矩阵

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + graph.num_nodes)
            edge_mat_list.append(graph.edge_index + start_idx[i])

        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])
        total_nodes = start_idx[-1]

        # 如果 learn_eps 为 False，则在邻接矩阵中添加自环，即将中心节点和邻居节点一起聚合。
        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        return torch.sparse_coo_tensor(
            Adj_block_idx,
            Adj_block_elem,
            size=(total_nodes, total_nodes),
            device=self.device
        ).coalesce()

    def __preprocess_graphpool(self, batch_graph):
        ### 创建图中所有节点的 sum 或 average pooling 稀疏矩阵（num graphs x num nodes）

        start_idx = [0]

        # 计算填充的邻居列表
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + graph.num_nodes)

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ### average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1. / graph.num_nodes] * graph.num_nodes)
            else:
                ### sum pooling
                elem.extend([1] * graph.num_nodes)

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ### 元素最小值永远不会影响 max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ### 通过 epsilon 加权分别对邻居节点和中心节点进行 pooling

        if self.neighbor_pooling_type == "max":
            ## 如果是 max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # 如果是 sum 或 average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # 如果是 average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # 在与邻居聚合时重新加权中心节点表示
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # 非线性激活
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ### 将邻居节点和中心节点一起进行 pooling

        if self.neighbor_pooling_type == "max":
            ## 如果是 max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # 如果是 sum 或 average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # 如果是 average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # 邻居和中心节点的表示
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # 非线性激活
        h = F.relu(h)
        return h

    def forward(self, batch_graph, **kwargs):
        # 将所有图的节点特征拼接成一个张量
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        # 每层的隐藏表示列表（包括输入）
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)

            hidden_rep.append(h)

        score_over_layer = 0

        # 在每一层对图中所有节点进行 pooling
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout,
                                          training=self.training)

        return score_over_layer