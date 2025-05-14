import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, device=None, learn_eps=True):
        super(GIN, self).__init__()

        self.device = device
        self.nfeat = nfeat  # 输入特征维度
        self.nhid = nhid  # 隐藏层维度
        self.nclass = nclass  # 输出类别数
        self.dropout = dropout
        self.learn_eps = learn_eps

        # 图卷积层定义
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(nfeat, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid),
                nn.ReLU()
            ),
            eps=0.0, # 用可学习eps
            train_eps=learn_eps
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(nhid, nhid),  # 增加隐藏层
                nn.ReLU(),
                nn.Linear(nhid, nclass)
            ),
            eps=0.0,  # 启用可学习eps
            train_eps=learn_eps
        )
        # 兼容性伪属性（与 KGI对齐）
        self.attention = False
        # self.eps = nn.Parameter(torch.zeros(1)) if learn_eps else 0.0

    def __preprocess_adj(self, batch_graph):
        """统一邻接矩阵预处理逻辑"""
        edge_index = torch.cat([graph.edge_mat for graph in batch_graph], dim=1)
        edge_weight = torch.ones(edge_index.shape[1])
        return edge_index.to(self.device), edge_weight.to(self.device)

    def forward(self, batch_graph):
        # 合并节点特征和边索引
        graph = batch_graph[0]
        x = graph.node_features.to(self.device)
        edge_index = graph.edge_index.to(self.device)

        # First GIN layer
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GIN layer
        x = self.conv2(x, edge_index)

        return x  # Return raw logits for cross-entropy loss

    # 保持与GIN_KAN兼容的方法
    def get_dynamic_grid_size(self, degrees):
        """保持接口一致（实际无需实现）"""
        return None

    # 统一预测接口
    def predict(self, features=None, adj=None):
        self.eval()
        with torch.no_grad():
            return self.forward(features) if features else None