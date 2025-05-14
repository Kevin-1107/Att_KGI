import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .kan import KANLinear

class AttentionKAN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_heads=4, k_neighbors=50):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.k_neighbors = k_neighbors

        # 多头部投影
        self.q_proj = nn.Linear(in_dim, hidden_dim)
        self.k_proj = nn.Linear(in_dim, hidden_dim)
        self.v_proj = nn.Linear(in_dim, hidden_dim)

        # 位置编码
        self.pos_enc = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim))

        # KAN注意力权重生成
        self.kan = nn.Sequential(
            KANLinear(hidden_dim * 2, hidden_dim, grid_size=5),
            nn.GELU(),
            KANLinear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, h_src, h_dst, edge_index):
        num_edges = h_src.size(0)

        # 投影到多头空间 [E, num_heads, head_dim]
        q = self.q_proj(h_src).view(num_edges, self.num_heads, self.head_dim)
        k = self.k_proj(h_dst).view(num_edges, self.num_heads, self.head_dim)
        v = self.v_proj(h_dst).view(num_edges, self.num_heads, self.head_dim)

        # 添加位置编码
        q = q + self.pos_enc
        k = k + self.pos_enc

        # 计算注意力分数 [E, num_heads]
        att_logits = (q * k).sum(dim=-1) / (self.head_dim ** 0.5)
        att_weights = torch.softmax(att_logits, dim=1)
        att_weights = self.dropout(att_weights)

        # 聚合特征 [E, head_dim]
        attended = (att_weights.unsqueeze(-1) * v)
        attended = attended.view(num_edges, -1)  # [E, hidden_dim]

        # 拼接原始特征生成最终权重
        alpha = self.kan(torch.cat([h_src, attended], dim=1))
        return alpha.squeeze()

class GIN_KAN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim,
                 final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type,
                 device, dynamic_grid=True, max_degree=100, num_heads=4, residual_type="auto"):
        super(GIN_KAN, self).__init__()

        # 基本参数
        self.dynamic_grid = dynamic_grid
        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.max_degree = max_degree
        self.hidden_dim = hidden_dim

        self.kans = nn.ModuleList()
        self.attentions = nn.ModuleList([
            AttentionKAN(
                in_dim=hidden_dim,  # 与图分类一致
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ) for _ in range(num_layers - 1)
        ])
        self.skip_cons = nn.ModuleList()
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])
        self.num_heads = num_heads

        self.residual_type = residual_type

        self.res_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(num_layers - 1)
        ])

        for layer in range(num_layers - 1):

            self.kans.append(
                KANLinear(
                    hidden_dim if layer > 0 else input_dim,
                    hidden_dim,
                    grid_size=5,
                    dynamic_grid=dynamic_grid
                )
            )

            # 跳跃连接
            if layer == 0:
                self.skip_cons.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.skip_cons.append(nn.Identity())

            # 分类器
            self.classifier = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(final_dropout)

    def _get_edge_index(self, batch_graph):
        edge_list = []
        start_idx = 0
        for graph in batch_graph:
            edges = graph.edge_index + start_idx
            edge_list.append(edges)
            start_idx += graph.g
        return torch.cat(edge_list, dim=1).to(self.device)

    def __preprocess_neighbors(self, batch_graph):
        if self.neighbor_pooling_type == "max":
            return {"padded_neighbor_list": self.__preprocess_neighbors_maxpool(batch_graph)}
        else:
            return {"Adj_block": self.__preprocess_neighbors_sumavepool(batch_graph)}

    def __preprocess_neighbors_maxpool(self, batch_graph):
        max_deg = max(graph.max_neighbor for graph in batch_graph)
        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + graph.g)
            for j in range(len(graph.neighbors)):
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                pad += [-1] * (max_deg - len(pad))
                if not self.learn_eps:
                    pad.append(j + start_idx[i])
                padded_neighbor_list.append(pad)

        return torch.LongTensor(padded_neighbor_list)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        edge_mat_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + graph.g)
            edge_mat_list.append(graph.edge_index + start_idx[i])
        # 合并所有边索引
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])
        # 添加自环（如果需要）
        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, torch.ones(num_node)], 0)

        return torch.sparse.FloatTensor(
            Adj_block_idx,
            Adj_block_elem,
            (start_idx[-1], start_idx[-1])
        ).to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.unsqueeze(0).to(self.device)])
        return torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]

    def next_layer_eps(self, h, layer, **params):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, params["padded_neighbor_list"])
        else:
            pooled = torch.spmm(params["Adj_block"], h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(params["Adj_block"],
                                    torch.ones((params["Adj_block"].shape[0], 1)).to(self.device))
                pooled = pooled / degree

        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.kans[layer](pooled)
        return F.relu(self.batch_norms[layer](pooled_rep))

    def next_layer(self, h, layer, **params):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, params["padded_neighbor_list"])
        else:
            pooled = torch.spmm(params["Adj_block"], h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(params["Adj_block"],
                                    torch.ones((params["Adj_block"].shape[0], 1)).to(self.device))
                pooled = pooled / degree

        pooled_rep = self.kans[layer](pooled)
        return F.relu(self.batch_norms[layer](pooled_rep))

    def _aggregate(self, h, layer, params):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, params["padded_neighbor_list"])
        else:
            pooled = torch.spmm(params["Adj_block"], h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(params["Adj_block"],
                                    torch.ones((h.size(0), 1).to(self.device)))
                pooled = pooled / degree
        return pooled + (1 + self.eps[layer]) * h

    def forward(self, batch_graph, epoch=None):
        x = torch.cat([g.node_features for g in batch_graph], 0).to(self.device)
        edge_index = self._get_edge_index(batch_graph)
        adj_params = self.__preprocess_neighbors(batch_graph)

        hidden_states = [x]
        h = x

        # 动态计算残差类型（仅在首次前向传播时）
        if self.residual_type == "auto" and not hasattr(self, '_res_type_set'):
            avg_nodes = np.mean([g.g for g in batch_graph])
            print(f"数据集的平均节点数: {avg_nodes}")
            self.residual_type = "gate" if avg_nodes < 100 else "add"
            print(f"采用的残差连接方式: {self.residual_type}")
            self._res_type_set = True  # 标记已设置，避免重复计算

        for layer in range(self.num_layers - 1):
            # 动态网格更新
            if self.dynamic_grid and self.training and epoch is not None and (epoch % 5 == 0):
                with torch.no_grad():
                    self.kans[layer].update_grid(h)

            # 步骤1: GIN风格聚合
            h_pooled = self._aggregate(h, layer, adj_params)

            # 步骤2: KAN处理
            h_kan = self.kans[layer](h_pooled)
            h_kan = self.batch_norms[layer](h_kan)
            h_kan = F.relu(h_kan)

            # 步骤3: 注意力池化
            src, dst = edge_index
            h_src, h_dst = h_kan[src], h_kan[dst]
            alpha = self.attentions[layer](h_src, h_dst, edge_index)

            # 构建注意力邻接矩阵
            adj_att = torch.sparse_coo_tensor(
                edge_index,
                alpha.squeeze(),
                (h_kan.size(0), h_kan.size(0)),
                device=self.device
            )
            h_att = h_kan + torch.spmm(adj_att, h_kan)

            # 步骤4: 残差连接
            residual = self.skip_cons[layer](hidden_states[-1])
            if self.residual_type == "gate":
                gate = torch.sigmoid(self.res_gates[layer])
                h_next = gate * h_att + (1 - gate) * residual
            else:  # 简单相加
                h_next = h_att + residual
            h_next = self.layer_norms[layer](h_next)

            hidden_states.append(h_next)
            h = h_next

        return self.classifier(self.dropout(h))