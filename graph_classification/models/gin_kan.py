import torch
import torch.nn as nn
import torch.nn.functional as F

from kan import KANLinear

class AttentionKAN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_heads=4, k_neighbors=50):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.k_neighbors = k_neighbors

        self.kan = nn.Sequential(
            KANLinear(hidden_dim * 2, hidden_dim, grid_size=5),
            nn.GELU(),
            KANLinear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.1)

        # 多头部投影
        self.q_proj = nn.Linear(in_dim, hidden_dim)
        self.k_proj = nn.Linear(in_dim, hidden_dim)
        self.v_proj = nn.Linear(in_dim, hidden_dim)

        # 位置编码初始化
        self.pos_enc = nn.Parameter(torch.randn(1, self.num_heads, self.head_dim))


    def forward(self, h_src, h_dst, edge_index):
        num_edges = h_src.size(0)

        # 投影到多头空间 [num_edges, num_heads, head_dim]
        q = self.q_proj(h_src).view(num_edges, self.num_heads, self.head_dim)
        k = self.k_proj(h_dst).view(num_edges, self.num_heads, self.head_dim)
        v = self.v_proj(h_dst).view(num_edges, self.num_heads, self.head_dim)

        # 添加位置编码
        q = q + self.pos_enc
        k = k + self.pos_enc

        # 计算注意力分数 [num_edges, num_heads]
        att_logits = (q * k).sum(dim=-1) / (self.head_dim ** 0.5)
        att_weights = torch.softmax(att_logits, dim=1)
        att_weights = self.dropout(att_weights)

        # 聚合特征 [num_edges, head_dim]
        attended = (att_weights.unsqueeze(-1) * v)
        # 恢复原始特征维度 [num_edges, hidden_dim]
        attended = attended.view(num_edges, -1)
        # 拼接并生成注意力权重
        return self.kan(torch.cat([h_src, attended], dim=1))

    def _project(self, x):
        return x.view(x.size(0), self.num_heads, self.head_dim)

class GIN_KAN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim,
                 final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type,
                 device, dynamic_grid=True, num_heads=4, att_dropout=0.2, residual_type="gate"
                 ):
        super(GIN_KAN, self).__init__()
        assert graph_pooling_type in ["sum", "average", "max"]
        assert neighbor_pooling_type in ["sum", "average", "max"]

        # 基本参数
        self.dynamic_grid = dynamic_grid
        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.hidden_dim = hidden_dim

        self.graph_pooling_type = graph_pooling_type
        self.grad_scale = nn.Parameter(torch.ones(num_layers - 1))

        self.linears_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim)
                )
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim)
                )

        self.num_heads = num_heads
        self.att_dropout = att_dropout

        self.attentions = nn.ModuleList([
            AttentionKAN(
                in_dim=hidden_dim,  # 根据实际输入维度调整
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ) for _ in range(num_layers - 1)
        ])

        # KAN层定义
        self.kans = nn.ModuleList()
        for layer in range(num_layers - 1):
            if layer == 0:
                # 第0层: 输入维度为 input_dim
                self.kans.append(
                    KANLinear(
                        in_features=input_dim,
                        out_features=hidden_dim,
                        grid_size=5,
                        spline_order=3,
                        dynamic_grid=dynamic_grid
                    )
                )
            else:
                # 第1层及以后: 输入维度为 hidden_dim
                self.kans.append(
                    KANLinear(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        grid_size=5,
                        spline_order=3,
                        dynamic_grid=dynamic_grid
                    )
                )

        self.residual_type = residual_type
        self.res_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5))  # 每层独立门控参数
            for _ in range(num_layers - 1)
        ])

        self.skip_cons = nn.ModuleList()
        for layer in range(num_layers - 1):
            if layer == 0:
                # 输入维度到hidden_dim的投影
                self.skip_cons.append(nn.Linear(input_dim, hidden_dim))
            else:
                # 恒等映射，不需要参数
                self.skip_cons.append(nn.Identity())

        # 归一化与分类
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])
        self.dropout = nn.Dropout(final_dropout)
        self.register_buffer('degrees', None)

    def _get_edge_index(self, batch_graph):
        edge_list = []
        start_idx = 0
        for graph in batch_graph:
            edges = graph.edge_index + start_idx
            edge_list.append(edges)
            start_idx += graph.node_features.shape[0]
        return torch.cat(edge_list, dim=1).to(self.device)

    def __preprocess_graphpool(self, batch_graph):
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + graph.num_nodes)

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            if self.graph_pooling_type == "average":
                elem.extend([1. / graph.num_nodes] * graph.num_nodes)
            else:
                elem.extend([1] * graph.num_nodes)
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1])])

        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse_coo_tensor(  # 替换为新的API
            idx,
            elem,
            size=torch.Size([len(batch_graph), start_idx[-1]]),
            dtype=torch.float32,
            device=self.device
        )
        return graph_pool.to(self.device)

    def __preprocess_neighbors_maxpool(self, batch_graph):
        max_deg = max([len(graph.neighbors) for graph in batch_graph])
        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + graph.num_nodes)
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
            num_nodes = graph.node_features.shape[0]  # 正确获取节点数
            start_idx.append(start_idx[i] + num_nodes)  # 累积节点索引

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

    def next_layer(self, h, layer, Adj_block=None):
        # 原始聚合
        pooled = torch.spmm(Adj_block, h)
        if self.neighbor_pooling_type == "average":
            degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
            pooled = pooled / degree

        pooled_rep = self.kans[layer](pooled)
        h_next = F.relu(self.batch_norms[layer](pooled_rep))

        # 残差连接
        gate = torch.sigmoid(self.res_gates[layer])
        residual = self.skip_cons[layer](h)
        h_next = gate * h_next + (1 - gate) * residual
        # 添加层归一化
        h_next = self.layer_norms[layer](h_next)
        return h_next

    def forward(self, batch_graph, epoch=None):
        # 节点特征拼接
        x = torch.cat([g.node_features for g in batch_graph], 0).to(self.device)
        h_current = x.clone()  # 显式初始化h
        edge_index = self._get_edge_index(batch_graph)
        graph_pool = self.__preprocess_graphpool(batch_graph)  # 图池化预处理
        # 预处理邻接结构
        padded_neighbor_list = None
        Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph) if self.neighbor_pooling_type != "max" else None

        # 特征传播
        hidden_rep = [x]

        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == "max":
                h_next = self.next_layer_eps(h_current, layer, padded_neighbor_list)
            else:
                h_next = self.next_layer(h_current, layer, Adj_block)

            # 动态网格更新
            if self.dynamic_grid and self.training and epoch is not None and (epoch % 5 == 0):
                with torch.no_grad():
                    self.kans[layer].update_grid(h_current)

            # 计算注意力权重
            src, dst = edge_index
            h_src = h_next[src]
            h_dst = h_next[dst]
            alpha = self.attentions[layer](h_src, h_dst, edge_index).squeeze()  # (num_edges,)

            # 同步筛选边索引和注意力权重
            rows, cols = edge_index
            values = alpha
            # Adj_att = torch.sparse_coo_tensor(
            #     torch.stack([rows, cols]),
            #     values,
            #     size=(h_next.size(0), h_next.size(0)),
            #     device=self.device
            # )
            # # 注意力池化
            # pooled_att = torch.spmm(Adj_att, h_next)
            pooled_att = torch.zeros_like(h_next)
            pooled_att.index_add_(0, rows, alpha.unsqueeze(-1) * h_next[cols])  # 按行累加

            # 应用残差连接
            if self.residual_type == "gate":
                # 门控残差连接
                gate = torch.sigmoid(self.res_gates[layer])
                residual = self.skip_cons[layer](hidden_rep[-1])
                h_current = gate * h_next + (1 - gate) * residual
            else:
                # 默认简单相加
                h_current = h_next + pooled_att

            hidden_rep.append(h_current)

        score_over_layer = 0
        for layer_idx in range(self.num_layers):
            pooled_h = torch.spmm(graph_pool, hidden_rep[layer_idx])
            score_over_layer += F.dropout(
                self.linears_prediction[layer_idx](pooled_h),
                self.final_dropout,
                training=self.training
            )

        return score_over_layer