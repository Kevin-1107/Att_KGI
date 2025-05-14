import gc
import networkx as nx
import numpy as np
from pathlib import Path
import torch
from sklearn.model_selection import StratifiedKFold

class GraphStreamLoader:
    def __init__(self, dataset, indices, batch_size, degree_as_tag):
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.degree_as_tag = degree_as_tag
        self.file_path = Path(f'dataset/{dataset}/{dataset}.txt')
        self._precompute_offsets()
        self._precompute_metadata()

    def _precompute_offsets(self):
        """预计算文件偏移和基础元数据(修复版本)"""
        self.offsets = []
        self.node_counts = []
        self.labels = []

        with self.file_path.open('rb') as f:  # 二进制模式
            self.total_graphs = int(f.readline().decode().strip())
            for _ in range(self.total_graphs):
                self.offsets.append(f.tell())
                header = f.readline().decode().strip().split()
                n_nodes = int(header[0])
                self.node_counts.append(n_nodes)
                self.labels.append(int(header[1]))

                # 精确跳过节点行
                for _ in range(n_nodes):
                    f.readline()

    def _precompute_metadata(self):
        """预计算标签映射和特征字典(修复版本)"""
        self.label_map = {l: i for i, l in enumerate(sorted(set(self.labels)))}
        self.feat_dict = {}

        with self.file_path.open('rb') as f:
            f.readline()  # Skip total graphs
            for _ in range(self.total_graphs):
                header = f.readline().decode().strip().split()
                n_nodes = int(header[0])

                for _ in range(n_nodes):
                    parts = f.readline().decode().strip().split()
                    tag = int(parts[0])
                    if tag not in self.feat_dict:
                        self.feat_dict[tag] = len(self.feat_dict)

    def __iter__(self):
        """流式加载器实现"""
        for i in range(0, len(self.indices), self.batch_size):
            batch = []
            for idx in self.indices[i:i + self.batch_size]:
                batch.append(self._load_single_graph(idx))
                if len(batch) % 10 == 0:
                    gc.collect()  # 定期垃圾回收
            yield batch
            del batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _load_single_graph(self, idx):
        """按需加载单个图"""
        with self.file_path.open('rb') as f:
            f.seek(self.offsets[idx])

            # 读取图头信息
            header = f.readline().decode().strip().split()
            n_nodes = int(header[0])
            label = self.label_map[int(header[1])]

            # 读取节点信息
            g = nx.Graph()
            node_tags = []
            for j in range(n_nodes):
                parts = f.readline().decode().strip().split()
                tag = self.feat_dict[int(parts[0])]
                node_tags.append(tag)
                # 添加节点和边
                g.add_node(j)
                neighbors = list(map(int, parts[2:2 + int(parts[1])]))  # 修正索引
                for k in neighbors:
                    g.add_edge(j, k)  # 确保边被正确添加

            # 生成特征
            if self.degree_as_tag:
                degrees = list(dict(g.degree).values())
                node_features = torch.tensor(degrees, dtype=torch.float32).unsqueeze(-1)
            else:
                num_tags = len(self.feat_dict)
                node_features = torch.zeros(n_nodes, num_tags)
                node_features[range(n_nodes), node_tags] = 1

            return S2VGraph(g, label, node_tags, node_features)  # 确保传递g参数

    def _preprocess_batch(self, batch_graph):
        # 异步预处理
        processed = []
        for g in batch_graph:
            # 特征优化
            features = self.feature_cache.pop(0)
            # 边索引优化
            edge_index = torch.from_numpy(g.edge_index).long().to(self.device)
            # 图结构优化
            processed.append((features, edge_index, g.label))
        return processed

    def _preprocess_graph(self, g):
        if not hasattr(g, 'optimized'):
            g.node_features = g.node_features.to(torch.float16)  # 半精度
            g.optimized = True
        return g

class S2VGraph(object):
    __slots__ = ['num_nodes', 'label', 'node_tags', 'node_features', 'edge_index', 'neighbors']  # 使用__slots__减少内存开销

    def __init__(self, g, label, node_tags, node_features):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.num_nodes = len(g)
        self.label = label
        self.node_tags = node_tags
        self.node_features = node_features
        self.edge_index = self._get_edge_index(g)
        self.neighbors = [list(g.neighbors(i)) for i in range(self.num_nodes)]

        del g  # 立即释放原始图对象

    def _get_edge_index(self, g):
        edges = list(g.edges())
        return torch.LongTensor(edges).t().contiguous()

    def _get_node_features(self, node_tags):
        features = torch.FloatTensor(node_tags)
        features = (features - features.mean()) / (features.std() + 1e-6)
        features = torch.nan_to_num(features, nan=0.0)  # 强制处理NaN
        return features.requires_grad_(True)


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    loader = GraphStreamLoader(dataset, range(10000), 1, degree_as_tag)  # 假设最多10000个图
    return [loader._load_single_graph(i) for i in range(loader.total_graphs)], len(loader.label_map)

def separate_data(graph_list, seed, fold_idx):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    labels = [graph.label for graph in graph_list]
    splits = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, test_idx = splits[fold_idx]
    train_graphs = [graph_list[i] for i in train_idx]
    test_graphs = [graph_list[i] for i in test_idx]
    return train_graphs, test_graphs
