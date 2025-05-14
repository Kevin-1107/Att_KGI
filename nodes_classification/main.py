import argparse

import matplotlib.pyplot as plt
import torch.optim as optim
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import *

from nodes_classification.models.gin import GIN
from nodes_classification.models.gin_kan import GIN_KAN


class GraphData:
    def __init__(self, node_features, edge_index, labels=None, train_mask=None, val_mask=None, test_mask=None):
        self.node_features = node_features
        self.edge_index = edge_index  # 统一使用edge_index命名
        self.labels = labels  # 节点标签
        self.train_mask = train_mask  # 训练掩码
        self.val_mask = val_mask  # 验证掩码
        self.test_mask = test_mask  # 测试掩码
        self.g = node_features.shape[0]  # 新增 g 属性（节点数）
        self.neighbors = [[] for _ in range(self.g)]
        # 填充邻居列表（根据邻接矩阵）
        for src, dst in edge_index.t().tolist():
            self.neighbors[src].append(dst)
        self.max_neighbor = max(len(n) for n in self.neighbors)

def prepare_batch_graph(adj, features):
    if isinstance(adj, torch.Tensor) and adj.is_sparse:
        edge_index = adj.coalesce().indices()
    else:
        edge_index = adj.coalesce().indices()
    return [GraphData(features, edge_index)]

def train_and_evaluate(model, batch_graph, labels, idx_train, idx_val, device, model_name, patience=5):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience_counter = 0  # 用于记录验证集损失未改善的次数
    # 添加梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(batch_graph)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss_train.item())

        # Calculate training accuracy
        _, preds = output[idx_train].max(1)
        correct = preds.eq(labels[idx_train]).sum().item()
        train_acc = correct / idx_train.size(0)
        train_accuracies.append(train_acc)

        # Calculate validation loss and accuracy
        model.eval()
        with torch.no_grad():
            output = model(batch_graph)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            val_losses.append(loss_val.item())

            _, preds = output[idx_val].max(1)
            correct = preds.eq(labels[idx_val]).sum().item()
            val_acc = correct / idx_val.size(0)
            val_accuracies.append(val_acc)

        model.train()

        # 早停机制：如果验证损失没有改善，计数加1；否则，重置计数
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1} for {model_name}')
            break

        if epoch % 10 == 0:
            print(f'{model_name} Epoch: {epoch + 1:04d}',
                  f'loss_train: {loss_train.item():.4f}',
                  f'train_acc: {train_acc:.4f}',
                  f'loss_val: {loss_val.item():.4f}',
                  f'val_acc: {val_acc:.4f}')

    return train_losses, val_losses, train_accuracies, val_accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
    parser.add_argument('--seed', type=int, default=10, help='seed')
    args = parser.parse_args()


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data = Dataset(root='./dataset', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # Convert sparse matrices to dense
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    labels = torch.LongTensor(labels).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    batch_graph = prepare_batch_graph(adj, features)

    models = {

        'KGI': GIN_KAN(
            num_layers=2,
            num_mlp_layers=2,
            input_dim=features.shape[1],
            hidden_dim=64,
            output_dim=labels.max().item() + 1,
            final_dropout=0.5,
            learn_eps=True,
            graph_pooling_type=None,
            neighbor_pooling_type='sum',
            device=device,
            dynamic_grid=True,
            max_degree=100,
            residual_type="auto"
            ),

        'GIN': GIN(
            nfeat=features.shape[1],  # 使用原始参数名
            nhid=64,
            nclass=labels.max().item() + 1,
            dropout=0.5,
            learn_eps=True,
            device=device),
        # 'GCN': GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, dropout=0.5, with_relu=False,
        #            with_bias=False, weight_decay=5e-4, device=device),
    }

    # Store loss and accuracy values for each model
    train_loss_values = {}
    val_loss_values = {}
    train_accuracy_values = {}
    val_accuracy_values = {}

    for model_name, model in models.items():
        model.attention = False  # Explicitly define the attention attribute
        train_losses, val_losses, train_accuracies, val_accuracies = train_and_evaluate(
            model, batch_graph, labels, idx_train, idx_val, device, model_name
        )
        train_loss_values[model_name] = train_losses
        val_loss_values[model_name] = val_losses
        train_accuracy_values[model_name] = train_accuracies
        val_accuracy_values[model_name] = val_accuracies

    # Plot the loss curves
    plt.figure()
    for model_name, train_losses in train_loss_values.items():
        plt.plot(range(len(train_losses)), train_losses, label=f'{model_name} Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train Loss of Different Models')
    plt.show()

    plt.figure()
    for model_name, val_losses in val_loss_values.items():
        plt.plot(range(len(val_losses)), val_losses, label=f'{model_name} Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss of Different Models')
    plt.show()

    # Plot the accuracy curves
    plt.figure()
    for model_name, train_accuracies in train_accuracy_values.items():
        plt.plot(range(len(train_accuracies)), train_accuracies, label=f'{model_name} Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train Accuracy of Different Models')
    plt.show()

    plt.figure()
    for model_name, val_accuracies in val_accuracy_values.items():
        plt.plot(range(len(val_accuracies)), val_accuracies, label=f'{model_name} Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy of Different Models')
    plt.show()

    # Evaluate each model
    for model_name, model in models.items():
        model.eval()
        output = model(batch_graph)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print(f"{model_name} Test set results: accuracy= {acc_test.item():.4f}")

if __name__ == '__main__':
    main()
