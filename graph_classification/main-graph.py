import os
import gc
import psutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
from models.gin import GIN
from models.gin_kan import GIN_KAN

from torch.cuda.amp import GradScaler

# 定义损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    # 设置模型为训练模式
    model.train()
    scaler = GradScaler(enabled=False)
    args.batch_size = calculate_safe_batch_size(train_graphs, device)  # 动态调整
    # 创建数据加载器并循环使用
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')
    loss_accum = 0
    optimizer.zero_grad()

    for pos in pbar:
        # 确保每次迭代都有loss值
        if pos % 5 == 0 and pos > 0:
            args.batch_size = min(args.batch_size + 2, 32)
            gc.collect()
            if device.type == 'cpu':
                used_mem = psutil.Process().memory_info().rss / 1024 ** 2
                pbar.set_postfix_str(f"Mem: {used_mem:.1f}MB")
        # 随机选择一个批次的训练图
        selected_idx = np.random.choice(len(train_graphs), args.batch_size, replace=False)
        batch_graph = [train_graphs[i] for i in selected_idx]
        output = model(batch_graph, epoch=epoch)
        labels = torch.LongTensor([g.label for g in batch_graph]).to(device)

        loss = criterion(output, labels)
        loss = loss / max(1, args.accumulation_steps)
        loss.backward()

        if (pos + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0 if device.type == 'cpu' else 2.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # 更彻底的内存释放

        loss_accum += loss.item()
        pbar.set_description(f'Epoch {epoch} Loss: {loss.item():.4f}')

        # 更新进度条显示的描述
        pbar.set_description('epoch: %d' % epoch)

    # 计算平均损失
    average_loss = loss_accum / total_iters
    print("loss training: %f" % average_loss)

    # return average_loss
    return loss_accum / total_iters


# 在测试时以小批次的方式传递数据以避免内存溢出（不进行反向传播）
def pass_data_iteratively(model, graphs, minibatch_size=64):
    # 设置模型为评估模式
    model.eval()
    # 初始化输出列表
    output = []
    # 获取所有图的索引
    idx = np.arange(len(graphs))
    # 以小批次的方式传递数据
    with torch.no_grad():
        for i in range(0, len(graphs), minibatch_size):
            sampled_idx = idx[i:i + minibatch_size]
            if len(sampled_idx) == 0:
                continue
        # 获取该批次的图并传递给模型，获取输出
            output.append(model([graphs[j] for j in sampled_idx]))
    # 返回所有输出的拼接结果
    return torch.cat(output, 0)


def test(args, model, device, train_graphs, test_graphs, epoch):
    # 在训练集上进行测试
    acc_train = eval(args, model, device, train_graphs)
    # 在测试集上进行测试
    acc_test = eval(args, model, device, test_graphs)

    # 打印训练和测试的准确率
    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test


def eval(args, model, device, graphs):
    # 以小批次方式传递数据，获取输出
    output = pass_data_iteratively(model, graphs)
    # 获取所有图的标签
    labels = torch.LongTensor([graph.label for graph in graphs]).to(device)
    # 计算预测结果的最大值索引
    pred = output.max(1, keepdim=True)[1]
    # 计算准确率
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    return correct / len(graphs)

def calculate_safe_batch_size(graphs, device):
    avg_nodes = np.mean([g.num_nodes for g in graphs])
    # device.type == 'cpu':
    # 更保守的估算：每个节点消耗200B内存
    mem_per_node = 200
    mem_per_graph = avg_nodes * mem_per_node
    # 预留300MB安全空间
    if device.type == 'cpu':
        available_mem = psutil.virtual_memory().available - 300 * 1024 ** 2
    else:
        available_mem = torch.cuda.mem_get_info()[0] - 300 * 1024 ** 2

    # 计算安全批量大小
    safe_size = max(1, int(available_mem / mem_per_graph))
    return min(safe_size, 8)  # 最大不超过8


def run_experiment(args, model, device):
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    avg_nodes = np.mean([g.num_nodes for g in train_graphs])
    residual_type = "gate" if avg_nodes < 100 else "add"
    print(f"数据集的平均节点数: {avg_nodes}")
    print(f"采用的残差连接方式: {residual_type}")

    model = model(
        num_layers=args.num_layers,
        num_mlp_layers=args.num_mlp_layers,
        input_dim=train_graphs[0].node_features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        final_dropout=args.final_dropout,
        learn_eps=args.learn_eps,
        graph_pooling_type=args.graph_pooling_type,
        neighbor_pooling_type=args.neighbor_pooling_type,
        device=device,
        residual_type=residual_type
    ).to(device)

    main_params = []
    att_params = []

    # 遍历所有参数并分类
    for name, param in model.named_parameters():
        if 'att_kan' in name or 'attentions' in name:
            att_params.append(param)
        else:
            main_params.append(param)

    optimizer = optim.AdamW([
        {'params': main_params, 'weight_decay': 1e-4},
        {'params': att_params, 'lr': args.lr * 0.1}
    ], lr=args.lr, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * args.iters_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # 在加载数据后添加动态批处理
    args.batch_size = calculate_safe_batch_size(train_graphs, device)
    print(f"自动设置安全批量大小: {args.batch_size}")

    best_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        if isinstance(model, GIN_KAN):
            if epoch < 10:
                # 遍历所有注意力层
                for layer in range(model.num_layers - 1):  # 关键修改
                    for param in model.attentions[layer].parameters():
                        param.requires_grad = False
            else:
                for layer in range(model.num_layers - 1):
                    for param in model.attentions[layer].parameters():
                        param.requires_grad = True

        train(args, model, device, train_graphs, optimizer, epoch)
        scheduler.step()
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)
        if acc_test > best_test_acc:
            best_test_acc = acc_test

    return best_test_acc


def main():
    parser = argparse.ArgumentParser(description='GIN Variants with KAN')
    parser.add_argument('--dataset', type=str, default="PTC", help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50, help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 350)')
    parser.add_argument('--num_layers', type=int, default=5, help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='number of MLP layers (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units (default: 64)')
    parser.add_argument('--seed', type=int, default=0, help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0, help='the index of fold in 10-fold validation. Should be less than 10. (default: 0)')
    parser.add_argument('--filename', type=str, default="results.txt", help='output file')
    parser.add_argument('--learn_eps', action="store_true", help='Whether to learn the epsilon weighting')
    parser.add_argument('--degree_as_tag', action="store_true", help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", help='type of graph pooling: sum, average or max')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", help='type of neighboring pooling: sum, average or max')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--num_runs', type=int, default=1, help='number of runs to perform')
    parser.add_argument('--clip_grad', type=float, default=2.0)
    parser.add_argument('--stable_lr', type=float, default=0.001)

    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--att_dropout', type=float, default=0.2, help='Attention dropout rate')
    parser.add_argument('--mlp_dropout', type=float, default=0.3, help='MLP internal dropout')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--use_attention', action='store_false', help='Enable attention mechanism in GIN')
    parser.add_argument('--residual_type', type=str, default="auto", help='Residual connection type: auto/gate/add')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    for model_name in ['GIN', 'KGI']:
        if model_name == 'GIN':
            model = GIN
        else:
            model = GIN_KAN

        all_test_accs = []
        for run in range(args.num_runs):
            print(f"Run {run + 1}/{args.num_runs}")
            test_acc = run_experiment(args, model, device)
            all_test_accs.append(test_acc)

        avg_test_acc = np.mean(all_test_accs)
        std_test_acc = np.std(all_test_accs)

        with open(args.filename, 'a') as f:
            f.write(f"{model_name} Test Accuracies: {all_test_accs}\n")
            f.write(f"{model_name} Average Test Accuracy: {avg_test_acc}\n")
            f.write(f"{model_name} Standard Deviation of Test Accuracy: {std_test_acc}\n\n")

        print(f"{model_name} Average Test Accuracy: {avg_test_acc}")
        print(f"{model_name} Standard Deviation of Test Accuracy: {std_test_acc}")

if __name__ == '__main__':
    # 启用内存优化模式
    torch.backends.cudnn.benchmark = False  # 对CPU无影响但防止意外内存分配
    torch.backends.cuda.max_split_size_mb = 128  # 控制缓存分配

    # 设置内存限制（需要psutil）
    mem = psutil.virtual_memory()
    limit = int(mem.total * 0.8)
    torch.cuda.set_per_process_memory_fraction(0.8) if torch.cuda.is_available() else None
    os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'  # 减少线程竞争

    main()