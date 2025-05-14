import torch
import torch.nn as nn
import torch.nn.functional as F

###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.3):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.mlps = nn.ModuleList()

        self.layers = nn.ModuleList()

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Multi-layer model
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

            # 批量归一化层（可选）
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim)
                for _ in range(num_layers - 1)
            ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)