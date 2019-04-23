import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing


class EdgeConv(MessagePassing):
    def __init__(self, F_in, F_out):
        super(EdgeConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.mlp = Sequential(Linear(2 * F_in, F_out),
                              ReLU(),
                              Linear(F_out, F_out))

    def forward(self, x, edge_index):
        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)  # shape [N, F_out]

    def message(self, x_i, x_j):
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # shape [E, 2 * F_in]
        return self.mlp(edge_features)  # shape [E, F_out]

import numpy as np

conv = EdgeConv(16, 32)

# Simple random node features
x = torch.zeros((5, 16), dtype=torch.float)
x[np.arange(5), np.random.randint(0, 16, 5)] = 1

# Generate simple edge index
edge_unit = torch.tensor([[0, 1],
                          [1, 0]], dtype=torch.long)
edge_index = torch.cat(tuple([i + edge_unit for i in range(5 - 1)]), dim=1)

# Reproduce KeyError: 'size'
x = conv(x, edge_index)
