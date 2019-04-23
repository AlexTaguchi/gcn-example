import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add') # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, x_j.size(0), dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j


import numpy as np

conv = GCNConv(16, 32)

# Simple random node features
x = torch.zeros((5, 16), dtype=torch.float)
x[np.arange(5), np.random.randint(0, 16, 5)] = 1

# Generate simple edge index
edge_unit = torch.tensor([[0, 1],
                          [1, 0]], dtype=torch.long)
edge_index = torch.cat(tuple([i + edge_unit for i in range(5 - 1)]), dim=1)

# Reproduce KeyError: 'size'
x = conv(x, edge_index)

