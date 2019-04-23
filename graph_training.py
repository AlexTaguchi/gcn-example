# Import modules
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter_mean

# Synthetic training set of 1000 molecules
molecules = []
for _ in range(1000):

    # 3 to 10 atoms (nodes) per molecule
    atom_count = np.random.randint(3, 11)

    # 3 features (atom types) per node
    node_feature = torch.zeros((atom_count, 3), dtype=torch.float)
    atom_types = np.random.randint(0, 3, atom_count)
    node_feature[np.arange(atom_count), atom_types] = 1

    # Molecule is a string of atoms with undirected edges
    edge_unit = torch.tensor([[0, 1],
                              [1, 0]], dtype=torch.long)
    edge_index = torch.cat(tuple([i + edge_unit for i in range(atom_count - 1)]), dim=1)

    # Label molecule as a hit if atom type 0 appears three times in a row
    for i in range(atom_count - 1):
        if node_feature[i:i + 2, 0].sum() == 2:
            y = torch.tensor([1], dtype=torch.long)
            break
        else:
            y = torch.tensor([0], dtype=torch.long)

    # Store graph in variable
    molecules.append(Data(x=node_feature, edge_index=edge_index, y=y))

# Create dataloaders
train_loader = DataLoader(molecules[:800], batch_size=50)
test_loader = DataLoader(molecules[800:], batch_size=200)


# Custom Graph Convolutional Network
class MyGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MyGCNConv, self).__init__(aggr='mean')
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        # x = [nodes, in_channels]
        # edge_index = [2, directed edges]

        # Linear transformation of central node feature matrix
        x_i = self.lin1(x)

        # Linear transformation of neighbor node feature matrix
        x_j = self.lin2(x)

        # Propagate messages
        return self.propagate(edge_index, x=x_j, center=x_i)

    def message(self, x_j, edge_index):
        # x_j = [directed edges, out_channels]

        # Normalize node features
        row, col = edge_index
        deg = degree(row, dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, center):
        # aggr_out = [nodes, out_channels]

        # Sum new central and neighbor node embeddings
        return center + aggr_out


# Two-layer GCN with MLP classfier
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = MyGCNConv(3, 16)
        self.conv2 = MyGCNConv(16, 16)
        self.lin1 = nn.Linear(16, 16)
        self.lin2 = nn.Linear(16, 2)

    def forward(self, batch):

        # Retrieve node features and edge indices from batch
        x, edge_index = batch.x, batch.edge_index

        # Graph convolutional layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Average node latent representations
        x = scatter_mean(x, batch.batch, dim=0)

        # Multilayer perceptron
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x


# Instantiate model
model = Net()

# Initialize optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model for 200 epoches
print('\n%===TRAINING===%')
model.train()
for epoch in range(200 + 1):
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch:3d}: Loss = {loss:.4f}')

# Evaluate model performance
model.eval()
for batch in test_loader:
    _, pred = model(batch).max(dim=1)
    correct = pred.eq(batch.y).sum().item()
    acc = correct / len(batch.y)
    print(f'Final Accuracy: {acc:.3f}')
