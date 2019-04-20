# Import modules
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

# Synthetic training set of 1000 molecules
molecules = []
for _ in range(1000):

    # Number of atoms (nodes) in molecule
    atom_count = np.random.randint(3, 11)

    # Node features
    x = torch.zeros((atom_count, 3), dtype=torch.float)
    atom_types = np.random.randint(0, 3, atom_count)
    x[np.arange(atom_count), atom_types] = 1

    # String of atoms with undirected edges
    edge_unit = torch.tensor([[0, 1],
                              [1, 0]], dtype=torch.long)
    edge_index = torch.cat(tuple([i + edge_unit for i in range(atom_count - 1)]), dim=1)

    # Label molecule as a hit if three zeros in a row
    for i in range(atom_count - 2):
        if x[i:i + 3].sum() > 2:
            y = torch.tensor([1], dtype=torch.long)
        else:
            y = torch.tensor([0], dtype=torch.long)

    # Store graph in variable
    molecules.append(Data(x=x, edge_index=edge_index, y=y))

# Create dataloader
loader = DataLoader(molecules[:800], batch_size=50)


# Two-layer GCN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = scatter_mean(x, batch.batch, dim=0)

        return F.log_softmax(x, dim=1)


# Instantiate model
model = Net()

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Train model for 200 epoches
print('\n%===TRAINING===%')
model.train()
for epoch in range(10):
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
    # if epoch % 20 == 0:
    print(f'Epoch {epoch:3d}: Loss = {loss:.4f}')

# # Evaluate model performance
# model.eval()
# _, pred = model(data).max(dim=1)
# correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
# acc = correct / data.test_mask.sum().item()
# print(f'Final Accuracy: {acc:.4f}')

