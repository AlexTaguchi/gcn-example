# ~~~IMPORT MODULES~~~ #
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


# ~~~GRAPH OF H2O~~~ #
# Define one-hot node features of H-O-H
x = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float)

# Define directed edges between the source (row 1) and target (row 2)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

# Store graph in variable
data = Data(x=x, edge_index=edge_index)

# Accessing graph attributes
print('\n%===ACCESSING GRAPH ATTRIBUTES===%')
print('Graph keys: %s' % str(data.keys))
print('Node features:')
print(data['x'])
print('\'edge_attr\' in data: %s' % str('edge_attr' in data))

# Useful graph methods
print('\n%===USEFUL GRAPH METHODS===%')
print('Number of nodes: %d' % data.num_nodes)
print('Number of features: %d' % data.num_features)
print('Number of edges: %d' % data.num_edges)
print('Isolated nodes: %s' % data.contains_isolated_nodes())
print('Self loops: %s' % data.contains_self_loops())
print('Directed graph: %s' % data.is_directed())

# Convert to GPU graph
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)


# ~~~BENCHMARK DATASET~~~ #
# Import benchmark dataset
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

# Dataset attributes
print('\n%===DATASET ATTRIBUTES===%')
print('Number of samples: %d' % len(dataset))
print('Number of features: %d' % dataset.num_features)
print('Number of labels: %d' % dataset.num_classes)
print('Undirected graph: %s' % dataset[0].is_undirected())
print('First sample of dataset:')
print(dataset[0])

# Dataset manipulations
dataset = dataset.shuffle()
train_dataset = dataset[:540]
test_dataset = dataset[540:]


# ~~~MINI-BATCHES~~~ #
# Set up a dataloader to handle mini-batching
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Loop through some mini-batches
print('\n%===MINI-BATCHES===%')
counter = 0
for batch in loader:
    counter += 1
    print('Mini-batch %d: %s (%d graphs)' % (counter, str(batch), batch.num_graphs))

    # Batch is a column vector of graph identifiers for all nodes in the batch:
    # batch = [[0], ... [0], [1], ... [n-2], [n-1], ... [n-1]]
    # Scatter functions allow you to apply reduction operations to each graph
    print('Original size of batch: %s' % str(batch.x.size()))
    x = scatter_mean(batch.x, batch.batch, dim=0)
    print('Reduced size of batch: %s' % str(x.size()))
    if counter == 3:
        break


# ~~~GCN TRAINING~~~ #
# Import Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')


# Two-layer GCN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Instantiate model
model = Net().to(device)
data = dataset[0].to(device)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Train model for 200 epoches
print('\n%===TRAINING===%')
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch:3d}: Loss = {loss:.4f}')

# Evaluate model performance
model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print(f'Final Accuracy: {acc:.4f}')
