# ~~~IMPORT MODULES~~~ #
import torch
from torch_geometric.data import Data


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
print('Directed Graph: %s' % data.is_directed())

# Convert to GPU graph
if torch.cuda.is_available():
    device = torch.device('cuda')
    data = data.to(device)
