import glob

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

# create an object to save all the graphs
graphs = []

# create a list with all the graphs
structures_path = '/home/pol/Documents/work/crystal-graph-neural-networks/normalized_graphs'
structures_list = glob.glob(f'{structures_path}mp-*')

# save all the graphs in graphs variable
num_strucuture = 1
for graph_path in structures_list:
    loaded_graph_data = torch.load(graph_path)

    graphs.append(loaded_graph_data)

    print(f'Graph loaded {num_strucuture} of {len(structures_list)}')

    num_strucuture = num_strucuture + 1



class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(graphs.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, graphs.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

model = GCN()
print(model)