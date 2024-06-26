# Pol Benítez Colominas, February 2024 - May 2024
# Universitat Politècnica de Catalunya

# Code to normalize the generated crystal graphs

# system modules
import os
import shutil
import glob

# pytorch and torch geometric modules
import torch
from torch_geometric.data import Data

# create an object to save all the graphs
graphs = []

# create a list with all the graphs
structures_path = 'graph_structures/'
structures_list = glob.glob(f'{structures_path}cap-*')

# save all the graphs in graphs variable
num_strucuture = 1
for graph_path in structures_list:
    loaded_graph_data = torch.load(graph_path)

    graphs.append(loaded_graph_data)

    print(f'Graph loaded {num_strucuture} of {len(structures_list)}')

    num_strucuture = num_strucuture + 1

# concatenate node and edge features of all the graphs
node_features = torch.cat([graph.x for graph in graphs], dim=0)
edge_features = torch.cat([graph.edge_attr for graph in graphs], dim=0)

# normalize node features
max_node = torch.tensor([ 88.0000,   3.9800, 244.0000, 298.0000])
min_node = torch.tensor([ 0.0000,  0.7900,  1.0080, 42.0000])
node_features = (node_features - min_node)/(max_node - min_node)

# normalize edge features
max_edge = 5.5000
min_edge = 0.6932
edge_features = (edge_features - min_edge)/(max_edge - min_edge)

# save the normalization values for future data
normalization_parameters = open('normalized_parameters.txt', 'w')
normalization_parameters.write('max_node  min_node  max_edge  min_edge\n')
normalization_parameters.write(f'{max_node}  {min_node}  {max_edge}  {min_edge}')
normalization_parameters.close()

print('')
print('Graphs normalized!!')
print('')

# normalize all the graphs and save them in torch binary files
if os.path.exists('normalized_graphs'):
    shutil.rmtree('normalized_graphs')
os.mkdir('normalized_graphs')

node_idx = 0
edge_idx = 0

num_strucuture = 1
for graph in graphs:
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges

    normalized_x = (node_features[node_idx: node_idx + num_nodes]).clone().detach()
    normalized_edge_attr = (edge_features[edge_idx: edge_idx + num_edges]).clone().detach()
    normalized_edge_index = (graph.edge_index).clone().detach()
    y_data = (graph.y).clone().detach()

    normalized_graph = Data(x=normalized_x, edge_attr=normalized_edge_attr, edge_index=normalized_edge_index, y=y_data)

    node_idx = node_idx + num_nodes
    edge_idx = edge_idx + num_edges

    name_to_save = structures_list[num_strucuture - 1].split('/')[1].split('.')[0]

    torch.save(normalized_graph, 'normalized_graphs/' + name_to_save + '.pt')

    print(f'Normalized graph {num_strucuture} of {len(graphs)}')

    num_strucuture = num_strucuture + 1