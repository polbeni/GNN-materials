# Pol Benítez Colominas, March 2024
# Universitat Politècnica de Catalunya

# Crystal graph neural network (cgnn) model

import os
import csv

import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

# generate a list with all the graph structures names
file_list = []

with open('graphs-bg.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        file_list.append(row[0])

# save all the normalized graphs in a object and assign the value of each graph (band gap)
dataset_graphs = []

path_data = 'normalized_graphs/'

df_materials = pd.read_csv('graphs-bg.csv')

for filename in file_list:
    data = torch.load(path_data + filename + '.pt')
    bg = df_materials.loc[df_materials['material-id'] == filename, 'bandgap'].values[0]
    data.y = float(bg)
    dataset_graphs.append(data)

print('All the graphs loaded in a Data structure!')
print('')

# define the graph neural network model
class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)  # Global pooling
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
    
# define the model
num_features = dataset_graphs[0].num_features
num_classes = 1  # Assuming you are predicting a single parameter
hidden_channels = 64
model = GNN(num_features, hidden_channels, num_classes)

# define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# split train and test set
train_size = int(0.8 * len(dataset_graphs))
test_size = len(dataset_graphs) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset_graphs, [train_size, test_size])

# data loader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# train loop
model.train()
for epoch in range(50):  # Number of epochs
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}')

# evaluation
model.eval()
total_loss = 0
with torch.no_grad():
    for data in test_loader:
        out = model(data.x, data.edge_index, data.batch)
        total_loss += criterion(out, data.y.float()).item()

print(f'Average Test Loss: {total_loss / len(test_loader)}')