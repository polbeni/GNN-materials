# Pol Benítez Colominas, May 2025
# Universitat Politècnica de Catalunya

# Compute predictions of test set for the selected model from hyperparameter testing


################################# LIBRARIES ###############################
import os
import csv
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool
###########################################################################



################################ PARAMETERS ###############################
model_name = 'model1_1e-2_128_06'                                       # Name of the selected model

hidden = 128                                                            # Number of hidden channels in the convolutional layers
dropout = 0.6                                                           # Fraction of elements to dropout

path_to_graphs = '../materials-dataset-new/normalized_graphs/'          # Path to the normalized graphs
path_to_csv = '../materials-dataset-new/graphs-bg.csv'                  # Path to csv file with graphs names and value
path_to_test = 'test_graphs.txt'                                        # Path to text file with the name of test set graphs

model_path = 'trained_model'                                            # Path or name of the final trained model

outputs_dir = 'outputs_file_final/'                                     # Path to dir where outputs are saved

seed_model_torch = 12345                                                # Seed for the model

min_bg_boolean = True                                                   # Ignore materials with band gaps smaller than the desired value
min_bg = 0.4                                                            # Minimum band gap value
###########################################################################



########################## MODEL ARCHITECTURE #############################
class CGCNN(torch.nn.Module):
    """
    Graph Convolution Neural Network model
    """

    def __init__(self, features_channels, hidden_channels, seed_model, dropout):
        super(CGCNN, self).__init__()
        torch.manual_seed(seed_model)

        self.dropout = dropout

        # Convolution layers
        self.conv1 = GraphConv(features_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)

        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        # Linear layers
        self.lin1 = Linear(hidden_channels, 16)
        self.lin2 = Linear(16, 1)

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embedding
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = x.relu()

        # Mean pooling to reduce dimensionality
        x = global_max_pool(x, batch)  

        # Apply neural network for regression prediction problem

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.sigmoid()

        return x
###########################################################################



################################ FUNCTIONS ################################
def model_prediction(model, graph, norm_ct):
    """
    Returns the model prediction for a given graph

    Inputs:
        model: model to use in the prediction
        graph: graph structure to make prediction
        norm_ct: array with normalization constants with the format [min, max]
    """

    graph.x = graph.x.to(device).float()
    graph.edge_index = graph.edge_index.to(device).long()
    graph.edge_attr = graph.edge_attr.to(device).float()
    graph.y = graph.y.to(device).float()

    graph = graph.to(device)

    model = model.to(device).float()

    # Make prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch).to(device)

    # Multiply the predicted value by the normalization constant
    prediction_desnorm = prediction[0][0]*(norm_ct[1] - norm_ct[0]) + norm_ct[0]

    return prediction_desnorm
###########################################################################



################################### MAIN ##################################
# Check if a GPU (CUDA) is available 
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU not available. Using CPU.")


# Load the normalized graphs of the test set and save them in an array
file_list = []

with open(path_to_csv, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        if min_bg_boolean == True:
            if float(row[1]) > min_bg: 
                file_list.append(row[0])
        else:           
            file_list.append(row[0])

dataset_graphs = []

df_materials = pd.read_csv(path_to_csv)

test_materials =[]
with open(path_to_test, 'r') as file:
    for row in file:
        test_materials.append(row.split()[0])

for filename in file_list:
    if filename in test_materials:
        data = torch.load(path_to_graphs + filename + '.pt')
        dataset_graphs.append(data)

print(f'A total of {len(dataset_graphs)} graphs loaded.')


# Normalize the outputs using the normalization constants used in the training
outputs_graphs = torch.cat([graph.y for graph in dataset_graphs], dim=0)

with open('output_normalization.txt', 'r') as file:
    file.readline()
    line = file.readline()

    max_output = float(line.split()[0])
    min_output = float(line.split()[1])

outputs_graphs = (outputs_graphs - min_output)/(max_output - min_output)

num_struc = 0
for graph in dataset_graphs:
    graph.y = outputs_graphs[num_struc]

    num_struc = num_struc + 1


# Create the model
model = CGCNN(features_channels=dataset_graphs[0].num_node_features, hidden_channels=hidden, seed_model=seed_model_torch, dropout=dropout)
model = model.to(device)


# Load the trained model
model.load_state_dict(torch.load(model_name + '/' + model_path))


# Put the model in evaluation mode
model.eval()


# Create the folder to save the outputs
if os.path.exists(outputs_dir):
    shutil.rmtree(outputs_dir)
os.mkdir(outputs_dir)


# Compute predictions for the test graphs and save the results in a text file
real_value_test = []
predicted_value_test = []

model.eval()

for num_graph in range(len(dataset_graphs)):
    graph = dataset_graphs[num_graph]

    real_value_test.append(graph.y*(max_output - min_output) + min_output)

    graph.x = graph.x.to(device).float()
    graph.edge_index = graph.edge_index.to(device).long()
    graph.edge_attr = graph.edge_attr.to(device).float()
    graph.y = graph.y.to(device).float()
    graph = graph.to(device)

    batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)

    model = model.to(device).float()

    # Make prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch).to(device)

    # Multiply the predicted value by the normalization constant
    predicted_value_test.append(prediction[0][0]*(max_output - min_output) + min_output)


# Save the prediction vs DFT results
with open(outputs_dir + 'prediction_vs_DFT_test.txt', 'w') as file:
    file.write('Prediction E_g (eV)           DFT E_g (eV)\n')

    for case in range(len(real_value_test)):
        file.write(f'{predicted_value_test[case]}           {real_value_test[case]}\n')


# Plot the predictions
plt.figure()
plt.title('Predictions test')
plt.xlabel('DFT computed band gap (eV)')
plt.ylabel('Predicted band gap (eV)')
plt.xlim(0, 8)
plt.ylim(0, 8)
real_value_test = torch.tensor(real_value_test)
predicted_value_test = torch.tensor(predicted_value_test)
plt.plot(real_value_test.cpu().numpy()[:], predicted_value_test.cpu().numpy()[:], linestyle='', marker='o', alpha=0.6, color='springgreen', label='test')
plt.plot([0, 8], [0, 8], linestyle='--', color='forestgreen')
plt.tight_layout()
plt.savefig(outputs_dir + 'predictions_test_plot.pdf')


# Compute some metrics to evaluate the model
mse_test = mean_squared_error(real_value_test,predicted_value_test)

mae_test = mean_absolute_error(real_value_test,predicted_value_test)

r2_test = r2_score(real_value_test,predicted_value_test)

max_test = max_error(real_value_test,predicted_value_test)

with open(outputs_dir + 'metrics_model.txt', 'w') as file:
    file.write('MODEL METRICS\n')
    file.write('\n')

    file.write('TEST\n')
    file.write(f'MSE:                  {mse_test}\n')
    file.write(f'MAE:                  {mae_test}\n')
    file.write(f'r2:                   {r2_test}\n')
    file.write(f'Maximum error:        {max_test}\n')
###########################################################################