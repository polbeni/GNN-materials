# Pol Benítez Colominas, June 2025
# Universitat Politècnica de Catalunya

# Compute predictions for the other sets (uniform and phononic)


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

from models_cgcnn import *
###########################################################################



################################ PARAMETERS ###############################
hidden = 128                                                            # Number of hidden channels in the convolutional layers
dropout = 0.4                                                           # Fraction of elements to dropout

database_type = 'phononic'                                                   # The other type of physical sets

path_to_graphs = '/home/claudio/pol/graphs-models-bg/retraining/retrain-cgcnn/generate-graphs/HSEsol/normalized_graphs/'          # Path to the normalized graphs
path_to_csv = '/home/claudio/pol/graphs-models-bg/retraining/retrain-cgcnn/generate-graphs/HSEsol/graphs-bg.csv'                  # Path to csv file with graphs names and value

model_path = 'trained_uniform'                                            # Path or name of the final trained model

outputs_dir = 'outputs_file_uniform/'                                     # Path to dir where outputs are saved

seed_model_torch = 12345                                                # Seed for the model
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
        if database_type == 'uniform':
            if row[2] == 'uniform' or row[2] == 'static':
                file_list.append(row[0])
        elif database_type == 'phononic':
            if row[2] == 'phononic' or row[2] == 'static':
                file_list.append(row[0])
        elif database_type == 'both':
            file_list.append(row[0])

dataset_graphs = []

df_materials = pd.read_csv(path_to_csv)

# Exclude those graphs with output outside the range
with open('output_normalization.txt', 'r') as file:
    file.readline()
    line = file.readline()

    max_output = float(line.split()[0])
    min_output = float(line.split()[1])


for filename in file_list:
    data = torch.load(path_to_graphs + filename + '.pt')
    if (data.y < max_output) and (data.y > min_output):
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
model = model5(features_channels=dataset_graphs[0].num_node_features, hidden_channels=hidden, seed_model=seed_model_torch, dropout=dropout)
model = model.to(device)


# Load the trained model
model.load_state_dict(torch.load(model_path))


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
real_value_test = torch.tensor(real_value_test)
predicted_value_test = torch.tensor(predicted_value_test)
min_plot = min([min(real_value_test), min(predicted_value_test)])
max_plot = max([max(real_value_test), max(predicted_value_test)])
plt.xlim(min_plot - min_plot*0.2, max_plot + max_plot*0.2)
plt.ylim(min_plot - min_plot*0.2, max_plot + max_plot*0.2)
plt.plot(real_value_test.cpu().numpy()[:], predicted_value_test.cpu().numpy()[:], linestyle='', marker='o', alpha=0.6, color='springgreen', label='test')
plt.plot([min_plot - min_plot*0.2, max_plot + max_plot*0.2], [min_plot - min_plot*0.2, max_plot + max_plot*0.2], linestyle='--', color='forestgreen')
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
