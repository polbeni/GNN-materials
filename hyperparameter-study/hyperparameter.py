# Pol Benítez Colominas, May 2025
# Universitat Politècnica de Catalunya

# Hyperparameter testing for CGCNN regression models



################################# LIBRARIES ###############################
import os
import shutil
from datetime import datetime
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

import torch
from torch_geometric.loader import DataLoader

from models_cgcnn import *
###########################################################################



############################# FIXED PARAMETERS ############################
num_epochs = 600                                                        # Number of epochs in the training
batch_size = 128                                                        # Number of samples in the batch
train_set_size = 0.7                                                    # Fraction of the trainin set size
validation_set_size = 0.15                                              # Fraction of the validation set size
test_set_size = 0.15                                                    # Fraction of the test set size

path_to_graphs = '../materials-dataset-new/normalized_graphs/'          # Path to the normalized graphs
path_to_csv = '../materials-dataset-new/graphs-bg.csv'                  # Path to csv file with graphs names and value

model_name = 'trained_model'                                            # Path or name of the final trained model

outputs_dir = 'outputs_file/'                                           # Path to dir where outputs are saved

seed_splitting = 42                                                     # Seed for the splitting of the training and test sets
seed_model_torch = 12345                                                # Seed for the model
###########################################################################



############################## HYPERPARAMETERS ############################
models = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6']   # CGCNN model
learning_rates = ['1e-2', '1e-3', '1e-4', '1e-5', '1e-6']               # Value of the learning rate step
hidden_array = ['32', '64', '128', '256']                               # Number of hidden channels in the convolutional layers
dropout_array = ['00', '02', '04', '06']                                # Fraction of elements to dropout

# Define dictionaries with the label values
models_dict = {
    'model1': model1,
    'model2': model2,
    'model3': model3,
    'model4': model4,
    'model5': model5,
    'model6': model6
}

lr_dict = {
    '1e-2': 1e-2,
    '1e-3': 1e-3,
    '1e-4': 1e-4,
    '1e-5': 1e-5,
    '1e-6': 1e-6
}

hidden_dict = {
    '32': 32,
    '64': 64,
    '128': 128,
    '256': 256
}

dropout_dict = {
    '00': 0,
    '02': 0.2,
    '04': 0.4,
    '06': 0.6
}
###########################################################################



################################ FUNCTIONS ################################
def train(model, criterion, train_loader, optimizer):
    """
    Determines the loss for the train and optimize the model

    Inputs:
        model: model to use in the training
        criterion: loss function
        train_loader: batched train data
        optimizer: optimization algorithm for the training
    """

    model.train()

    total_loss = 0

    # Iterate in batches over the training dataset
    for data in train_loader:  
        # Ensure tensors are on the right device and dtype
        data.x = data.x.to(device).float()
        data.edge_index = data.edge_index.to(device).long()
        data.edge_attr = data.edge_attr.to(device).float()
        data.y = data.y.to(device).float()

        data = data.to(device)
        
        out = model(data.x, data.edge_index, data.edge_attr, data.batch).to(device).squeeze(-1)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss = total_loss + loss.item()
    
    average_loss = total_loss / len(train_loader)

    return average_loss


def test(model, criterion, test_loader):
    """
    Check the performance of the model over the test set

    Inputs:
        model: model to use in the training
        criterion: loss function
        test_loader: batched test data
    """

    model.eval()

    total_loss = 0

    with torch.no_grad():
        # Iterate in batches over the test dataset
        for data in test_loader:  
            # Ensure tensors are on the right device and dtype
            data.x = data.x.to(device).float()
            data.edge_index = data.edge_index.to(device).long()
            data.edge_attr = data.edge_attr.to(device).float()
            data.y = data.y.to(device).float()

            data.to(device)

            out = model(data.x, data.edge_index, data.edge_attr, data.batch).to(device).squeeze(-1)
            loss = criterion(out, data.y)

            total_loss = total_loss + loss.item()

    average_loss = total_loss / len(test_loader)

    return average_loss


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


# Save in array the graphs in the train-validation sets
file_list_train = []

with open('train_val_graphs.txt', 'r') as file:
    for row in file:
        file_list_train.append(row.split()[0])


# Load the normalized graphs and save them in an array
dataset_graphs = []

df_materials = pd.read_csv(path_to_csv)

for filename in file_list_train:
    data = torch.load(path_to_graphs + filename + '.pt')
    dataset_graphs.append(data)

print(f'A total of {len(dataset_graphs)} graphs loaded.')


# Normalize the outputs and save the normalization constants in a file
outputs_graphs = torch.cat([graph.y for graph in dataset_graphs], dim=0)

max_output = outputs_graphs.max(dim=0).values
min_output = outputs_graphs.min(dim=0).values
outputs_graphs = (outputs_graphs - min_output)/(max_output - min_output)

print(f'Normalization of output, max value: {max_output}, min value: {min_output}')
output_normalization = open('output_normalization.txt', 'w')
output_normalization.write('max_output  min_output\n')
output_normalization.write(f'{max_output}  {min_output}')
output_normalization.close()

num_struc = 0
for graph in dataset_graphs:
    graph.y = outputs_graphs[num_struc]

    num_struc = num_struc + 1


# Define the size of the sets
train_size = int((train_set_size / (train_set_size + validation_set_size)) * len(dataset_graphs))
validation_size  = len(dataset_graphs) - train_size

print(f'The train set contains {train_size} graphs')
print(f'The validation set contains {validation_size} graphs')


# Create the train and validation sets
train_dataset, validation_dataset = torch.utils.data.random_split(dataset_graphs, [train_size, validation_size], generator=torch.Generator().manual_seed(seed_splitting))


# Generate the train and validation loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


# Hyperparameter loop
for model_it in models:
    for lr_it in learning_rates:
        for hidden_it in hidden_array:
            for dropout_it in dropout_array:
                # Define the hyperparameters
                learning_rate = lr_dict[lr_it]
                hidden = hidden_dict[hidden_it]
                dropout = dropout_dict[dropout_it]

                # Create a folder to save the results and trained model
                path_model = model_it + '_' + lr_it + '_' + hidden_it + '_' + dropout_it
                if os.path.exists(path_model):
                    shutil.rmtree(path_model)
                os.mkdir(path_model)

                # Create the model
                model = models_dict[model_it](features_channels=dataset_graphs[0].num_node_features, hidden_channels=hidden, seed_model=seed_model_torch, dropout=dropout)

                model = model.to(device)
                print(model)    

                # Define the optimizer and criterion
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = torch.nn.MSELoss()

                # Loop over the epochs to train the model and compute losses
                train_losses = []
                validation_losses = []

                for epoch in range(num_epochs):
                    train_loss = train(model, criterion, train_loader, optimizer)
                    validation_loss = test(model, criterion, validation_loader)

                    # Append losses
                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)

                    print(f'Epoch {epoch + 1} of a total of {num_epochs}')
                    print(f'     Train loss:   {train_loss}')
                    print(f'     Validation loss:    {validation_loss}')

                # Save the trained model
                torch.save(model.state_dict(), path_model + '/' + model_name)


                #### Save the outputs ####
                # Create the folder to save the outputs
                if os.path.exists(path_model + '/' + outputs_dir):
                    shutil.rmtree(path_model + '/' + outputs_dir)
                os.mkdir(path_model + '/' + outputs_dir)

                # Create data with basic information of our model
                with open(path_model + '/' + outputs_dir + 'model_info.txt', 'w') as file:
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    file.write(f'MODEL SUMMARY ({current_time})\n')
                    file.write('\n')

                    file.write('PARAMETERS\n')
                    file.write(f'num_epochs:          {num_epochs}\n')
                    file.write(f'learning_rate:       {learning_rate}\n')
                    file.write(f'batch_size:          {batch_size}\n')
                    file.write(f'train_set_size:      {train_set_size}\n')
                    file.write(f'hidden:              {hidden}\n')
                    file.write(f'dropout:             {dropout}\n')
                    file.write(f'seed_splitting:      {seed_splitting}\n')
                    file.write(f'seed_model_torch:    {seed_model_torch}\n')
                    file.write(f'path_to_graphs:      {path_to_graphs}\n')
                    file.write(f'path_to_csv:         {path_to_csv}\n')
                    file.write('\n')

                    file.write('MODEL ARCHITECTURE\n')
                    file.write(str(model))
                    file.write('\n')
                    file.write('\n')

                    file.write('DATASET SIZE\n')
                    file.write(f'Total number of graphs:  {len(dataset_graphs)}\n')
                    file.write(f'Graphs on training set:  {train_size}\n')
                    file.write(f'Graphs on validation set:      {validation_size}\n')


                # Save the loss for train and validation
                with open(path_model + '/' + outputs_dir + 'loss.txt', 'w') as file:
                    file.write('Epoch      Loss_train      Loss_validation\n')

                    for epoch in range(num_epochs):
                        file.write(f'{epoch + 1}      {train_losses[epoch]}      {validation_losses[epoch]}\n')
                
                # Plot the train/validation loss with epochs
                plt.figure()
                plt.yscale('log')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.plot(np.linspace(1, num_epochs, num_epochs-1), train_losses[1:], label='train')
                plt.plot(np.linspace(1, num_epochs, num_epochs-1), validation_losses[1:], label='validation')
                plt.legend()
                plt.tight_layout()
                plt.savefig(path_model + '/' + outputs_dir + 'loss_plot.pdf')
                plt.close()

                # Compute predictions for all the graphs and save them in files
                real_value_train = []
                predicted_value_train = []
                real_value_validation = []
                predicted_value_validation = []

                model.eval()

                for num_graph in range(len(train_dataset)):
                    graph = train_dataset[num_graph]

                    real_value_train.append(graph.y*(max_output - min_output) + min_output)

                    predict_value = model_prediction(model, graph, [min_output, max_output])
                    predicted_value_train.append(predict_value)

                for num_graph in range(len(validation_dataset)):
                    graph = validation_dataset[num_graph]

                    real_value_validation.append(graph.y*(max_output - min_output) + min_output)

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
                    predicted_value_validation.append(prediction[0][0]*(max_output - min_output) + min_output)

                # Save the prediction vs DFT results
                with open(path_model + '/' + outputs_dir + 'prediction_vs_DFT_train.txt', 'w') as file:
                    file.write('Prediction E_g (eV)           DFT E_g (eV)\n')

                    for case in range(len(real_value_train)):
                        file.write(f'{predicted_value_train[case]}           {real_value_train[case]}\n')

                with open(path_model + '/' + outputs_dir + 'prediction_vs_DFT_validation.txt', 'w') as file:
                    file.write('Prediction E_g (eV)           DFT E_g (eV)\n')

                    for case in range(len(real_value_validation)):
                        file.write(f'{predicted_value_validation[case]}           {real_value_validation[case]}\n')


                # Plot the predictions
                plt.figure()
                plt.xlabel('DFT computed band gap (eV)')
                plt.ylabel('Predicted band gap (eV)')
                plt.xlim(0, 8)
                plt.ylim(0, 8)
                real_value_train = torch.tensor(real_value_train)
                predicted_value_train = torch.tensor(predicted_value_train)
                plt.plot(real_value_train.cpu().numpy()[:], predicted_value_train.cpu().numpy()[:], linestyle='', marker='o', alpha=0.6, color='lightsteelblue', label='train')
                real_value_validation = torch.tensor(real_value_validation)
                predicted_value_validation = torch.tensor(predicted_value_validation)
                plt.plot(real_value_validation.cpu().numpy()[:], predicted_value_validation.cpu().numpy()[:], linestyle='', marker='o', alpha=0.6, color='salmon', label='validation')
                plt.plot([0, 8], [0, 8], linestyle='--', color='royalblue')
                plt.legend()
                plt.tight_layout()
                plt.savefig(path_model + '/' + outputs_dir + 'predictions_plot.pdf')
                plt.close()

                plt.figure()
                plt.title('Predictions train')
                plt.xlabel('DFT computed band gap (eV)')
                plt.ylabel('Predicted band gap (eV)')
                plt.xlim(0, 8)
                plt.ylim(0, 8)
                real_value_train = torch.tensor(real_value_train)
                predicted_value_train = torch.tensor(predicted_value_train)
                plt.plot(real_value_train.cpu().numpy()[:], predicted_value_train.cpu().numpy()[:], linestyle='', marker='o', alpha=0.6, color='lightsteelblue', label='train')
                plt.plot([0, 8], [0, 8], linestyle='--', color='royalblue')
                plt.tight_layout()
                plt.savefig(path_model + '/' + outputs_dir + 'predictions_train_plot.pdf')
                plt.close()

                plt.figure()
                plt.title('Predictions validation')
                plt.xlabel('DFT computed band gap (eV)')
                plt.ylabel('Predicted band gap (eV)')
                plt.xlim(0, 8)
                plt.ylim(0, 8)
                real_value_validation = torch.tensor(real_value_validation)
                predicted_value_validation = torch.tensor(predicted_value_validation)
                plt.plot(real_value_validation.cpu().numpy()[:], predicted_value_validation.cpu().numpy()[:], linestyle='', marker='o', alpha=0.6, color='salmon', label='validation')
                plt.plot([0, 8], [0, 8], linestyle='--', color='royalblue')
                plt.tight_layout()
                plt.savefig(path_model + '/' + outputs_dir + 'predictions_validation_plot.pdf')
                plt.close()

                # Compute some metrics to evaluate the model
                mse_train = mean_squared_error(real_value_train, predicted_value_train)
                mse_validation = mean_squared_error(real_value_validation, predicted_value_validation)

                mae_train = mean_absolute_error(real_value_train, predicted_value_train)
                mae_validation = mean_absolute_error(real_value_validation, predicted_value_validation)

                r2_train = r2_score(real_value_train, predicted_value_train)
                r2_validation = r2_score(real_value_validation, predicted_value_validation)

                max_train = max_error(real_value_train, predicted_value_train)
                max_validation = max_error(real_value_validation, predicted_value_validation)

                min_loss_train = min(train_losses)
                epoch_min_loss_train = train_losses.index(min_loss_train) + 1
                min_loss_validation = min(validation_losses)
                epoch_min_loss_validation = validation_losses.index(min_loss_validation) + 1


                with open(path_model + '/' + outputs_dir + 'metrics_model.txt', 'w') as file:
                    file.write('MODEL METRICS\n')
                    file.write('\n')

                    file.write('TRAIN\n')
                    file.write(f'MSE:                  {mse_train}\n')
                    file.write(f'MAE:                  {mae_train}\n')
                    file.write(f'r2:                   {r2_train}\n')
                    file.write(f'Maximum error:        {max_train}\n')
                    file.write(f'Epoch minimum loss:   {epoch_min_loss_train}\n')
                    file.write('\n')

                    file.write('VALIDATION\n')
                    file.write(f'MSE:                  {mse_validation}\n')
                    file.write(f'MAE:                  {mae_validation}\n')
                    file.write(f'r2:                   {r2_validation}\n')
                    file.write(f'Maximum error:        {max_validation}\n')
                    file.write(f'Epoch minimum loss:   {epoch_min_loss_validation}\n')


                # Clear cache and delete model
                del model
                torch.cuda.empty_cache()
                gc.collect()
###########################################################################
