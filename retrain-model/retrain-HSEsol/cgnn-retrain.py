# Pol Benítez Colominas, March 2024 - May 2024
# Universitat Politècnica de Catalunya

# Crystal Graph Neural Network (CGNN) model for band gap prediction

import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool


# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU not available. Using CPU.")


# Define the machine learning model parameters
num_epochs = 200
batch_size = 16
hidden = 256
learning_rate = 1e-3
train_set_size = 0.75
dropout = 0.6 


# Load the normalized graphs and save them in an array
file_list = []

with open('/home/pol/work/crystal-graph-neural-networks/new-cgnn/graphs-retrain-HSEsol/graphs-bg.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        file_list.append(row[0])

dataset_graphs = []

path_data = '/home/pol/work/crystal-graph-neural-networks/new-cgnn/graphs-retrain-HSEsol/normalized_graphs/'

df_materials = pd.read_csv('/home/pol/work/crystal-graph-neural-networks/new-cgnn/graphs-retrain-HSEsol/graphs-bg.csv')

for filename in file_list:
    data = torch.load(path_data + filename + '.pt')
    dataset_graphs.append(data)

print(f'A total of {len(dataset_graphs)} graphs loaded.')


# Normalize the outputs
outputs_graphs = torch.cat([graph.y for graph in dataset_graphs], dim=0)

max_output = 6.950200080871582
min_output = 0.05009999871253967
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


# Define the size of the train and test set
train_size = int(train_set_size * len(dataset_graphs))
test_size  = len(dataset_graphs) - train_size

print(f'The train set contains {train_size} graphs')
print(f'The test set contains {test_size} graphs')


# Create the train and test sets
train_dataset, test_dataset = torch.utils.data.random_split(dataset_graphs, [train_size, test_size], generator=torch.Generator().manual_seed(42))


# Generate the train and test loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


# Define the CGNN model
class GCNN(torch.nn.Module):
    """
    Graph Convolution Neural Network model
    """

    def __init__(self, features_channels, hidden_channels):
        super(GCNN, self).__init__()
        torch.manual_seed(12346)

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

        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.sigmoid()

        return x
    

model = GCNN(features_channels=dataset_graphs[0].num_node_features, hidden_channels=hidden)
    
model.load_state_dict(torch.load('pretrained_model'))
print(model)


# Define the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()


# Define functions to train the model with the train set and evaluate its performance over the train and test sets
def train(model, criterion, train_loader, optimizer):
    """
    Train the model with the train set
    """

    model.train()
    total_loss = 0
    all_predictions = []
    all_ground_truths = []

    for data in train_loader:  # Iterate in batches over the training dataset
        # Ensure tensors are on the right device and dtype
        data.x = data.x.to(device).float()
        data.edge_index = data.edge_index.to(device).long()
        data.edge_attr = data.edge_attr.to(device).float()
        data.y = data.y.to(device).float()
        
        
        out = model(data.x, data.edge_index, data.edge_attr, data.batch).to(device).squeeze(-1)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        all_predictions.append(out.detach().cpu())
        all_ground_truths.append(data.y.detach().cpu())
    
    average_loss = total_loss / len(train_loader)

    return average_loss, all_predictions, all_ground_truths


def test(model, criterion, test_loader):
    """
    Check the performance of the model over the test set
    """

    model.eval()
    total_loss = 0
    all_predictions = []
    all_ground_truths = []
    with torch.no_grad():
        for data in test_loader:  # Iterate in batches over the test dataset
            # Ensure tensors are on the right device and dtype
            data.x = data.x.to(device).float()
            data.edge_index = data.edge_index.to(device).long()
            data.edge_attr = data.edge_attr.to(device).float()
            data.y = data.y.to(device).float()

            out = model(data.x, data.edge_index, data.edge_attr, data.batch).to(device).squeeze(-1)
            loss = criterion(out, data.y)

            total_loss += loss.item()
            all_predictions.append(out.cpu())
            all_ground_truths.append(data.y.cpu())

    average_loss = total_loss / len(test_loader)

    return average_loss, all_predictions, all_ground_truths


# Loop over the epochs to train the model
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    train_loss, train_predictions, train_ground_truths = train(model, criterion, train_loader, optimizer)
    test_loss, test_predictions, test_ground_truths = test(model, criterion, test_loader)

    # Append losses
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {epoch + 1} of a total of {num_epochs}')
    print(f'     Train loss:   {train_loss}')
    print(f'     Test loss:    {test_loss}')


# Plot the train/test loss with epochs
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(np.linspace(1, num_epochs, num_epochs), train_losses[:], label='train')
plt.plot(np.linspace(1, num_epochs, num_epochs), test_losses[:], label='test')
plt.legend()
plt.tight_layout()
plt.savefig('loss_plot.pdf')


# Show the predicted value vs DFT value
real_value_train = []
predicted_value_train = []
real_value_test = []
predicted_value_test = []

model.eval()

for num_graph in range(len(train_dataset)):
    graph = train_dataset[num_graph]

    real_value_train.append(graph.y*(max_output - min_output) + min_output)

    graph.x = graph.x.to(device).float()
    graph.edge_index = graph.edge_index.to(device).long()
    graph.edge_attr = graph.edge_attr.to(device).float()
    graph.y = graph.y.to(device).float()
    graph = graph.to(device)

    model = model.to(device).float()

    # Make prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch).to(device)

    # Convert prediction to CPU if necessary
    prediction = prediction.cpu()
    predicted_value_train.append(prediction[0][0]*(max_output - min_output) + min_output)


for num_graph in range(len(test_dataset)):
    graph = test_dataset[num_graph]

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

    # Convert prediction to CPU if necessary
    prediction = prediction.cpu()
    predicted_value_test.append(prediction[0][0]*(max_output - min_output) + min_output)

plt.figure()
plt.xlabel('DFT computed band gap (eV)')
plt.ylabel('Predicted band gap (eV)')
#max_value = max(np.max(real_value_train), np.max(real_value_test), np.max(predicted_value_train), np.max(predicted_value_test))
plt.xlim(0, 2.2)
plt.ylim(0, 2.2)
plt.plot(real_value_train[:], predicted_value_train[:], linestyle='', marker='o', alpha=0.6, color='lightsteelblue', label='train')
plt.plot(real_value_test[:], predicted_value_test[:], linestyle='', marker='o', alpha=0.6, color='salmon', label='test')
plt.plot([0, 3], [0, 3], linestyle='--', color='royalblue')
plt.legend()
plt.tight_layout()
plt.savefig('predictions_plot.pdf')


# Save some metrics of the model
mse_train = mean_squared_error(real_value_train,predicted_value_train)
mse_test = mean_squared_error(real_value_test,predicted_value_test)

mae_train = mean_absolute_error(real_value_train,predicted_value_train)
mae_test = mean_absolute_error(real_value_test,predicted_value_test)

metrics_file = open('metrics.txt', 'w')
metrics_file.write('Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics for train and test set:\n')
metrics_file.write(f'MSE train:   {mse_train}\n')
metrics_file.write(f'MSE test:    {mse_test}\n')
metrics_file.write(f'MAE train:   {mae_train}\n')
metrics_file.write(f'MAE test:    {mae_test}\n')
metrics_file.write('\n')
metrics_file.write('\n')
metrics_file.write('Train and test loss after each epoch:\n')
for epoch in range(num_epochs):
    metrics_file.write(f'Epoch {epoch + 1} of a total of {num_epochs}\n')
    metrics_file.write(f'     Train loss:   {train_losses[epoch]}\n')
    metrics_file.write(f'     Test loss:    {test_losses[epoch]}\n')
metrics_file.close()


# Save the model
model_path = 'trained_model'
torch.save(model.state_dict(), model_path)


# Open a model
#model = GCNN(features_channels=dataset_graphs[0].num_node_features, hidden_channels=batch_size)
#model.load_state_dict(torch.load(model_path))
#model.eval()
