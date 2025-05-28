# Pol Benítez Colominas, May 2025
# Universitat Politècnica de Catalunya

# CGCNN models



################################# LIBRARIES ###############################
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool
###########################################################################



########################## MODELS ARCHITECTURE ############################
# Model 1: 3CL, max_pooling, 2LN, sigmoid 
class model1(torch.nn.Module):
    """
    Graph Convolution Neural Network model
    """

    def __init__(self, features_channels, hidden_channels, seed_model, dropout):
        super(model1, self).__init__()
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
    

# Model 2: 3CL, mean_pooling, 2LN, sigmoid 
class model2(torch.nn.Module):
    """
    Graph Convolution Neural Network model
    """

    def __init__(self, features_channels, hidden_channels, seed_model, dropout):
        super(model2, self).__init__()
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
        x = global_mean_pool(x, batch)  

        # Apply neural network for regression prediction problem

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.sigmoid()

        return x
    

# Model 3: : 3CL, max_pooling, 2LN, relu
class model3(torch.nn.Module):
    """
    Graph Convolution Neural Network model
    """

    def __init__(self, features_channels, hidden_channels, seed_model, dropout):
        super(model3, self).__init__()
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
        x = x.relu()

        return x
    

# Model 4: 3CL, mean_pooling, 2LN, relu
class model4(torch.nn.Module):
    """
    Graph Convolution Neural Network model
    """

    def __init__(self, features_channels, hidden_channels, seed_model, dropout):
        super(model4, self).__init__()
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
        x = global_mean_pool(x, batch)  

        # Apply neural network for regression prediction problem

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()

        return x
    

# Model 5: 2CL, max_pooling, 2LN, sigmoid 
class model5(torch.nn.Module):
    """
    Graph Convolution Neural Network model
    """

    def __init__(self, features_channels, hidden_channels, seed_model, dropout):
        super(model5, self).__init__()
        torch.manual_seed(seed_model)

        self.dropout = dropout

        # Convolution layers
        self.conv1 = GraphConv(features_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)

        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

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

        # Mean pooling to reduce dimensionality
        x = global_max_pool(x, batch)  

        # Apply neural network for regression prediction problem

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.sigmoid()

        return x
    

# Model 6: 4CL, max_pooling, 2LN, sigmoid 
class model6(torch.nn.Module):
    """
    Graph Convolution Neural Network model
    """

    def __init__(self, features_channels, hidden_channels, seed_model, dropout):
        super(model6, self).__init__()
        torch.manual_seed(seed_model)

        self.dropout = dropout

        # Convolution layers
        self.conv1 = GraphConv(features_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)

        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)

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
        x = self.conv4(x, edge_index, edge_attr)
        x = self.bn4(x)
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