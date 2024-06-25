# Pol Benítez Colominas, June 2024
# Universitat Politècnica de Catalunya

# Predict band gaps with the retrained model

import json
import math

import numpy as np

# pymatgen modules
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter

from pymatgen.io.cif import CifParser
# pytorch and torch geometric modules
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool


# Functions to create the gaphs
with open('atoms_dict.json', 'r') as json_file:
    atoms_dict = json.load(json_file)


def get_nodes(struct):
    """
    This function recives a unit cell structure (pymatgen structure object) and returns the node list
    
    Inputs:
        struct: structure object
    Outputs:
        node_list: list of nodes with their features
    """
    # get the number of atoms in the unit cell
    atoms_number = struct.num_sites

    # create the node list
    node_list = [None]*atoms_number

    # save nodes in node list with the features of the given atom
    for atom in range(atoms_number):
        node_list[atom] = atoms_dict[(struct.sites[atom]).species_string]

    return node_list

def get_edges(struct, lim_dist):
    """
    This function recives a unit cell structure (pymatgen structure object) and returns the adjacency list, i.e., 
    a list of pairs of nodes that are closer than the desired distance, and the edges list, i.e, a list with the 
    features of each element of the adjecent list, here the euclidean distance. In order to do this a proper 
    supercell is created

    Inputs:
        struct: structure object
        lim_dist: maximum distance to consider edges
    Outputs:
        adjacency_list: list of pairs of nodes that verify to be closer than lim_dist
        edge_list: list of features for all the edges in adjacency list
    """
    # create adjacency and edge lists
    adjacency_list = []
    edge_list = []

    # get the lattice parameters and the smallest parameter
    lattice_parameters = struct.lattice.abc
    min_parameter = min(lattice_parameters)

    # find the minimum supercell (with central cell, n=3,5,...) to consider all the connections for the given limit distance
    n_supercell = 3
    param_supercell = min_parameter
    while param_supercell < lim_dist:
        n_supercell = n_supercell + 2
        param_supercell = min_parameter*(n_supercell - 2) 

    # number for the atoms in the centered cell after creating a supercell
    atoms_centered_cell = math.trunc((n_supercell**3)/2) + 1

    # get the number of atoms in the unit cell
    atoms_number = struct.num_sites

    # create the supercell
    scaling_matrix = [[n_supercell, 0, 0], [0, n_supercell, 0], [0, 0, n_supercell]]
    supercell = struct.make_supercell(scaling_matrix)

    # get the number of atoms in the supercell
    atoms_supercell_number = supercell.num_sites

    # check if there is a connection between two atoms (count just one of the directions, example: just (0,2), not (0,2) and (2,0))
    for atom in range(atoms_number):
        for atom_super in range(atoms_supercell_number - atom*(n_supercell**3)):
            a_cell = (supercell.sites[atom*(n_supercell**3) + atoms_centered_cell]).coords[0]
            b_cell = (supercell.sites[atom*(n_supercell**3) + atoms_centered_cell]).coords[1]
            c_cell = (supercell.sites[atom*(n_supercell**3) + atoms_centered_cell]).coords[2]

            a_super = (supercell.sites[atom_super + atom*(n_supercell**3)]).coords[0]
            b_super = (supercell.sites[atom_super + atom*(n_supercell**3)]).coords[1]
            c_super = (supercell.sites[atom_super + atom*(n_supercell**3)]).coords[2]

            euclidean_distance = ((a_cell - a_super)**2 + (b_cell - b_super)**2 + (c_cell - c_super)**2)**0.5

            if (euclidean_distance <= lim_dist) and (euclidean_distance > 1e-5): 
                edge_pair = [atom, math.trunc((atom_super + atom*(n_supercell**3))/(n_supercell**3))]

                edge_feature = [euclidean_distance]

                # chech if it is self-loop, if not save twice to be undirected
                if edge_pair[0] != edge_pair[1]:
                    adjacency_list.append(edge_pair)
                    edge_pair2 = [edge_pair[1], edge_pair[0]]
                    adjacency_list.append(edge_pair2)

                    edge_list.append(edge_feature)
                    edge_list.append(edge_feature)
                else:
                    adjacency_list.append(edge_pair)
                    edge_list.append(edge_feature)

    return adjacency_list, edge_list

# CGNN model
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

        # Linear layers
        self.lin1 = Linear(hidden_channels, 16)
        self.lin2 = Linear(16, 1)

    def forward(self, x, edge_index, edge_attr, batch):

        # Node embedding
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()

        # Mean pooling to reduce dimensionality
        x = global_mean_pool(x, batch)  

        # Apply neural network for regression prediction problem

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)

        return x

model = GCNN(features_channels=4, hidden_channels=256)
    
model.load_state_dict(torch.load('trained_model'))

model.eval()


# Function to predict the band gap
def predict_bg(path):
    """
    Predicts the band gap using the cgnn trained model

    Inputs:
        path: path to the POSCAR file
    Outputs:
        bg: predicted band gap
    """

    structure = Poscar.from_file(path).structure

    cif_writer = CifWriter(structure)
    cif_filename = path + '.cif'
    cif_writer.write_file(cif_filename)

    parser = CifParser(cif_filename)
    structure_object = parser.parse_structures(primitive=True)[0]

    nodes = get_nodes(structure_object)

    adjacency, edges = get_edges(structure_object, edge_radius)

    nodes_torch = torch.tensor(nodes, dtype=torch.float32)
    adjacency_torch = torch.tensor(adjacency, dtype=torch.long)
    edges_torch = torch.tensor(edges, dtype=torch.float32)

    node_features = (nodes_torch - min_node)/(max_node - min_node)

    edge_features = (edges_torch - min_edge)/(max_edge - min_edge)

    graph = Data(x=node_features, edge_index=adjacency_torch.t().contiguous(), edge_attr=edge_features)

    with torch.no_grad():
        prediction = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

    bg = prediction*(max_output - min_output) + min_output

    return bg


# Normalization values used during the training
max_node = torch.tensor([ 88.0000,   3.9800, 244.0000, 298.0000])
min_node = torch.tensor([ 0.0000,  0.7900,  1.0080, 42.0000])

max_edge = 5.5000
min_edge = 0.6932

max_output = 6.950200080871582
min_output = 0.05009999871253967


# Distance cutoff used to create the graphs
edge_radius = 5.5   


# Predict values for the desired POSCAR files
print(predict_bg('POSCAR')) 

predicted_bg = []
for x in range(100):
    path = 'XDATCAR-poscars/POSCAR-' + str(x + 1).zfill(3)

    predicted_bg.append(float(predict_bg(path)))

mean_value = np.mean(predicted_bg)
std_dev = np.std(predicted_bg)
error = std_dev/np.sqrt(len(predicted_bg) - 1)

print(f'The mean predicted band gap is: {mean_value}±{error} eV')