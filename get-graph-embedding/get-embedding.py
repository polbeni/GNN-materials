# Pol Benítez Colominas, January 2026
# The University of Tokyo and Universitat Politècnica de Catalunya

# Get the graph embedding of a given structure from a trained GNN model


################################# LIBRARIES ###############################
import os
import csv
import shutil
from datetime import datetime
import json
import math
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

from pymatgen.io.vasp import Poscar

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data

from models_cgcnn import *
###########################################################################



################################ PARAMETERS ###############################
hidden = 64                                                             # Number of hidden channels in the convolutional layers
dropout = 0.4                                                           # Fraction of elements to dropout
seed_model_torch = 12345                                                # Seed for the model

structure_path = 'POSCAR'                                               # Path to the structure files

model_path = '../trained-models/bg-phon'                                            # Path or name of the final trained model
###########################################################################



################################ FUNCTIONS ################################
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


def generate_graph(path):
    """
    Generates a graph for a given POSCAR structure

    Inputs:
        path: path to the structure file
    """

    poscar = Poscar.from_file(path)
                
    structure_object = poscar.structure 

    nodes = get_nodes(structure_object)

    adjacency, edges = get_edges(structure_object, edge_radius)

    nodes_torch = torch.tensor(nodes)
    adjacency_torch = torch.tensor(adjacency)
    edges_torch = torch.tensor(edges)

    nodes_torch = (nodes_torch - min_node)/(max_node - min_node)
    edges_torch = (edges_torch - min_edge)/(max_edge - min_edge)

    graph = Data(x=nodes_torch, edge_index=adjacency_torch.t().contiguous(), edge_attr=edges_torch)

    return graph


def get_graph_embedding(graph, model):
    """
    Uses the GNN model to get the embedding of a desired graph

    Inputs:
        graph: graph object for the structure
        model: GNN model object
    """

    graph.x = graph.x.to(device).float()
    graph.edge_index = graph.edge_index.to(device).long()
    graph.edge_attr = graph.edge_attr.to(device).float()
    graph = graph.to(device)

    #batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)

    model = model.to(device).float()

    with torch.no_grad():
        emb = model.get_embedding(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
    
    structure_embedding = emb.squeeze(0).cpu().numpy()

    return structure_embedding
###########################################################################



################################### MAIN ##################################
# Check if a GPU (CUDA) is available 
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU not available. Using CPU.")


# Import the normalization constants (inputs are taken by hand from normalized_parameters.txt)
max_node = torch.Tensor([ 88.0000,   3.9800, 244.0000, 298.0000])
min_node = torch.Tensor([ 0.0000,  0.7900,  1.0080, 42.0000])
max_edge = 5.5
min_edge = 0.7115


# Create the model, import the trained model and put it in the evaluation mode
model = model5(features_channels=4, hidden_channels=hidden, seed_model=seed_model_torch, dropout=dropout)
model = model.to(device)

#model.load_state_dict(torch.load(model_path)) # for cuda
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # for cpu

model.eval()


# Generate the graphs and predict the band gap for the generated POSCARs
with open('atoms_dict.json', 'r') as json_file:
    atoms_dict = json.load(json_file)

edge_radius = 5.5 # define the maximum longitude to consider edge connections (angstroms)

graph = generate_graph(structure_path)

embedding = get_graph_embedding(graph, model)
print(embedding)
###########################################################################