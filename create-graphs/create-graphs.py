# Pol Benítez Colominas, January 2024
# Universitat Politècnica de Catalunya

# Code to generate graphs from unit cell structure files (as cif or POSCAR files)

import os
import json
import math
import glob

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

import torch
from torch_geometric.data import Data

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
    features of each element of the adjecent list, here the distance x, y and z (cartesian coordinates) for the 
    connection. In order to do this a proper supercell is created

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
    max_parameter = min(lattice_parameters)

    # find the minimum supercell to consider all the connections for the given limit distance
    n_supercell = 2
    param_supercell = max_parameter
    while param_supercell < lim_dist:
        n_supercell = n_supercell + 1
        param_supercell = max_parameter*(n_supercell - 1) 

    # get the number of atoms in the unit cell
    atoms_number = struct.num_sites

    # create the supercell
    scaling_matrix = [[n_supercell, 0, 0], [0, n_supercell, 0], [0, 0, n_supercell]]
    supercell = struct.make_supercell(scaling_matrix)

    # get the number of atoms in the supercell
    atoms_supercell_number = supercell.num_sites

    # check if there is a connection between two atoms
    for atom in range(atoms_number):
        for atom_super in range(atoms_supercell_number):
            a_cell = (supercell.sites[atom*(n_supercell**3)]).coords[0]
            b_cell = (supercell.sites[atom*(n_supercell**3)]).coords[1]
            c_cell = (supercell.sites[atom*(n_supercell**3)]).coords[2]

            a_super = (supercell.sites[atom_super]).coords[0]
            b_super = (supercell.sites[atom_super]).coords[1]
            c_super = (supercell.sites[atom_super]).coords[2]

            euclidean_distance = ((a_cell - a_super)**2 + (b_cell - b_super)**2 + (c_cell - c_super)**2)**0.5

            if (euclidean_distance <= lim_dist) and (euclidean_distance > 1e-5):
                edge_pair = [atom, math.trunc(atom_super/(n_supercell**3))]
                adjacency_list.append(edge_pair)

                a_dist = abs(a_cell - a_super)
                b_dist = abs(b_cell - b_super)
                c_dist = abs(c_cell - c_super)
                edge_feature = [a_dist, b_dist, c_dist]
                edge_list.append(edge_feature)

    return adjacency_list, edge_list

# define the maximum longitude to consider edge connections (angstroms)
edge_radius = 5.5

# create a list with all the structures that we want transform to a graph (in this case cif files with name mp-#.cif)
structures_path = 'structures/'
structures_list = glob.glob(f'{structures_path}mp-*')

# open a file to save the problematic structures
discarted_structures = open('discarted_structures.txt', 'w')

# transform each structure
"""
number_struc = 1
total_number_struc = len(structures_list)
for struc_path in structures_list:
    print(f'Structure number {number_struc} of a total of {total_number_struc}')

    # avoid corrupt cif files
    try:
        parser = CifParser(struc_path)
        structure_object = parser.get_structures()[0]

        nodes = get_nodes(structure_object)

        adjacency, edges = get_edges(structure_object, edge_radius)
        
        print('Graph generated!')
    except:
        print('Problem with the cif file')

        discarted_structures.write(f'{os.path.basename(struc_path)}\n')

    # check if nodes or edge list is empty (if it is the case discart the structure)
    if (len(edges) == 0) or (len(nodes) == 0):
        print('But empty lists')

        discarted_structures.write(f'{os.path.basename(struc_path)}\n')
    
    number_struc = number_struc + 1
"""

discarted_structures.close()

parser = CifParser('structures/mp-1167.cif')
structure_object = parser.get_structures()[0]

nodes = get_nodes(structure_object)

adjacency, edges = get_edges(structure_object, edge_radius)

nodes_torch = torch.tensor(nodes)
adjacency_torch = torch.tensor(adjacency)
edges_torch = torch.tensor(edges)

data = Data(x=nodes_torch, edge_index=adjacency_torch, edge_attr=edges_torch)

print(nodes_torch)
print(adjacency_torch)
print(adjacency_torch)

torch.save(data, 'mp-1167.pt')

loaded_graph_data = torch.load('mp-1167.pt')


######## IMPORTANT ########
# change edges to save just one direction, (0,2) and not (0,2) and (2,0)