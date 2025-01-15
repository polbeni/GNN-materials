# Pol Benítez Colominas, January 2024 - April 2024
# Universitat Politècnica de Catalunya

# Code to generate graphs from unit cell structure files (as cif or POSCAR files)

# system modules
import os
import shutil
import json
import math
import glob

# pymatgen modules
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

# pytorch and torch geometric modules
import torch
import torch_geometric.utils as utils
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

# define the maximum longitude to consider edge connections (angstroms)
edge_radius = 5.5   

# create a dictionary with materials id and bandgaps
materials_dict = {}
with open('materials.txt', 'r') as file:
    next(file)

    for line in file:
        materials_dict[line.split()[0] + '.cif'] = float(line.split()[1])

# create a list with all the structures that we want transform to a graph (in this case cif files with name mp-#.cif)
structures_path = 'CAP-structures/'
structures_list = glob.glob(f'{structures_path}hsesol_*')

# open a file to save the problematic structures
discarted_structures = open('discarted_structures.txt', 'w')

# transform each structure
if os.path.exists('graph_structures'):
    shutil.rmtree('graph_structures')
os.mkdir('graph_structures')

number_struc = 1
total_number_struc = len(structures_list)
for struc_path in structures_list:
    print(f'Structure number {number_struc} of a total of {total_number_struc}')

    # avoid corrupt cif files
    try:
        parser = CifParser(struc_path)
        structure_object = parser.parse_structures(primitive=True)[0]

        nodes = get_nodes(structure_object)

        adjacency, edges = get_edges(structure_object, edge_radius)

        nodes_torch = torch.tensor(nodes)
        adjacency_torch = torch.tensor(adjacency)
        edges_torch = torch.tensor(edges)

        discarted = False
        
        print('Graph generated!')
    except:
        print('Problem with the cif file')

        discarted = True

        discarted_structures.write(f'{os.path.basename(struc_path)}\n')

    # check if nodes or edge list is empty (if it is the case discart the structure)
    if (len(edges) == 0) or (len(nodes) == 0):
        print('But empty lists')

        discarted = True

        discarted_structures.write(f'{os.path.basename(struc_path)}\n')

    # save the structures in torch geometric files (if they are not corrupt)
    if discarted == False:
        bg = materials_dict[os.path.basename(struc_path)]

        data = Data(x=nodes_torch, edge_index=adjacency_torch.t().contiguous(), edge_attr=edges_torch, y=torch.tensor([float(bg)]))

        path_to_save = struc_path.split('/')[1].split('.')[0]

        torch.save(data, 'graph_structures/' + path_to_save + '.pt')
    
    number_struc = number_struc + 1

discarted_structures.close()
