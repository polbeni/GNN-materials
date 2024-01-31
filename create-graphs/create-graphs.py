# Pol Benítez Colominas, January 2024
# Universitat Politècnica de Catalunya

# Code to generate graphs from unit cell structure files (as cif or POSCAR files)

import json
import math

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

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
    
    atoms_number = struct.num_sites

    node_list = [None]*atoms_number

    for atom in range(atoms_number):
        node_list[atom] = atoms_dict[(struct.sites[atom]).species_string]


    return node_list

parser = CifParser('structures/mp-32780.cif')
structure_object = parser.get_structures()[0]
nodes1 = get_nodes(structure_object)
print(nodes1)

#parser = CifParser('structures/mp-976868.cif')
#structure_object = parser.get_structures()[0]
#nodes2 = get_nodes(structure_object)
#print(nodes2)


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
    adjacency_list = []
    edge_list = []

    lattice_parameters = struct.lattice.abc
    max_parameter = min(lattice_parameters)

    n_supercell = 2
    param_supercell = max_parameter
    while param_supercell < lim_dist:
        n_supercell = n_supercell + 1
        param_supercell = max_parameter*(n_supercell - 1) 

    atoms_number = struct.num_sites

    scaling_matrix = [[n_supercell, 0, 0], [0, n_supercell, 0], [0, 0, n_supercell]]
    supercell = struct.make_supercell(scaling_matrix)

    atoms_supercell_number = supercell.num_sites

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


adjacency1, edge1 = get_edges(structure_object, 5.5)
print(adjacency1)
print(edge1)
