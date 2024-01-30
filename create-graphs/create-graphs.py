# Pol Benítez Colominas, January 2024
# Universitat Politècnica de Catalunya

# Code to generate graphs from unit cell structure files (as cif or POSCAR files)

import json

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

with open('atoms_dict.json', 'r') as json_file:
    atoms_dict = json.load(json_file)


def get_nodes(struct):
    """
    This function recives a unit cell structure (pymatgen structure object) and returns the node set
    
    Inputs:
        struct: structure object
    Outputs:
        node_set: set of nodes
    """
    
    atoms_number = struct.num_sites

    node_set = [None]*atoms_number

    for atom in range(atoms_number):
        node_set[atom] = atoms_dict[(struct.sites[atom]).species_string]


    return node_set

parser = CifParser('structures/mp-32780.cif')
structure_object = parser.get_structures()[0]
nodes1 = get_nodes(structure_object)
print(nodes1)

parser = CifParser('structures/mp-976868.cif')
structure_object = parser.get_structures()[0]
nodes2 = get_nodes(structure_object)
print(nodes2)
