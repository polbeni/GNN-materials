# Pol Benítez Colominas, May 2025
# Universitat Politècnica de Catalunya

# Visualize a graph created from a POSCAR file



################################# LIBRARIES ###############################
import math
import json

from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter
from pymatgen.io.cif import CifParser

import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.offline import plot
###########################################################################



################################ PARAMETERS ###############################
edge_radius = 5.5                                                       # maximum longitude to consider edge connections (angstroms)
###########################################################################



################################ FUNCTIONS ################################
def get_edges_mod(struct, lim_dist):
    """
    This modify version returns the node positions in the real space and the line defined for each edge in the real space

    Inputs:
        struct: structure object
        lim_dist: maximum distance to consider edges
    Outputs:
        adjacency_list: list of pairs of nodes that verify to be closer than lim_dist
        edge_list: list of features for all the edges in adjacency list
    """

    # create a list for the nodes positions and edges vectors
    node_positions = []
    edge_vectors = []

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

    # create a logical matrix indicating if a atom in the supercell is already in the saved node positions
    node_in_cell = [False]*atoms_number
    node_in_super = [False]*atoms_supercell_number

    # check if there is a connection between two atoms (count just one of the directions, example: just (0,2), not (0,2) and (2,0))
    for atom in range(atoms_number):
        for atom_super in range(atoms_supercell_number - atom*(n_supercell**3)):
            a_cell = (supercell.sites[atom*(n_supercell**3) + atoms_centered_cell]).coords[0]
            b_cell = (supercell.sites[atom*(n_supercell**3) + atoms_centered_cell]).coords[1]
            c_cell = (supercell.sites[atom*(n_supercell**3) + atoms_centered_cell]).coords[2]
            atom_type_cell = (supercell.sites[atom*(n_supercell**3) + atoms_centered_cell]).species

            a_super = (supercell.sites[atom_super + atom*(n_supercell**3)]).coords[0]
            b_super = (supercell.sites[atom_super + atom*(n_supercell**3)]).coords[1]
            c_super = (supercell.sites[atom_super + atom*(n_supercell**3)]).coords[2]
            atom_type_super = (supercell.sites[atom_super + atom*(n_supercell**3)]).species

            euclidean_distance = ((a_cell - a_super)**2 + (b_cell - b_super)**2 + (c_cell - c_super)**2)**0.5

            if (euclidean_distance <= lim_dist) and (euclidean_distance > 1e-5): 
                edge_pair = [atom, math.trunc((atom_super + atom*(n_supercell**3))/(n_supercell**3))]

                # Save the atom if not already saved
                if node_in_cell[atom] == False:
                    node_positions.append([a_cell, b_cell, c_cell, atom_type_cell, 'unitcell'])
                    
                    node_in_cell[atom] = True

                if [a_super, b_super, c_super, atom_type_super] not in node_positions:
                    node_positions.append([a_super, b_super, c_super, atom_type_super, 'supercell'])

                # Save the lines defined by vectors
                # chech if it is self-loop, if not save twice to be undirected (to be consistent with the repeated edges)
                if edge_pair[0] != edge_pair[1]:
                    edge_vectors.append([[a_cell, b_cell, c_cell], [a_super, b_super, c_super]])
                    edge_vectors.append([[a_cell, b_cell, c_cell], [a_super, b_super, c_super]])
                else:
                    edge_vectors.append([[a_cell, b_cell, c_cell], [a_super, b_super, c_super]])

    return node_positions, edge_vectors


def get_graph_for_visual(path, edge_radius):
    """
    Generates a graph from a POSCAR file

    Inputs:
        path: path to the POSCAR file
        edge_radius: maximum distance to consider chemical bonding
        min_node, max_node: array with minimum and maximum node features for normalization
        min_edge, max_edge: array with minimum and maximum edge features for normalization
    """

    structure = Poscar.from_file(path).structure

    cif_writer = CifWriter(structure)
    cif_filename = path + '.cif'
    cif_writer.write_file(cif_filename)

    parser = CifParser(cif_filename)
    structure_object = parser.parse_structures(primitive=True)[0]

    node_positions, edge_vectors = get_edges_mod(structure_object, edge_radius)

    return node_positions, edge_vectors


def plot_structure(node_positions, edge_vectors):
    """
    Represent the structure

    Inputs:
        adjacency_list: list of pairs of nodes that verify to be closer than lim_dist
        edge_list: list of features for all the edges in adjacency list
    """

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define marker/color for each species
    species_properties = {
        'Ag': {'marker': 'o', 'color': 'silver', 'label': 'Ag'},
        'S': {'marker': 's', 'color': 'yellow', 'label': 'S'},
        'Br': {'marker': '^', 'color': 'brown', 'label': 'Br'}
    }

    # Track which species have been labeled to avoid duplicate legends
    plotted_labels = set()

    # Plot each atom individually
    for atom in node_positions:
        x, y, z, comp, type_cell = atom
        spec = comp.reduced_formula  # Convert Composition to string (e.g., "Ag1")
        
        props = species_properties[spec]

        if type_cell == 'unitcell':
            size = 200
        else:
            size = 50
        
        # Plot the atom
        label = props['label'] if props['label'] not in plotted_labels else None
        if size == 200:
            ax.scatter(x, y, z,
                    marker=props['marker'],
                    color=props['color'],
                    edgecolor='black',
                    s=size)
        else:
            ax.scatter(x, y, z,
                    marker=props['marker'],
                    color=props['color'],
                    edgecolor='black',
                    s=size,
                    label=label)
            if label:
                plotted_labels.add(props['label'])
        
        
    # Represent all the connections
    for line in edge_vectors:
        p1, p2 = line
        ax.plot([p1[0], p2[0]], 
                [p1[1], p2[1]], 
                [p1[2], p2[2]], 
                color='black', 
                linestyle='--', 
                linewidth=1,
                alpha=0.5)

    # Add labels and legend
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    #ax.set_title('3D Atomic Positions')
    ax.legend()

    plt.tight_layout()
    plt.show()


def save_structure_to_html(node_positions, edge_vectors, filename="structure.html"):
    """
    Represent the structure and save it in a html format to be opened

    Inputs:
        adjacency_list: list of pairs of nodes that verify to be closer than lim_dist
        edge_list: list of features for all the edges in adjacency list
        filename: name to the html file
    """
    # Replicate your plotting logic in Plotly
    species_properties = {
        'Ag': {'marker': 'circle', 'color': 'silver', 'label': 'Ag'},
        'S': {'marker': 'square', 'color': 'yellow', 'label': 'S'},
        'Br': {'marker': 'diamond', 'color': 'brown', 'label': 'Br'}
    }

    # Create atom traces
    atom_traces = []
    plotted_labels = set()
    
    for atom in node_positions:
        x, y, z, comp, type_cell = atom
        spec = comp.reduced_formula
        props = species_properties[spec]
        
        size = 15 if type_cell == 'unitcell' else 5
        show_legend = props['label'] not in plotted_labels
        
        atom_traces.append(go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode='markers',
            marker=dict(
                symbol=props['marker'],
                color=props['color'],
                size=size,
                line=dict(color='black', width=1)
            ),
            name=props['label'] if show_legend else None,
            showlegend=show_legend
        ))
        if show_legend:
            plotted_labels.add(props['label'])

    # Create bond traces (exactly matching your original style)
    bond_traces = []
    for p1, p2 in edge_vectors:
        bond_traces.append(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            opacity=0.5,
            showlegend=False
        ))

    # Combine and save
    fig = go.Figure(data=atom_traces + bond_traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    plot(fig, filename=filename, auto_open=False)
###########################################################################



################################### MAIN ##################################
# Generate the graphs
with open('atoms_dict.json', 'r') as json_file:
    atoms_dict = json.load(json_file)

node_positions, edge_vectors = get_graph_for_visual('POSCAR', edge_radius)

plot_structure(node_positions, edge_vectors)
print(len(edge_vectors))
save_structure_to_html(node_positions, edge_vectors)

###########################################################################