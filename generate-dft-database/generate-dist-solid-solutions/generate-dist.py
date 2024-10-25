# Pol Benítez Colominas, October 2024
# University of Cambridge and Universitat Politècnica de Catalunya

# Applies gaussian noise to distort a given structure and study the convergence with the functional

import os
import shutil
import random

import numpy as np

from pymatgen.io.vasp import Poscar
from pymatgen.core import Element, Lattice


#### Basic variables definition ####

total_number = 1000                      # total number of structures
interval_width = [0.25, 1.25]            # interval of min, max width for the gaussian distortion
interval_lattice = [9.586, 9.738]        # lattice parameter for pure Ag3SBr, Ag3SI
path_to_save = 'generated-structures'    # path to save the resulting structures


#### Functions ####

def generate_solid_solution(poscar, interval_lattice):
    """
    It generates a random solid solution from the original POSCAR file and sets the proportional lattice parameters

    Inputs:
        poscar -> pymatgen poscar structure
        interval_lattice -> array with the lattice parameter for Ag3SBr and Ag3SI cases
    """

    random_array = [random.choice(['Br', 'I']) for _ in range(8)]

    it_loop = 0
    for i in range(-8, 0):
        poscar[i] = Element(random_array[it_loop])

        it_loop = it_loop + 1

    num_I = random_array.count('I')
    diff_lattice = interval_lattice[1] - interval_lattice[0]
    new_lattice = interval_lattice[0] + (num_I / 8) * diff_lattice

    new_lattice_vals = Lattice.from_parameters(a=new_lattice, b=new_lattice, c=new_lattice, alpha=90, beta=90, gamma=90)

    poscar.lattice = new_lattice_vals

    species_order = ["Ag", "S", "Br", "I"]
    poscar.sort(key=lambda site: species_order.index(site.species_string))
    
    return poscar


def get_atoms_array(poscar_file):
    """
    Returns an array with the unique atoms in POSCAR

    Inputs:
        poscar_file -> path to the POSCAR file (and name of the file)
    """

    poscar = Poscar.from_file(poscar_file)

    element_symbols = poscar.site_symbols

    unique_elements = []
    for element in element_symbols:
        if element not in unique_elements:
            unique_elements.append(element)

    return unique_elements


def create_potcar(atoms, path_potcar):
    """
    Creates a POTCAR file (for now with GGA functionals) for the desired atoms

    Inputs:
        atoms -> array with atoms
        path_potcar -> path to save the POTCAR file (and name of the file)
    """

    potcar_generation_command = 'cat '

    for atom in atoms:
        potcar_generation_command = potcar_generation_command + 'save/POTCARs/' + atom + '-POTCAR '

    potcar_generation_command = potcar_generation_command + '> ' + path_potcar

    os.system(potcar_generation_command)

    return


#### Distorted structures generation ####

poscar = Poscar.from_file('save/POSCAR')
original_poscar = poscar.structure

for struc in range(total_number):
    cp_poscar = original_poscar.copy()

    solid_solution = generate_solid_solution(cp_poscar, interval_lattice)

    disp = random.uniform(interval_width[0], interval_width[1])

    for site in cp_poscar:
        displacement_vector = np.random.uniform(0, disp, 3)

        site.coords = site.coords + displacement_vector

    os.mkdir(path_to_save + '/' + 'struc-' + str(struc + 1).zfill(4))

    # generate distorted POSCAR
    path_struc_PBEsol = path_to_save + '/' + 'struc-' + str(struc + 1).zfill(4) + '/POSCAR'
    cp_poscar.to(filename=path_struc_PBEsol, fmt='Poscar')

    # copy KPOITNS file
    source = 'save/KPOINTS' 
    destination = path_to_save + '/' + 'struc-' + str(struc + 1).zfill(4) + '/KPOINTS'
    shutil.copy(source, destination)

    # copy INCAR file
    source = 'save/INCAR' 
    destination = path_to_save + '/' + 'struc-' + str(struc + 1).zfill(4) + '/INCAR'
    shutil.copy(source, destination)

    # copy bash file
    source = 'save/run.sh' 
    destination = path_to_save + '/' + 'struc-' + str(struc + 1).zfill(4) + '/run.sh'
    shutil.copy(source, destination)

    # generate POTCAR file
    atoms_list = get_atoms_array(path_to_save + '/' + 'struc-' + str(struc + 1).zfill(4) + '/POSCAR')
    create_potcar(atoms_list, path_to_save + '/' + 'struc-' + str(struc + 1).zfill(4) + '/POTCAR')


#### Generate bash files to send and check the jobs ####
send_jobs = open(path_to_save + '/send_jobs.sh', "w")
send_jobs.write('#!/bin/bash \n')
send_jobs.write(' \n')
send_jobs.write('for i in {0001..' + str(total_number).zfill(4) + '} \n')
send_jobs.write('do \n')
send_jobs.write('cd struc-$i \n')
send_jobs.write('sbatch run.sh \n')
send_jobs.write('cd ../ \n')
send_jobs.write('done \n')
os.system("chmod +x " + path_to_save + "/send_jobs.sh")

check_results = open(path_to_save + '/check_results.sh', "w")
check_results.write('#!/bin/bash \n')
check_results.write(' \n')
check_results.write('for i in {0001..' + str(total_number).zfill(4) + '} \n')
check_results.write('do \n')
check_results.write('echo structure $i \n')
check_results.write('tail -n 1 struc-$i/OSZICAR \n')
check_results.write('done \n')
os.system("chmod +x " + path_to_save + "/check_results.sh") 