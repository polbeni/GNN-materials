# Pol Benítez Colominas, January 2024
# Universitat Politècnica de Catalunya

# This script converts POSCAR files to cif files


from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter

# create an array with the id of all the materials that we want
materialids = open('materials.txt', 'r')
materialids.readline()

num_structures = 75748

array_ids = []
for x in range(num_structures):
    actual_line = materialids.readline()
    actual_id = actual_line.split()[0]
    array_ids.append(actual_id)

# open each poscar and save it as a cif file
for x in range(num_structures):
    poscar_filename = 'structures/' + array_ids[x]
    structure = Poscar.from_file(poscar_filename).structure

    cif_writer = CifWriter(structure)
    cif_filename = 'structures-cif/' + array_ids[x] + '.cif'
    cif_writer.write_file(cif_filename)
