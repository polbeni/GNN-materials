# Pol Benítez Colominas, January 2024
# Universitat Politècnica de Catalunya

# This script download the strucutres (in POSCAR file format) that we are interested from Materials Project dataset

# Official Materials Project API documentation: https://docs.materialsproject.org/downloading-data/using-the-api/examples

from mp_api.client import MPRester
from pymatgen.io.vasp.inputs import Poscar

# API key provided to each user registered in materials project
api_key = "your api key"

# save in an array the ids of the materials of interest
materialids = open('materials.txt', 'r')
materialids.readline()

num_structures = 75748

array_ids = []
for x in range(num_structures):
    actual_line = materialids.readline()
    actual_id = actual_line.split()[0]
    array_ids.append(actual_id)

# read the structures from materials project
with MPRester(api_key) as mpr:

    docs = mpr.summary.search(material_ids=array_ids, fields=["structure"])

# save the structures in POSCAR format
for x in range(num_structures):
    structure = docs[x].structure
    poscar = Poscar(structure)
    file_name = 'structures/' + array_ids[x]
    poscar.write_file(file_name)
