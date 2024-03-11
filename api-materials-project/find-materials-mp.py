# Pol Benítez Colominas, January 2024 - March 2024
# Universitat Politècnica de Catalunya

# This script looks for materials in the materials project dataset with the desired properties, here materials with
# band gap larger than 0.05 eV (non-metals) and that contain at least Ag, S, Se, Br or I, and creates a file with the 
# id of the found materials and their properties of interest (band gap), and also saves the structure in the desired extension
# The band gaps have been determined with GGA/GGA+U exchange-correlation functionals (https://docs.materialsproject.org/methodology/materials-methodology/electronic-structure)

# Official Materials Project API documentation: https://docs.materialsproject.org/downloading-data/using-the-api/examples


from mp_api.client import MPRester
from pymatgen.io.vasp.inputs import Poscar

# API key provided to each user registered in materials project
api_key = "your api key"

# ask for the materials of interest, in this case materials with band gap larger than 0.05 eV
with MPRester(api_key) as mpr:
    docs = mpr.summary.search(
        band_gap=(0.05, None), fields=["material_id", "band_gap", "structure", "elements"]
    )

# declare the type of file you want to save the structures (cif or POSCAR)
file_fmt = 'cif'
    
# create a file with the id of the material and their band gap    
materials_file = open('materials.txt', 'w')
materials_file.write('Material ID       Band gap (eV) \n')

# list with the elements (at least one of them) that we are interested
possible_elements = ['Ag', 'S', 'Se', 'Br', 'I']

# save the structures that contain elements in possible_elements list in the materials file
for x in range(len(docs)):
    material = docs[x]

    materialid = material.material_id
    bandgap = material.band_gap
    elements_list = material.elements

    ions_list = []
    for el in range(len(elements_list)):
        ions_list.append(str(elements_list[el]))

    already_there = False
    for ion in possible_elements:
        if (ion in ions_list) and (already_there == False):
            materials_file.write(f'{materialid}       {bandgap} \n')

            structure = docs[x].structure
            if file_fmt == 'POSCAR':
                poscar = Poscar(structure)
                file_name = 'structures/' + materialid + '.poscar'
                poscar.write_file(file_name)
            elif file_fmt == 'cif':
                file_name = 'structures/' + materialid + '.cif'
                structure.to(fmt="cif", filename=file_name)
            else:
                print('Non supported file format')

            already_there = True

materials_file.close() # total of 18993 materials


#### Old code, that just takes all the materials with band gap >0.1 eV, total of 75748 materials ####
"""
with MPRester(api_key) as mpr:
    docs = mpr.summary.search(
        band_gap=(0.1, None), fields=["material_id", "band_gap"]
    )
    
# save in a file the id of the material and their band gap    
materials_file = open('materials.txt', 'w')
materials_file.write('Material ID       Band gap (eV) \n')

for x in range(len(docs)):
    material = docs[x]

    materialid = material.material_id
    bandgap = material.band_gap

    materials_file.write(f'{materialid}       {bandgap} \n')
"""
