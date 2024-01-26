# Pol Benítez Colominas, January 2024
# Universitat Politècnica de Catalunya

# This script looks for materials in the materials project dataset with the desired properties, here materials with
# band gap larger than 0.1 eV, and creates a file with the id of the found materials and their properties of interest

# Official Materials Project API documentation: https://docs.materialsproject.org/downloading-data/using-the-api/examples


from mp_api.client import MPRester

# API key provided to each user registered in materials project
api_key = "your api key"

# ask for the materials of interest, in this case materials with band gap larger than 0.1 eV
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
