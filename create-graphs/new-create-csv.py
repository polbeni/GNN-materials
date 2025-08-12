# Pol Benítez Colominas, February 2024 - August 2025
# Universitat Politècnica de Catalunya

# Generates a file with the id of the material and its outputs (band gap, energy per atom, fermi level, hydrostatic stress, and type of dataset)

import csv

discarted = []

with open('discarted_structures.txt', 'r') as file:
    for line in file:
        name = line.strip()
        name = name.replace('.cif', '')

        discarted.append(name)

structures = []
bandgaps = []
energies = []
fermies = []
hydrostatics = []
type_db = []

with open('materials.txt', 'r') as file:
    next(file)

    for line in file:
        struct = line.split()[0]
        bg = line.split()[1]
        e0 = line.split()[2]
        ef = line.split()[3]
        hy = line.split()[4]
        db = line.split()[5]

        if struct not in discarted:
            structures.append(struct)
            bandgaps.append(bg)
            energies.append(e0)
            fermies.append(ef)
            hydrostatics.append(hy)
            type_db.append(db)

with open('graphs-bg.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['material-id', 'bandgap', 'energy', 'fermi', 'hydrostatic', 'type_db'])

    for item1, item2, item3, item4, item5, item6 in zip(structures, bandgaps, energies, fermies, hydrostatics, type_db):
        writer.writerow([item1, item2, item3, item4, item5, item6])