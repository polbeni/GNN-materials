# Pol Benítez Colominas, February 2024 - March 2024
# Universitat Politècnica de Catalunya

# Generates a file with the id of the material and its band gap value

import csv

discarted = []

with open('discarted_structures.txt', 'r') as file:
    for line in file:
        name = line.strip()
        name = name.replace('.cif', '')

        discarted.append(name)

structures = []
bandgaps = []

with open('materials.txt', 'r') as file:
    next(file)

    for line in file:
        struct = line.split()[0]
        bg = line.split()[1]

        if struct not in discarted:
            structures.append(struct)
            bandgaps.append(bg)

with open('graphs-bg.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['material-id', 'bandgap'])

    for item1, item2 in zip(structures, bandgaps):
        writer.writerow([item1, item2])