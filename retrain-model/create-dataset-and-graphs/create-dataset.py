# Pol Benítez Colominas, June 2024
# Universitat Politècnica de Catalunya

# Code to generate CAP dataset from DFT computations

from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter

def electronic_bandGap(file_name):
    """
    This functions uses DOSCAR file generated in VASP simulations and returns the Fermi energy
    the band gap, and the energies of the band gap (respect the exchange-correlation functional
    used).

    file_name: path of the DOSCAR file
    """
    
    file = open(file_name, "r")

    for x in range(6):
        actual_string = file.readline()
        if x == 5:
            fermiEnergy = float(actual_string.split()[3])

    file.close()

    file = open(file_name, "r")

    for x in range(6):
        file.readline()

    for x in file:
        actual_string = x

        if (float(actual_string.split()[0]) <= fermiEnergy+0.1) and (float(actual_string.split()[0]) >= fermiEnergy-0.1):
            density_bandGap = float(actual_string.split()[2])

            break

    file.close()

    file = open(file_name, "r")

    for x in range(6):
        file.readline()

    for x in file:
        actual_string = x

        if float(actual_string.split()[2]) == density_bandGap:
            minEnergy = float(actual_string.split()[0])

            break   

    for x in file:
        actual_string = x

        if float(actual_string.split()[2]) != density_bandGap:
            maxEnergy = float(actual_string.split()[0])

            break 
    bandGap = maxEnergy - minEnergy

    file.close()
    
    return fermiEnergy, minEnergy, maxEnergy, bandGap

materials = open('materials.txt', 'w')
materials.write('material id       band gap (eV)\n')

for num in range(246):
    if num+1 not in [120, 121, 188, 190]:
        path_directory = 'dft-results/CAP-' + str(num+1).zfill(4)

        _, _, _, bandgap = electronic_bandGap(path_directory + '/DOSCAR')

        structure = Poscar.from_file(path_directory + '/POSCAR').structure

        cif_writer = CifWriter(structure)
        cif_filename = 'CAP-structures/cap-' + str(num+1).zfill(4) + '.cif'
        cif_writer.write_file(cif_filename)


        materials.write(f'cap-{str(num+1).zfill(4)}.cif       {bandgap}\n')

        print(f'Phase generated {num+1} of a total of 246')
    else:
        print('Jumped')

materials.close()