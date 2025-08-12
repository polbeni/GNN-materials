# Pol Benítez Colominas, August 2025
# Universitat Politècnica de Catalunya

# Retrieve data from the vasprun.xml file

import numpy as np
from pymatgen.io.vasp import Vasprun

def retrieve_vasprun(path_to_vasprun, type_calculation):
    """
    Retrieve the following data from the vasprun.xml file: E_0, E_F, and stress tensor, and determines
    the hydrostatic stress from the stress tensor

    Inputs:
        path_to_vasprun: path to the vasprun.xml file
        type_calculation: level of theory used (PBEsol or HSEsol)
    """

    vr = Vasprun(path_to_vasprun)

    n_atoms = vr.final_structure.num_sites

    final_step = vr.ionic_steps[-1]
    e0 = final_step['electronic_steps'][-1]['e_0_energy']
    e0_per_atom = e0 / n_atoms

    fermi_level = vr.efermi

    if type_calculation == 'PBEsol':
        stress_tensor = np.array(vr.ionic_steps[-1]['stress'])  # stress components are in kB
        stress_tensor_gpa = stress_tensor[:][:] / 10 # units in GPa
        hydrostatic_stress = (stress_tensor_gpa[0][0] + stress_tensor_gpa[1][1] + stress_tensor_gpa[2][2]) / 3
    elif type_calculation == 'HSEsol':
        hydrostatic_stress = None

    return e0_per_atom, fermi_level, hydrostatic_stress


E_0, E_F, sigma_h = retrieve_vasprun('database-uniform/results-PBEsol/pure-compounds/struc-0001/vasprun.xml', 'PBEsol')
print(E_0, E_F, sigma_h)

E_0, E_F, sigma_h = retrieve_vasprun('database-uniform/results-HSEsol/pure-compounds/struc-0001/vasprun.xml', 'HSEsol')
print(E_0, E_F, sigma_h)