"""Manufactured RASPA-style bytes for generic dossier control-flow tests."""
from pathlib import Path


def prepare_native_result(work, gas='Kr', filename='native.data'):
    work=Path(work);work.mkdir(parents=True)
    cif=work/'framework.cif'
    cif.write_text('data_fixture\n_cell_length_a 30\n_cell_length_b 30\n_cell_length_c 30\n'
        '_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n'
        'loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_charge\nC1 C 0 0 0 0\n')
    (work/'simulation.input').write_text('SimulationType MonteCarlo\nNumberOfCycles 1000\n'
        'NumberOfInitializationCycles 2000\nForcefield UFF\nUnitCells 1 1 1\nFrameworkName framework\n'
        'ExternalTemperature 298\nExternalPressure 100000\nComponent 0 MoleculeName '+gas+'\n'
        'MoleculeDefinition TraPPE\nWidomProbability 1.0\nCreateNumberOfMolecules 0\n')
    output=work/'Output/System_0'/filename;output.parent.mkdir(parents=True)
    return cif,output


def write_native_result(output, loading):
    Path(output).write_text('Number of unitcells [a]: 1\nNumber of unitcells [b]: 1\n'
        'Number of unitcells [c]: 1\nSimulation finished\n'
        'Average loading absolute [mol/kg framework] '+str(loading)+' +/- 0.03\n')
