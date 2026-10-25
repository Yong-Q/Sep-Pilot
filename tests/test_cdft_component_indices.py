"""Native cDFT density indices and Henry indices may use different origins."""
import importlib.util
from pathlib import Path
import pytest


@pytest.mark.parametrize('indices', [(0,), (1,), (0, 1), (1, 2)])
def test_collection_maps_complete_native_components_without_losing_density(tmp_path, indices):
    path = Path(__file__).resolve().parents[1] / 'tools/cdft/cDFT_Initialization/collect_cdft_results.py'
    spec = importlib.util.spec_from_file_location('cdft_indices_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work = tmp_path / 'input_dir/MOF-5'
    work.mkdir(parents=True)
    rows = ['Molecule    Average_density(mol/L)     Max_density/bulk']
    rows += [f'{idx} {0.006 * (order + 1):.3f} ({100 / len(indices)}%) 147.869'
             for order, idx in enumerate(indices)]
    rows.append(' total density: 3.85843e-06')
    (work / 'output.dat').write_text('\n'.join(rows) + '\n')
    # hr.dat is zero-based even when a pure-gas output.dat starts at 1.
    (work / 'hr.dat').write_text('298 unit:mol/(L*atm)\n0 0.26302\n1 0.5\n')
    gases = ['CH4'] if len(indices) == 1 else ['CH4', 'N2']

    result = module.collect_results(str(tmp_path), gases)

    assert result[0]['CH4_density_mol_L'] == 0.006
    assert result[0]['CH4_percentage'] == 100 / len(indices)
    assert result[0]['CH4_henry_mol_L_atm'] == 0.26302
    if len(indices) == 2:
        assert result[0]['N2_density_mol_L'] == 0.012
        assert result[0]['N2_henry_mol_L_atm'] == 0.5
