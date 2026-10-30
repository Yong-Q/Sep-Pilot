"""Synthetic CSV fixtures verify analysis contracts, never cDFT physics."""
import csv
from pathlib import Path
import pytest

from agents.cdft_screening import analyze_screening


def fixtures(root):
    cifs = root / 'cifs';cifs.mkdir()
    for name in ('alpha', 'beta'):
        (cifs / (name + '.cif')).write_text('synthetic structure')
    primary = root / 'primary.csv';secondary = root / 'secondary.csv'
    primary.write_text('MOF,Kr_density_mol_L\nalpha,6\nbeta,8\n')
    secondary.write_text('MOF,Xe_density_mol_L\nbeta,4\nalpha,1\n')
    return dict(data_csv=str(primary), input_path=str(secondary), gases=['Kr', 'Xe'],
                cif_dir=str(cifs), top_n=1, output_csv=str(root / 'ranked.csv'), output_dir=str(root / 'selected'))


def test_aligns_material_identities_ranks_and_stages_exact_top_n(tmp_path):
    p = fixtures(tmp_path)
    result = analyze_screening(p, tmp_path)
    assert result['ok']
    assert result['rows'][0]['MOF'] == 'alpha' and result['rows'][0]['uptake_ratio'] == 6
    assert result['selected_materials'] == ['alpha']
    assert (tmp_path / 'selected/alpha.cif').read_text() == 'synthetic structure'
    assert not (tmp_path / 'selected/beta.cif').exists()
    assert 'not IAST' in result['ratio_definition']
    assert result['conditions_verified'] is False
    assert result['component_mode_verified'] is False
    with pytest.raises(ValueError, match='exists'):
        analyze_screening(p, tmp_path)


@pytest.mark.parametrize('bad_csv', [
    'MOF,Xe_density_mol_L\nalpha,0\nbeta,4\n',
    'MOF,Xe_density_mol_L\nalpha,nan\nbeta,4\n',
    'MOF,Xe_density_mol_L\nalpha,1\nalpha,2\n',
    'MOF,Xe_density_mol_L\nalpha,1\n',
    'MOF,loading\nalpha,1\nbeta,4\n',
])
def test_bad_or_incomplete_evidence_never_publishes_rankings(tmp_path, bad_csv):
    p = fixtures(tmp_path);Path(p['input_path']).write_text(bad_csv)
    with pytest.raises(ValueError):
        analyze_screening(p, tmp_path)
    assert not Path(p['output_csv']).exists()
    assert not Path(p['output_dir']).exists()
