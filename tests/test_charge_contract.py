import json

from agents.charge_contract import validate_framework_charges


CIF = """data_test
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_charge
C1 C 0 0 0 1 0.0
"""


def test_batch_charge_validation_writes_one_report_without_modifying_cifs(tmp_path):
    source = tmp_path / 'charged_cifs'
    source.mkdir()
    files = [source / f'MOF_{index}_pacmof.cif' for index in range(2)]
    for path in files:
        path.write_text(CIF)
    before = [path.read_bytes() for path in files]
    output = tmp_path / 'charge_validation_report.json'

    result = validate_framework_charges(source, 0.0, 'PACMOF', 1e-4, output)

    assert result['ok'] and result['n_cifs'] == result['n_valid'] == 2
    assert result['output_files'] == [str(output.resolve())]
    assert json.loads(output.read_text())['status'] == 'validated'
    assert [path.read_bytes() for path in files] == before


def test_batch_charge_validation_requires_explicit_report_path(tmp_path):
    source = tmp_path / 'charged_cifs'
    source.mkdir()
    (source / 'MOF_pacmof.cif').write_text(CIF)
    try:
        validate_framework_charges(source, 0.0, 'PACMOF')
    except ValueError as error:
        assert 'output_path' in str(error)
    else:
        raise AssertionError('directory validation silently omitted durable evidence')
