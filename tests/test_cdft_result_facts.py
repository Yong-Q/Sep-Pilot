import hashlib
import json
from pathlib import Path

import pytest


def fixture(tmp_path, *, cap=60, length=84, angle=90):
    root=tmp_path/'owned';work=root/'batch';(work/'input').mkdir(parents=True)
    text=f'''water 100 0
Nmaxx Nmaxy Nmaxz
{cap} {cap} {cap}
Lxa Lyb Lzc dx angle_a angle_b angle_c
{length} 32 32 .5 90 90 {angle}
Nutation Precession Rotation
10 10 10
Temperature(K) Kapa Delta_Max Torr
298 .95 2 .1
Kind_of_gas
1
Name Epsilon(K) Sigma(A) Bulk_Density(molec/A^3)
1 71.4 3.798 .0000243
Special_case
0
Id1 Id2 Epsilon(K) Sigma(A)
Read_Vext EOS Cutoff deos N3_cutoff Diele_con Cutoff_inside cut_coul
0 1 12.9 1 .99 1 .5 15
Mass(g/mol) Ds_Knudsen(m^2/s) MFV_parameter
28 2E-8 .712
Kind_of_solvent_atoms
1
ID diameter(A) Epsilon(K) charge
0 3.798 71.4 0
Number_of_solvent_atoms1
1
ID x y z
0 0 0 0
Number_of_solute_atoms
1
ID x y z diameter(A) Epsilon(K) charge
0 0 0 0 3 50 0
'''
    (work/'input/a.dat').write_text(text)
    run=work/'input_dir/a';run.mkdir(parents=True)
    (run/'input.dat').write_text(text)
    (run/'output.dat').write_text('Pressure of bulk: .98 atm\nMolecule Average_density(mol/L) Max_density/bulk\n1 1.2 (100%) 22\n total density: .0007226\n')
    (run/'hr.dat').write_text('298 unit:mol/(L*atm)\n0 1.3\n')
    return root,work,run


def inspect(root,work,**kwargs):
    from agents.cdft_result_facts import inspect_cdft_quality
    return inspect_cdft_quality(work,owned_root=root,gases=['N2'],expected_materials=['a'],**kwargs)


def test_legacy_finite_output_and_completed_scheduler_never_prove_convergence(tmp_path):
    root,work,_=fixture(tmp_path)
    facts=inspect(root,work,job_receipt={'state':'COMPLETED','job_id':'123'})
    assert facts['scientific_convergence'] is None
    assert facts['materials']['a']['native_convergence'] is None
    assert facts['materials']['a']['exit_code'] is None
    assert facts['complete_material_outputs'] is True
    assert facts['hamiltonian_comparison_verified'] is False


@pytest.mark.parametrize('status,verdict',[(0,True),(1,None),(2,False),(7,False),(11,False)])
def test_native_termination_is_separate_from_technical_exit(tmp_path,status,verdict):
    root,work,run=fixture(tmp_path)
    (run/'solver.log').write_text(f'Termination status: {status}\nmaximum norm for gradient: 1.0e-10\niterations: 20\n')
    facts=inspect(root,work)
    assert facts['materials']['a']['native_convergence'] is verdict
    assert facts['technical_completion'] is None


def test_grid_and_triclinic_warnings_are_observations_not_silent_parameter_changes(tmp_path):
    root,work,_=fixture(tmp_path,length=32,angle=60)
    before=(work/'input/a.dat').read_bytes();facts=inspect(root,work)
    grid=facts['materials']['a']['input_facts']
    assert grid['grid_capped'] and min(grid['perpendicular_heights_A'])<30
    assert 'cutoff_cell_height' in grid['warnings']
    assert (work/'input/a.dat').read_bytes()==before


def test_missing_materials_are_retained_as_failures(tmp_path):
    from agents.cdft_result_facts import inspect_cdft_quality
    root,work,_=fixture(tmp_path)
    facts=inspect_cdft_quality(work,owned_root=root,gases=['N2'],expected_materials=['a','missing'])
    assert not facts['complete_material_outputs']
    assert facts['materials']['missing']['errors']
    assert facts['scientific_convergence'] is None


@pytest.mark.parametrize('target',['input','output','log'])
def test_symlink_or_foreign_files_are_not_followed(tmp_path,target):
    root,work,run=fixture(tmp_path);foreign=tmp_path/'foreign';foreign.write_text('not permitted')
    path={'input':work/'input/a.dat','output':run/'output.dat','log':run/'solver.log'}[target]
    if path.exists():path.unlink()
    path.symlink_to(foreign)
    facts=inspect(root,work)
    assert facts['materials']['a']['errors'] and facts['scientific_convergence'] is None


def test_conflicting_log_status_and_nonfinite_norm_are_not_converged(tmp_path):
    root,work,run=fixture(tmp_path)
    (run/'solver.log').write_text('Termination status: 0\nTermination status: 7\nmaximum norm for gradient: nan\n')
    facts=inspect(root,work)
    assert facts['materials']['a']['native_convergence'] is None


def test_forged_exit_receipt_cannot_replace_native_log_or_matching_input(tmp_path):
    root,work,run=fixture(tmp_path)
    (run/'execution.json').write_text(json.dumps({'exit_code':0,'converged':True,'input_sha256':'fake','stdout_sha256':'fake'}))
    facts=inspect(root,work)
    assert facts['scientific_convergence'] is None
    assert facts['materials']['a']['exit_code'] is None
    assert facts['materials']['a']['errors']


def test_nonfinite_density_cannot_pass_material_completeness(tmp_path):
    root,work,run=fixture(tmp_path)
    (run/'output.dat').write_text('Molecule Average_density(mol/L)\n1 nan (100%) 2\n')
    assert not inspect(root,work)['complete_material_outputs']


@pytest.mark.parametrize('bad',['missing','unit','temperature','nonfinite','duplicate'])
def test_henry_evidence_requires_complete_indices_units_and_matching_temperature(tmp_path,bad):
    root,work,run=fixture(tmp_path)
    path=run/'hr.dat'
    if bad=='missing':path.unlink()
    elif bad=='unit':path.write_text('298 unit:mol/kg/bar\n0 1.3\n')
    elif bad=='temperature':path.write_text('300 unit:mol/(L*atm)\n0 1.3\n')
    elif bad=='nonfinite':path.write_text('298 unit:mol/(L*atm)\n0 nan\n')
    else:path.write_text('298 unit:mol/(L*atm)\n0 1.3\n0 1.4\n')
    assert not inspect(root,work)['complete_material_outputs']


def test_fabricated_converged_log_cannot_certify_science_without_owned_producer_binding(tmp_path):
    root,work,run=fixture(tmp_path)
    (run/'solver.log').write_text('Termination status: 0\nmaximum norm for gradient: 1e-10\n')
    assert inspect(root,work)['scientific_convergence'] is None


def test_bound_future_native_receipt_retains_failed_exit_even_with_valid_outputs(tmp_path):
    root,work,run=fixture(tmp_path)
    (run/'solver.log').write_text('Termination status: 7\nmaximum norm for gradient: 1.2\n')
    (run/'stderr.log').write_text('native line search failure\n')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (run/'execution.json').write_text(json.dumps({'exit_code':7,'timed_out':False,
        'input_sha256':digest(run/'input.dat'),'stdout_sha256':digest(run/'solver.log'),
        'stderr_sha256':digest(run/'stderr.log')}))
    facts=inspect(root,work)
    assert facts['technical_completion'] is False and facts['scientific_convergence'] is False


def test_total_byte_budget_stops_further_file_reads(tmp_path,monkeypatch):
    import agents.cdft_result_facts as module
    root,work,_=fixture(tmp_path)
    monkeypatch.setattr(module,'MAX_TOTAL_BYTES',(work/'input/a.dat').stat().st_size,raising=False)
    original=module._read_regular;calls=[]
    def observe(path,**kwargs):
        calls.append(str(path));return original(path,**kwargs)
    monkeypatch.setattr(module,'_read_regular',observe)
    facts=inspect(root,work)
    assert len(calls)==1 and not facts['complete_material_outputs']


def test_collection_quality_uses_original_inputs_and_detects_missing_copied_material(tmp_path):
    from agents.cdft_result_facts import inspect_cdft_batch_quality
    root,work,run=fixture(tmp_path)
    original=root/'original';original.mkdir()
    (original/'a.dat').write_bytes((work/'input/a.dat').read_bytes())
    (original/'missing.dat').write_bytes((work/'input/a.dat').read_bytes())
    facts=inspect_cdft_batch_quality(work,owned_root=root,gases=['N2'],source_input_dir=original)
    assert facts['expected_materials_verified'] and set(facts['expected_materials'])=={'a','missing'}
    assert not facts['complete_material_outputs']
    assert facts['scientific_convergence'] is None


def test_missing_original_input_provenance_is_unknown_not_a_fake_failure(tmp_path):
    from agents.cdft_result_facts import inspect_cdft_batch_quality
    root,work,_=fixture(tmp_path)
    facts=inspect_cdft_batch_quality(work,owned_root=root,gases=['N2'])
    assert facts['status']=='unknown' and not facts['expected_materials_verified']
    assert facts['scientific_convergence'] is None and facts.get('failed') is not True
