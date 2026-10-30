"""Trusted prepared cDFT observations, never model-provided resource facts."""
import copy,hashlib,os
from pathlib import Path
import pytest
from agents.workload_context import enrich_resource_request


def native_input(*,caps='60 60 60',lengths='4.5 5.5 6.5 0.5 90 90 90',rotations='2 3 4',sites=3,atoms=2):
    return f'''water 100 0
Nmaxx Nmaxy Nmaxz
{caps}
Lxa Lyb Lzc dx angle_a angle_b angle_c
{lengths}
Nutation Precession Rotation
{rotations}
Temperature(K) Kapa Delta_Max Torr
298 0.95 2 0.1
Kind_of_gas
1
Name Epsilon(K) Sigma(A) Bulk_Density(molec/A^3)
1 236 3.75 0.0000243
Special_case
0
Id1 Id2 Epsilon(K) Sigma(A)
Read_Vext EOS Cutoff deos N3_cutoff Diele_con Cutoff_inside cut_coul
0 1 12.9 1 0.99 1 0.5 15
Mass(g/mol) Ds_Knudsen(m^2/s) MFV_parameter
44 2e-8 0.712
Kind_of_solvent_atoms
2
ID diameter(A) Epsilon(K) charge
0 3.43 52.84 0.7
1 3.43 52.84 -0.35
Number_of_solvent_atoms1
{sites}
ID x y z
''' + ''.join('0 0 0 0\n' for _ in range(sites)) + f'Number_of_solute_atoms\n{atoms}\nID x y z diameter(A) Epsilon(K) charge\n' + ''.join('1 0 0 0 3 30 0\n' for _ in range(atoms))


def case(tmp_path):
    root=tmp_path/'owned';directory=root/'input';directory.mkdir(parents=True)
    (directory/'one.dat').write_text(native_input())
    exe=tmp_path/'solver';exe.write_bytes(b'configured executable');exe.chmod(0o700)
    req={'kind':'before_submission','nodes':[{'step_id':'one','tool':'run_cdft','arguments':{'action':'submit','input_dir':str(directory)},'workload_facts':{'status':'ready','work_items':999},'executable_sha256':'untrusted'}]}
    return root,directory,exe,req


def test_native_bounds_and_binary_identity_replace_model_claims_without_mutating_request(tmp_path):
    root,directory,exe,req=case(tmp_path);before=copy.deepcopy(req)
    out=enrich_resource_request(req,owned_root=root,cdft_executable=exe);n=out['nodes'][0];f=n['workload_facts']
    assert req==before
    assert f['status']=='ready' and f['work_items']==1
    assert f['native_bounds']=={'max_grid_points':10*12*14,'max_gases':1,'max_sites_per_gas':3,'max_framework_atoms':2,'max_rotations':24}
    assert n['executable_sha256']==hashlib.sha256(exe.read_bytes()).hexdigest()
    assert f['inputs'][0]['sha256']==hashlib.sha256((directory/'one.dat').read_bytes()).hexdigest()


@pytest.mark.parametrize('change',['bad_count','nan','zero_grid','truncated','input_symlink','outside_input','exe_symlink','exe_fifo','missing_exe'])
def test_unproven_inputs_or_executable_cannot_supply_ready_bounds(tmp_path,change):
    root,d,exe,req=case(tmp_path)
    if change=='bad_count':(d/'one.dat').write_text(native_input().replace('Kind_of_gas\n1','Kind_of_gas\n-1'))
    elif change=='nan':(d/'one.dat').write_text(native_input(lengths='nan 5 6 .5 90 90 90'))
    elif change=='zero_grid':(d/'one.dat').write_text(native_input(caps='0 60 60'))
    elif change=='truncated':(d/'one.dat').write_text(native_input()[:-15])
    elif change=='input_symlink':(d/'one.dat').unlink();(d/'one.dat').symlink_to(exe)
    elif change=='outside_input':req['nodes'][0]['arguments']['input_dir']=str(tmp_path)
    elif change=='exe_symlink':link=tmp_path/'link';link.symlink_to(exe);exe=link
    elif change=='exe_fifo':exe=tmp_path/'fifo';os.mkfifo(exe)
    elif change=='missing_exe':exe=None
    n=enrich_resource_request(req,owned_root=root,cdft_executable=exe)['nodes'][0]
    assert n['workload_facts']['status']!='ready'
    assert 'native_bounds' not in n['workload_facts'] and 'executable_sha256' not in n


def test_pipeline_and_unsupported_requests_cannot_retain_invented_facts(tmp_path):
    root,d,exe,req=case(tmp_path);req['nodes'][0]['arguments']['action']='pipeline'
    req['nodes'].append({'tool':'run_string_tst','arguments':{},'workload_facts':{'status':'ready'},'executable_sha256':'fake'})
    nodes=enrich_resource_request(req,owned_root=root,cdft_executable=exe)['nodes']
    assert [n['workload_facts']['status'] for n in nodes]==['needs_prepared_inputs','unsupported_workload']
    assert all('executable_sha256' not in n for n in nodes)


@pytest.mark.parametrize('lengths,axes', [
    ('4 5.5 6.5 0.5 90 90 90', [9,12,14]),
    ('4.5 5 6.5 0.5 90 90 90', [10,11,14]),
    ('4.5 5.5 6 0.5 90 90 90', [10,12,13]),
    ('4 5 6 0.5 90 90 90', [9,11,13]),
])
def test_native_observation_rejects_each_unsupported_odd_axis_without_trusting_profile(tmp_path,lengths,axes):
    root,directory,exe,request=case(tmp_path)
    before=copy.deepcopy(request)
    (directory/'one.dat').write_text(native_input(lengths=lengths))
    observed=enrich_resource_request(request,owned_root=root,cdft_executable=exe)['nodes'][0]
    assert request==before
    assert observed['workload_facts']['status']=='invalid_workload'
    assert observed['workload_facts']['reason']=='unsupported_native_grid'
    assert observed['workload_facts']['grid_axes']==axes
    assert 'native_bounds' not in observed['workload_facts']
    assert 'executable_sha256' not in observed
