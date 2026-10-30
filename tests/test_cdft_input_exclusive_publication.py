"""Prepared native inputs are immutable even if an output appears after approval."""
import importlib.util
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest


@pytest.fixture
def generator(tmp_path,monkeypatch):
    p=Path(__file__).resolve().parents[1]/'tools/cdft/cDFT_Initialization/data_input.py'
    spec=importlib.util.spec_from_file_location('exclusive_cdft_generator',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    uff=tmp_path/'uff';uff.mkdir()
    for gas in ('N2','CO2'):
        (uff/f'{gas}.def').write_text('Kind_of_solvent_atoms\n1\nID diameter(A) Epsilon(K) charge\n0 3.4 30 0\nNumber_of_solvent_atoms\n1\nID x y z\n0 0 0 0\n')
    monkeypatch.setattr(m,'_UFF_DIR',str(uff));monkeypatch.setattr(m,'_load_cg_ff',lambda path:{g:{'mass':28,'sigma':3.4,'epsilon':30} for g in ('N2','CO2')})
    monkeypatch.setattr(m,'_parse_cif_tokens',lambda lines:(40,40,40,90,90,90,0,[['C',0,0,0,3.4,30,12,0.1]]))
    return m


def invoke(m,source,out,gas='CO2'):
    return m.generate_cdft_inputs(source,[gas],298,[0.0000243],output_dir=out,framework_charge=False)


def sources(tmp_path,*names):
    root=tmp_path/'cifs';root.mkdir()
    for name in names:(root/f'{name}.cif').write_text('data_synthetic\n')
    return root


@pytest.mark.parametrize('gas',['N2','CO2'])
def test_first_native_publication_succeeds_for_each_supported_gas(tmp_path,generator,gas):
    source=sources(tmp_path,'one');out=tmp_path/'new/inputs'
    files=invoke(generator,source,out,gas)
    assert files==[str(out/'one.dat')]
    assert (out/'one.dat').read_text().startswith('water')


def test_all_target_collisions_are_checked_before_any_new_native_file(tmp_path,generator):
    source=sources(tmp_path,'a','z');out=tmp_path/'out';out.mkdir();(out/'z.dat').write_text('existing native evidence')
    with pytest.raises(FileExistsError):invoke(generator,source,out)
    assert (out/'z.dat').read_text()=='existing native evidence' and not (out/'a.dat').exists()


@pytest.mark.parametrize('collision',['file_symlink','directory','output_symlink','ancestor_symlink'])
def test_existing_or_symlinked_outputs_are_never_followed_or_replaced(tmp_path,generator,collision):
    source=sources(tmp_path,'one');out=tmp_path/'out';victim=tmp_path/'victim';victim.mkdir();old=victim/'old';old.write_text('preserve')
    if collision=='output_symlink':out.symlink_to(victim,target_is_directory=True)
    elif collision=='ancestor_symlink':out.symlink_to(victim,target_is_directory=True);out=out/'new_child'
    else:
        out.mkdir()
        if collision=='file_symlink':(out/'one.dat').symlink_to(old)
        else:(out/'one.dat').mkdir()
    with pytest.raises(OSError):invoke(generator,source,out)
    assert old.read_text()=='preserve'
    if collision=='ancestor_symlink':assert not (victim/'new_child').exists()
    elif collision=='output_symlink':assert not (victim/'one.dat').exists()


def test_execution_time_collision_preserves_evidence_created_after_preflight(tmp_path,generator,monkeypatch):
    source=sources(tmp_path,'one');out=tmp_path/'out';parse=generator._parse_cif_tokens
    def race(lines):
        (out/'one.dat').write_text('late evidence')
        return parse(lines)
    monkeypatch.setattr(generator,'_parse_cif_tokens',race)
    with pytest.raises(FileExistsError):invoke(generator,source,out)
    assert (out/'one.dat').read_text()=='late evidence'


def test_concurrent_native_writers_have_one_winner_and_no_overwrite(tmp_path,generator,monkeypatch):
    source=sources(tmp_path,'one');out=tmp_path/'out';parse=generator._parse_cif_tokens;barrier=threading.Barrier(2)
    def synchronized(lines):barrier.wait(timeout=5);return parse(lines)
    monkeypatch.setattr(generator,'_parse_cif_tokens',synchronized)
    def run():
        try:invoke(generator,source,out);return 'created'
        except FileExistsError:return 'collision'
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:run(),range(2)))
    assert sorted(results)==['collision','created']
    assert (out/'one.dat').read_text().startswith('water')



@pytest.mark.parametrize('existing',[False,True])
def test_failed_generation_preserves_partial_native_input_as_evidence(tmp_path,generator,monkeypatch,existing):
    import functools,hashlib
    source=sources(tmp_path,'one');out=tmp_path/'out';out.mkdir()
    if existing:invoke(generator,source,out)
    note=out/'failure-audit.note';note.write_bytes(b'existing evidence must remain')
    before={p.name:(p.read_bytes(),p.stat().st_ino,p.stat().st_mtime_ns)for p in out.iterdir()}
    original=generator._orig_generate_cdft_inputs;partial=[]
    @functools.wraps(original)
    def observed(*args,**kwargs):
        try:return original(*args,**kwargs)
        except RuntimeError:
            candidate=Path(kwargs['output_dir'])/'one.dat'
            partial.append(candidate.read_bytes())
            raise
    def fail(*args):raise RuntimeError('synthetic asset failure after header')
    monkeypatch.setattr(generator,'_orig_generate_cdft_inputs',observed)
    monkeypatch.setattr(generator,'_load_uff_def',fail)
    with pytest.raises(RuntimeError,match='synthetic asset failure'):invoke(generator,source,out)
    assert partial and partial[0].startswith(b'water')
    assert {p.name:(p.read_bytes(),p.stat().st_ino,p.stat().st_mtime_ns)for p in out.iterdir()}==before
    assert (out/'one.dat').exists()is existing
