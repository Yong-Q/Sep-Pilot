"""Synthetic LAMMPS protocol fixtures; never execute MPI or scientific software."""
import importlib.util,re,shlex
from pathlib import Path
import pytest


def runtime():
    path=Path(__file__).resolve().parents[1]/'tools/md_runtime.py'
    spec=importlib.util.spec_from_file_location('md_stage_budget_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run_fake(tmp_path,monkeypatch,**options):
    md=runtime();cif=tmp_path/'input.cif';cif.write_text('synthetic')
    monkeypatch.setattr(md,'_executable',lambda x:x)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK',raising=False)
    captured=[]
    def execute(argv,cwd,env,stdout,**kw):
        captured.append(argv);cwd=Path(cwd)
        if argv[0]=='configured-interface':
            (cwd/'in.input').write_text('units real\nread_data data.input\nlog log.input\n')
            (cwd/'data.input').write_text('synthetic data')
        else:
            script=Path(argv[-1]).read_text();captured.append(script)
            out=shlex.split(re.findall(r'^write_data (.+)$',script,re.M)[-1])[0];Path(out).write_text('synthetic result')
            for stage in ('warmup','equilibration','production'):
                match=re.search('SEPPILOT_STAGE '+stage+r' begin.*?\nrun (\d+)',script,re.S)
                if match:stdout.write(f'SEPPILOT_STAGE {stage} begin\nLoop time of 1 on 4 procs for {match[1]} steps with 3 atoms\nSEPPILOT_STAGE {stage} end\n')
            stdout.write('Temp Press Volume PotEng KinEng\n298 1 100 -2 3\n')
    monkeypatch.setattr(md.subprocess,'run',execute)
    result=md.run(str(cif),str(tmp_path/'out'),executable='configured-lammps',mpi_executable='configured-mpi',
        interface_executable='configured-interface',mode='md',temperature=298,n_procs=4,skip_convert=True,**options)
    return result,captured


def test_explicit_short_stages_reach_native_input_without_hidden_warmup(tmp_path,monkeypatch):
    result,captured=run_fake(tmp_path,monkeypatch,warmup_steps=0,equilibration_steps=0,production_steps=20)
    assert result['success']
    script=next(x for x in captured if isinstance(x,str))
    assert 'run 40000' not in script and 'run 20000' not in script
    assert result['stage_budgets']=={'warmup_steps':0,'equilibration_steps':0,'production_steps':20}
    assert result['actual_steps']=={'warmup':0,'equilibration':0,'production':20,'total':20}
    assert captured[1][:4]==['configured-mpi','-np','4','configured-lammps']


def test_existing_md_defaults_and_duration_conversion_are_preserved(tmp_path,monkeypatch):
    result,_=run_fake(tmp_path,monkeypatch,npt_ps=.1,timestep=.5)
    assert result['success'] and result['stage_budgets']=={'warmup_steps':40000,'equilibration_steps':20000,'production_steps':200}


@pytest.mark.parametrize('options',[{'warmup_steps':-1},{'warmup_steps':True},{'equilibration_steps':1.2},{'production_steps':0},{'production_steps':False}])
def test_invalid_budgets_fail_before_interface_or_mpi(tmp_path,monkeypatch,options):
    result,captured=run_fake(tmp_path,monkeypatch,**options)
    assert not result['success'] and not captured


def test_schema_exposes_phase_budgets_and_executor_preserves_zero(tmp_path,monkeypatch):
    from agents import registry
    from types import SimpleNamespace
    schema=registry._build_default_registry()
    args={'cif_path':'input.cif','mode':'md','temperature':298,'warmup_steps':0,'equilibration_steps':0,'production_steps':20}
    assert not schema.validate_params('run_md_optimize',args), (registry.__file__, list(schema.get('run_md_optimize').input_schema['properties']))
    monkeypatch.setattr(registry,'get_config',lambda:SimpleNamespace(project_root=tmp_path,lammps_config={'executable':'configured-lammps','mpi_executable':'configured-mpi'}))
    monkeypatch.setattr(registry,'_ws',lambda _:str(tmp_path/'out'))
    monkeypatch.setattr(registry,'_register_jobwatch',lambda *a,**k:None)
    captured=[]
    monkeypatch.setattr(registry.slurm,'submit_and_return',lambda **kw:captured.append(kw) or {'submitted':True,'job_id':'42'})
    result=registry._exec_md_optimize(args)
    assert captured and '"warmup_steps": 0' in captured[0]['command'] and '"production_steps": 20' in captured[0]['command']
    assert result['stage_budgets']=={'warmup_steps':0,'equilibration_steps':0,'production_steps':20}
    assert 'actual_steps' not in result


def test_actual_steps_require_executed_native_stage_markers():
    md=runtime()
    assert md._actual_stage_steps('Loop time of 1 on 4 procs for 20 steps with 3 atoms') is None
    assert md._actual_stage_steps('print "SEPPILOT_STAGE production begin"\nLoop time of 1 on 4 procs for 20 steps with 3 atoms\nprint "SEPPILOT_STAGE production end"') is None
    assert md._actual_stage_steps('SEPPILOT_STAGE production begin\nSEPPILOT_STAGE production end') is None


def test_single_geometry_mode_rejects_md_integration_budget_before_execution(tmp_path,monkeypatch):
    md=runtime();p=tmp_path/'input.cif';p.write_text('synthetic')
    monkeypatch.setattr(md.subprocess,'run',lambda *a,**k:pytest.fail('must reject before interface/MPI'))
    result=md.run(str(p),str(tmp_path/'out'),executable='not-run',mode='single',production_steps=20)
    assert not result['success'] and 'mode=md' in result['error']
    from agents.registry import _build_default_registry
    assert _build_default_registry().validate_params('run_md_optimize',{'cif_path':str(p),'production_steps':20})

def test_executed_input_digest_attests_exact_script_and_mpi_command(tmp_path,monkeypatch):
    import hashlib
    result,captured=run_fake(tmp_path,monkeypatch,warmup_steps=0,equilibration_steps=0,production_steps=20)
    fact=result['execution_input'];source=Path(fact['path'])
    assert result['success'] and captured[1][-2:]==['-in',str(source)]
    assert fact['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert fact['bytes']==len(source.read_bytes()) and fact['verification']=='sha256_before_and_after_solver'
    assert fact['scientific_acceptance'] is False

def test_executed_input_hash_rejects_symlink(tmp_path):
    md=runtime();real=tmp_path/'real';real.write_text('input');alias=tmp_path/'alias';alias.symlink_to(real)
    with pytest.raises(OSError):md._execution_input_fact(alias)

def test_input_change_during_solver_is_not_attested(tmp_path,monkeypatch):
    md=runtime();cif=tmp_path/'input.cif';cif.write_text('synthetic')
    monkeypatch.setattr(md,'_executable',lambda x:x);monkeypatch.delenv('SLURM_CPUS_PER_TASK',raising=False)
    def execute(argv,cwd,env,stdout,**kw):
        cwd=Path(cwd)
        if argv[0]=='configured-interface':
            (cwd/'in.input').write_text('units real\nread_data data.input\nlog log.input\n')
            (cwd/'data.input').write_text('synthetic data')
        else:
            Path(argv[-1]).write_text('tampered after solver invocation')
    monkeypatch.setattr(md.subprocess,'run',execute)
    result=md.run(str(cif),str(tmp_path/'out'),executable='configured-lammps',mpi_executable='configured-mpi',
        interface_executable='configured-interface',mode='md',temperature=298,n_procs=4,skip_convert=True,
        warmup_steps=0,equilibration_steps=0,production_steps=20)
    assert result['success'] is False and 'changed during native execution' in result['error']
    assert 'execution_input' not in result

def test_existing_prepared_input_is_never_overwritten(tmp_path,monkeypatch):
    md=runtime();cif=tmp_path/'input.cif';cif.write_text('synthetic')
    monkeypatch.setattr(md,'_executable',lambda x:x);monkeypatch.delenv('SLURM_CPUS_PER_TASK',raising=False)
    def execute(argv,cwd,env,stdout,**kw):
        cwd=Path(cwd)
        if argv[0]=='configured-interface':
            (cwd/'in.input').write_text('units real\nread_data data.input\nlog log.input\n')
            (cwd/'data.input').write_text('synthetic data')
            (cwd/'md_in.input').write_text('preserve existing')
        else:pytest.fail('existing input must block MPI invocation')
    monkeypatch.setattr(md.subprocess,'run',execute)
    result=md.run(str(cif),str(tmp_path/'out'),executable='configured-lammps',mpi_executable='configured-mpi',
        interface_executable='configured-interface',mode='md',temperature=298,n_procs=4,skip_convert=True,
        warmup_steps=0,equilibration_steps=0,production_steps=20)
    assert result['success'] is False
    assert (tmp_path/'out/interface/md_in.input').read_text()=='preserve existing'
