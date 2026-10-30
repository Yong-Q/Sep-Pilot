"""Owned failed-only cDFT recovery: fake solver/scheduler, real evidence files."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from agents.state_io import json_transaction
from tests.test_cdft_result_facts import fixture as native_fixture
from tests.test_parallel_workflow import factory, node, eventually


@pytest.fixture
def partial_cdft(factory, tmp_path):
    scope = tmp_path / 'runs/tester/partial'
    inputs = scope / 'source'
    work = scope / 'original/data/batch'
    args = {'action': 'submit', 'gas': 'N2', 'temperature': 298, 'input_dir': str(inputs),
            'job_work_dir': str(scope / 'original'), 'memory_mb': 4096}
    rt = factory([node('cdft', tool='run_cdft', args=args, outputs=[str(work / 'results.csv')])], cid='partial')
    inputs.mkdir();(work / 'input').mkdir(parents=True)
    _, example, run = native_fixture(tmp_path / 'template')
    payload = (example / 'input/a.dat').read_bytes()
    for name in ['amber', 'birch', 'cedar']:
        (inputs / (name + '.dat')).write_bytes(payload)
        (work / 'input' / (name + '.dat')).write_bytes(payload)
        target = work / 'input_dir' / name;target.mkdir(parents=True)
        (target / 'input.dat').write_bytes(payload)
        if name != 'cedar':
            for file in ['output.dat', 'hr.dat']:
                (target / file).write_bytes((run / file).read_bytes())
    (work / 'slurm_42.log').write_text('  SUCCESS: amber\n  SUCCESS: birch\n  TIMEOUT: cedar exceeded 12s, killing.\n')
    script = work / 'cdft_job.sh';script.write_text('export CDFT_TIMEOUT=12\n')
    (work / 'results.csv').write_text('MOF,N2_density_mol_L,N2_henry_mol_L_atm\namber,1.2,1.3\nbirch,1.2,1.3\ncedar,,\n')
    calls = []
    def submit(params):
        calls.append(copy.deepcopy(params))
        return {'submitted': True, 'job_id': '42', 'work_dir': str(work), 'script_path': str(script), 'input_dir': str(inputs)}
    rt.main.registry.get('run_cdft').execute = submit
    def watched(job):
        current = rt.snapshot()['nodes']['cdft']
        return {'job_id': '42', 'username': rt.username, 'conv_id': rt.conv_id, 'tool': 'run_cdft',
                'attempt_id': current.get('attempt_id'), 'recovery_key': current.get('recovery_key'),
                'terminal': True, 'failed': False, 'state': 'COMPLETED', 'work_dir': str(work)}
    rt.job_watch = SimpleNamespace(get=watched, list=lambda: [])
    rt.start(1);rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['cdft']['status'] == 'waiting_jobs')
    rt.tick([{'job_id': '42', 'terminal': True, 'failed': False, 'state': 'COMPLETED'}])
    assert rt.snapshot()['nodes']['cdft']['status'] == 'prefinish'
    # Historical production bug being repaired: legacy kernel accepted partial
    # outputs. Preserve this past state; the new recovery never grants success.
    with json_transaction(rt.store.path) as data:
        old=data['workflows'][rt.workflow_id]['nodes']['cdft']
        old.update(status='succeeded',phase='finish',node_verification={'legacy_partial_acceptance':True})
        for key,lease in list(data.get('leases',{}).items()):
            if lease.get('workflow_id')==rt.workflow_id and lease.get('step_id')=='cdft':data['leases'].pop(key)
    rt._step('cdft',status='completed',done=True,validation={'output_contract':'passed'})
    source = {'origin': 'user', 'turn_id': 'authenticated-turn', 'owner': {'username': 'tester', 'conv_id': 'partial'}}
    from agents.defns import ORCHESTRATOR
    rt.main.current_agent = ORCHESTRATOR;rt.main._control_source = source
    return rt, work, calls, source


def prepare(rt, source, **changes):
    return rt.prepare_cdft_partial_recovery('cdft', timeout_seconds=24, walltime='00:01:00',
        control_source=source, authority_role='lead-orchestrator', **changes)


def test_native_preparation_stages_only_verified_timeout_and_never_reexecutes_successes(partial_cdft):
    rt, work, calls, source = partial_cdft
    before = rt.store.path.read_bytes();old = {str(p): p.read_bytes() for p in work.rglob('*') if p.is_file()}
    result = prepare(rt, source)
    assert result['submitted'] is False and result['accepted_result'] is False
    assert result['retry_materials'] == ['cedar'] and result['preserved_materials'] == ['amber', 'birch']
    assert len(result['proposal']['changes'])==2
    assert result['proposal']['requires_formal_approval'] is True
    subset = Path(result['input_dir'])
    assert [p.name for p in subset.iterdir()] == ['cedar.dat']
    assert (subset / 'cedar.dat').read_bytes() == (work / 'input/cedar.dat').read_bytes()
    assert rt.store.path.read_bytes() == before and len(calls) == 1
    assert all(Path(name).read_bytes() == value for name, value in old.items())


@pytest.mark.parametrize('change', ['unknown_failure', 'changed_input', 'symlink', 'source', 'token', 'timeout'])
def test_partial_recovery_rejects_unknown_or_unapproved_evidence(partial_cdft, change):
    rt, work, calls, source = partial_cdft;kwargs = {}
    if change == 'unknown_failure':(work / 'slurm_42.log').write_text('  SUCCESS: amber\n  SUCCESS: birch\n  FAILED: cedar\n')
    elif change == 'changed_input':(work / 'input/amber.dat').write_text('changed')
    elif change == 'symlink':
        p = work / 'input/cedar.dat';p.unlink();p.symlink_to(work / 'input/amber.dat')
    elif change == 'source':source = {**source, 'origin': 'system'}
    elif change == 'token':kwargs['expected_token'] = 'stale'
    else:
        with pytest.raises(ValueError):
            rt.prepare_cdft_partial_recovery('cdft', timeout_seconds=12, walltime='00:01:00',
                control_source=source, authority_role='lead-orchestrator')
        return
    with pytest.raises((ValueError, PermissionError)):
        prepare(rt, source, **kwargs)
    assert len(calls) == 1


def test_native_preparation_does_not_publish_authority_after_control_changes(partial_cdft, monkeypatch):
    rt, work, _, source = partial_cdft
    import agents.cdft_partial_recovery as recovery
    original = recovery.stage_recovery;valid = {'value': True}
    def changed(*args, **kwargs):
        result = original(*args, **kwargs);valid['value'] = False;return result
    monkeypatch.setattr(recovery, 'stage_recovery', changed)
    with pytest.raises(ValueError): prepare(rt, source, control_valid=lambda: valid['value'])


def record_preparation(rt, source):
    from agents.state_io import compact_checkpoint_evidence, write_checkpoint, apply_evidence_references
    result = prepare(rt, source)
    rt.main.memory.record_tool_call('lead-orchestrator', 'prepare_cdft_partial_recovery',
        {'step_id': 'cdft', 'timeout_seconds': 24, 'walltime': '00:01:00'}, json.dumps(result), failed=False)
    saved = rt.main.export_state();saved['owner'] = {'username': rt.username, 'conv_id': rt.conv_id}
    refs = compact_checkpoint_evidence(saved, rt.root / 'evidence')
    write_checkpoint(rt.root / 'session_checkpoint.json', saved)
    apply_evidence_references(rt.main.memory, refs)
    return result


@pytest.mark.parametrize('recorded', [False, True])
def test_subset_submission_requires_actual_recorded_preparation_not_a_manifest_alone(partial_cdft, recorded):
    from agents.cdft_partial_recovery import load_recovery_manifest
    rt, _, _, source = partial_cdft
    result = record_preparation(rt, source) if recorded else prepare(rt, source)
    if recorded:
        manifest = load_recovery_manifest(result['manifest_path'], root=rt.root,
            state=rt.snapshot(), goal=rt.main.goal_contract.to_dict(),
            jobs={'42': rt.job_watch.get('42')})
        assert manifest['original']['retry_materials'] == ['cedar']
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            load_recovery_manifest(result['manifest_path'], root=rt.root,
                state=rt.snapshot(), goal=rt.main.goal_contract.to_dict(), jobs={'42': rt.job_watch.get('42')})


def test_changed_success_output_invalidates_recovery_manifest(partial_cdft):
    from agents.cdft_partial_recovery import load_recovery_manifest
    rt, work, _, source = partial_cdft;result = record_preparation(rt, source)
    (work / 'input_dir/amber/hr.dat').write_text('298 unit:mol/(L*atm)\n0 9\n')
    with pytest.raises(ValueError):
        load_recovery_manifest(result['manifest_path'], root=rt.root,
            state=rt.snapshot(), goal=rt.main.goal_contract.to_dict(), jobs={'42': rt.job_watch.get('42')})


@pytest.fixture
def retry_results(partial_cdft, tmp_path):
    from agents.cdft_partial_recovery import load_recovery_manifest
    rt, work, calls, source = partial_cdft;prepared = record_preparation(rt, source)
    manifest = load_recovery_manifest(prepared['manifest_path'], root=rt.root,
        state=rt.snapshot(), goal=rt.main.goal_contract.to_dict(), jobs={'42': rt.job_watch.get('42')})
    retry = rt.root / 'retry/data/batch';(retry / 'input').mkdir(parents=True)
    (retry / 'input/cedar.dat').write_bytes((Path(prepared['input_dir']) / 'cedar.dat').read_bytes())
    run = retry / 'input_dir/cedar';run.mkdir(parents=True)
    for source_name, destination in [('input/cedar.dat', 'input.dat'),
            ('input_dir/amber/output.dat', 'output.dat'), ('input_dir/amber/hr.dat', 'hr.dat')]:
        (run / destination).write_bytes((work / source_name).read_bytes())
    attempt = {'job_id': '43', 'work_dir': str(retry), 'input_dir': prepared['input_dir'],
               'execution_call_id': 'actual-retry', 'attempt_id': 'retry-attempt'}
    return rt, work, calls, prepared, manifest, attempt


def test_merge_selects_unique_materials_with_exact_original_input_sha_and_keeps_unknown_convergence(retry_results):
    from agents.cdft_partial_recovery import merge_results
    rt, work, calls, prepared, manifest, attempt = retry_results
    old = {str(p): p.read_bytes() for p in work.rglob('*') if p.is_file()}
    result = merge_results(manifest, attempt, root=rt.root)
    assert result['submitted'] is False and result['validation_passed'] is True
    assert [row['MOF'] for row in result['rows']] == ['amber', 'birch', 'cedar']
    assert result['cdft_quality']['complete_material_outputs'] is True
    assert result['cdft_quality']['scientific_convergence'] is None
    from agents.result_validation import cdft_completion_assessment
    assessed=cdft_completion_assessment({'contract':{'arguments':{'completion_policy':'strict'}}},result['cdft_quality'])
    assert assessed['verdict']=='pass' and assessed['counts']=={'attempted':3,'succeeded':3,'failed':0}
    assert len(calls) == 1 and all(Path(name).read_bytes() == raw for name, raw in old.items())
    output = Path(result['output_csv']);before = output.stat()
    assert merge_results(manifest, attempt, root=rt.root)['output_csv'] == str(output)
    assert output.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize('change', ['retry_input', 'missing_result', 'extra_material', 'old_success', 'output_collision'])
def test_merge_never_overwrites_or_fills_missing_scientific_results(retry_results, change):
    from agents.cdft_partial_recovery import merge_results
    rt, work, calls, prepared, manifest, attempt = retry_results;retry = Path(attempt['work_dir'])
    if change == 'retry_input':(retry / 'input/cedar.dat').write_text('different scientific recipe')
    elif change == 'missing_result':(retry / 'input_dir/cedar/hr.dat').unlink()
    elif change == 'extra_material':(retry / 'input_dir/extra').mkdir()
    elif change == 'old_success':(work / 'input_dir/amber/hr.dat').write_text('changed')
    else:
        output = Path(prepared['manifest_path']).parent / 'merged/results.csv';output.parent.mkdir()
        output.write_text('existing other report')
    with pytest.raises((ValueError, FileExistsError)):
        merge_results(manifest, attempt, root=rt.root)
    assert len(calls) == 1


def test_future_merge_contract_uses_new_manifest_output_and_never_drops_other_outputs(tmp_path):
    from agents.workflow_patch import complete_compute_contract
    from agents.output_contract import normalize_output
    scope=tmp_path/'runs/tester/case';manifest=scope/'cdft_recovery'/('a'*64)/'recovery_manifest.json'
    candidate={'tool':'run_cdft','arguments':{'action':'collect_recovery','partial_recovery_manifest':str(manifest),
        'retry_step_id':'retry','job_work_dir':str(scope/'retry')},'expected_outputs':[]}
    result=complete_compute_contract(candidate,'merge',tmp_path,scope)
    expected=manifest.parent/'merged/results.csv'
    assert result['arguments']['output']==str(expected)
    assert result['expected_outputs']==[{'kind':'file','path':str(expected)}]
    candidate['expected_outputs']=[{'kind':'directory','path':str(scope/'extra'),'min_count':3}]
    with pytest.raises(ValueError):complete_compute_contract(candidate,'merge',tmp_path,scope)


def test_registry_retry_then_native_merge_runs_only_subset_with_actual_dag_claims(partial_cdft, monkeypatch):
    import subprocess
    import agents.registry as registry
    import agents.cdft_partial_recovery as recovery
    from agents.config import set_config
    from agents.watch_context import set_context, clear_context
    rt, original, calls, source = partial_cdft;prepared = record_preparation(rt, source)
    # Reproduce the already-delivered legacy state: the old kernel marked this
    # partial batch finished. New recovery still independently proves its two
    # valid outputs and one timeout; this fixture flag grants no native proof.
    with json_transaction(rt.store.path) as data:
        source_node=data['workflows'][rt.workflow_id]['nodes']['cdft']
        source_node.update(status='succeeded',phase='finish',node_verification={'legacy_partial_acceptance':True})
        for key,lease in list(data.get('leases',{}).items()):
            if lease.get('workflow_id')==rt.workflow_id and lease.get('step_id')=='cdft':data['leases'].pop(key)
    rt._step('cdft',status='completed',done=True,validation={'output_contract':'passed'})
    args = {**prepared['retry_arguments'], 'memory_mb': 2048, 'num_processes': 1}
    retry_root = Path(args['job_work_dir']);work = retry_root / 'data/batch';(work / 'input').mkdir(parents=True)
    (work / 'input/cedar.dat').write_bytes((Path(prepared['input_dir']) / 'cedar.dat').read_bytes())
    script = work / 'cdft_job.sh';script.write_text('export CDFT_TIMEOUT=24\n')
    retry = node('retry',tool='run_cdft',args=args,outputs=[str(work/'results.csv')])
    merge_args = {'action':'collect_recovery','partial_recovery_manifest':prepared['manifest_path'],
        'retry_step_id':'retry','job_work_dir':str(retry_root),'output':prepared['merged_output_csv'],'gas':'N2'}
    merger = node('merge',tool='run_cdft',args=merge_args,deps=['cdft','retry'],outputs=[prepared['merged_output_csv']])
    original_contract=copy.deepcopy(rt.snapshot()['nodes']['cdft']['contract'])
    rt.main.goal_contract.approved_nodes=[original_contract,retry,merger]
    rt.main.goal_contract.approved_plan_version=2
    rt.task_lines.apply_workflow_patch(rt.main._current_line_id,[{'operation':'upsert','step_id':n['step_id'],'node':n} for n in [retry,merger]],1,2,'authorized failed-only recovery')
    old_watch=rt.job_watch.get
    def watched(job):
        if str(job)=='42':return old_watch(job)
        current=rt.snapshot().get('nodes',{}).get('retry',{})
        return {'job_id':'43','username':rt.username,'conv_id':rt.conv_id,'tool':'run_cdft',
            'attempt_id':current.get('attempt_id'),'recovery_key':current.get('recovery_key'),
            'terminal':True,'failed':False,'state':'COMPLETED','work_dir':str(work)}
    rt.job_watch.get=watched
    monkeypatch.setattr(recovery,'owned_workflow_context',lambda root:(rt.snapshot(),rt.main.goal_contract.to_dict(),
        {'42':watched('42'),'43':watched('43')}))
    monkeypatch.setattr(registry,'_cdft_adapter_bootstrap',lambda *a:('',[]))
    monkeypatch.setattr(registry,'_scientific_python',lambda *a:'python')
    import agents.installations
    original_layout=agents.installations.cdft_installation_files
    monkeypatch.setattr(agents.installations,'cdft_installation_files',lambda *a:{**original_layout(*a),'cdft_executable':(script,'executable')})
    monkeypatch.setattr(registry,'_register_jobwatch',lambda *a,**k:None)
    actual_submit=[]
    def native_submit(command,**kwargs):
        actual_submit.append(command)
        return subprocess.CompletedProcess(command,0,json.dumps({'submitted':True,'job_id':'43','work_dir':str(work),
            'script_path':str(script),'n_mofs':1,'status':'submitted'}),'')
    monkeypatch.setattr(subprocess,'run',native_submit)
    set_config(rt.main.config)
    rt.main.registry.get('run_cdft').execute=registry._exec_cdft
    rt.start(2);rt.tick()
    eventually(lambda:rt.snapshot()['nodes']['retry']['status'] in {'waiting_jobs','failed'})
    assert rt.snapshot()['nodes']['retry']['status']=='waiting_jobs',rt.snapshot()['nodes']['retry'].get('error')
    assert len(actual_submit)==1 and len(calls)==1
    target=work/'input_dir/cedar';target.mkdir(parents=True)
    for filename in ['input.dat','output.dat','hr.dat']:
        source_file=original/('input/cedar.dat' if filename=='input.dat' else 'input_dir/amber/'+filename)
        (target/filename).write_bytes(source_file.read_bytes())
    (work/'results.csv').write_text('MOF,N2_density_mol_L,N2_henry_mol_L_atm\ncedar,1.2,1.3\n')
    rt.tick([{'job_id':'43','terminal':True,'failed':False,'state':'COMPLETED'}])
    # The new retry's real native dossier must be accepted through the normal
    # owner path before the dependent collector can receive a claim.
    assert rt.snapshot()['nodes']['merge']['status']=='pending'
    before=rt.snapshot()['nodes']['cdft']['result_ref']
    with pytest.raises(ValueError):
        set_context(rt.username,rt.conv_id,step_id='merge')
        try:recovery.collect_recovery(merge_args,root=rt.root)
        finally:clear_context()
    assert rt.snapshot()['nodes']['cdft']['result_ref']==before and len(actual_submit)==1
    dossier=rt.snapshot()['nodes']['retry']['validation_dossier']
    rt.finish_node('retry',[dossier['evidence_call_id']],
        'The failed-only functional retry now has its one complete finite native material; scientific convergence is unestablished.')
    rt.tick()
    eventually(lambda:rt.snapshot()['nodes']['merge']['status'] in {'succeeded','failed','validation_failed'})
    assert rt.snapshot()['nodes']['merge']['status']=='succeeded',rt.snapshot()['nodes']['merge'].get('error')
    assert Path(prepared['merged_output_csv']).is_file()
    assert rt.snapshot()['nodes']['cdft']['result_ref']==before and len(actual_submit)==1 and len(calls)==1


def test_confirmed_failed_scheduler_batch_is_not_automatically_finished_or_retried(partial_cdft):
    rt,work,calls,source=partial_cdft;original=rt.job_watch.get
    rt.job_watch.get=lambda job:{**original(job),'state':'FAILED','failed':True,'terminal':True}
    with json_transaction(rt.store.path) as data:
        data['workflows'][rt.workflow_id]['nodes']['cdft']['status']='failed'
    before=rt.store.path.read_bytes()
    with pytest.raises(ValueError):prepare(rt,source)
    assert rt.store.path.read_bytes()==before and len(calls)==1


@pytest.mark.parametrize('status',['prefinish','validation_failed'])
def test_future_strict_partial_source_needs_native_subsumption_and_is_not_falsely_finished(partial_cdft,status):
    rt,work,calls,source=partial_cdft
    with json_transaction(rt.store.path) as data:data['workflows'][rt.workflow_id]['nodes']['cdft']['status']=status
    before=rt.store.path.read_bytes()
    with pytest.raises(ValueError,match='subsumption'):prepare(rt,source)
    assert rt.store.path.read_bytes()==before and len(calls)==1


@pytest.mark.parametrize('foreign',[False,True])
def test_afterany_dependencies_are_derived_only_from_current_owned_workflow_jobs(partial_cdft,foreign):
    rt,work,calls,source=partial_cdft
    sibling={'step_id':'sibling','agent':'analyst','tool':'run_cdft','arguments':{'action':'submit'}}
    rt.main.goal_contract.approved_nodes.append(copy.deepcopy(sibling))
    with json_transaction(rt.store.path) as data:
        state=data['workflows'][rt.workflow_id];state['goal_contract']=rt.main.goal_contract.to_dict()
        state['nodes']['sibling']={'contract':sibling,'status':'waiting_jobs','job_ids':['99'],
            'token':'sibling-token','attempt_id':'sibling-attempt','recovery_key':'sibling-key'}
    old=rt.job_watch.get
    rt.job_watch.get=lambda job:old(job) if str(job)=='42' else {
        'username':'other' if foreign else rt.username,'conv_id':rt.conv_id,'terminal':False,'state':'RUNNING',
        'attempt_id':'sibling-attempt','recovery_key':'sibling-key'}
    if foreign:
        with pytest.raises(ValueError):prepare(rt,source)
    else:
        result=prepare(rt,source)
        assert result['wait_job_ids']==['99'] and result['retry_arguments']['wait_job_ids']==['99']
    assert len(calls)==1


def test_arbitrary_scheduler_dependency_cannot_reach_submission(partial_cdft,monkeypatch):
    import subprocess
    from agents import registry
    rt,work,calls,source=partial_cdft
    invoked=[];monkeypatch.setattr(subprocess,'run',lambda *a,**k:invoked.append(a))
    result=registry._exec_cdft({'action':'submit','input_dir':str(work/'input'),'wait_job_ids':['999']})
    assert result['blocked'] is True and result['executed'] is False and invoked==[]


def test_merge_rechecks_previously_hashed_sources_before_publication(retry_results,monkeypatch):
    import agents.cdft_partial_recovery as recovery
    rt,work,calls,prepared,manifest,attempt=retry_results
    original=recovery._read_regular;seen={'victim':False}
    victim=work/'input_dir/amber/hr.dat'
    def mutate_later(path,**kwargs):
        result=original(path,**kwargs)
        if Path(path)==victim:seen['victim']=True
        if str(path).endswith('/cedar/hr.dat') and seen['victim']:
            victim.write_text('298 unit:mol/(L*atm)\n0 9\n')
        return result
    monkeypatch.setattr(recovery,'_read_regular',mutate_later)
    with pytest.raises(ValueError):recovery.merge_results(manifest,attempt,root=rt.root)
    assert not (Path(prepared['manifest_path']).parent/'merged/results.csv').exists()


@pytest.mark.parametrize('control',[None,'cancel','new_uid'])
def test_real_owner_api_preparation_records_compact_native_witness_only_for_current_turn(partial_cdft,monkeypatch,control):
    import asyncio
    import threading
    import api
    from agents.state_io import compact_checkpoint_evidence,write_checkpoint,apply_evidence_references
    from agents.cdft_partial_recovery import load_recovery_manifest
    rt,work,calls,source=partial_cdft
    conv=SimpleNamespace(username=rt.username,conv_id=rt.conv_id,session=rt.main,
        current_uid='owned-turn',is_processing=False,interrupt_requested=False,interrupt_message='')
    rt.main._evidence_root=rt.root/'evidence';monkeypatch.setattr(rt.main,'_persist_chain',lambda *a,**k:None)
    def checkpoint(saved,reason):
        saved['owner']={'username':rt.username,'conv_id':rt.conv_id}
        refs=compact_checkpoint_evidence(saved,rt.root/'evidence')
        write_checkpoint(rt.root/'session_checkpoint.json',saved)
        apply_evidence_references(rt.main.memory,refs)
    rt.main._on_checkpoint=checkpoint
    monkeypatch.setattr(api.session_manager,'get_user',lambda _:SimpleNamespace(get_conversation=lambda cid:conv))
    monkeypatch.setattr(api.session_manager,'persist_conversations',lambda *a,**k:None)
    monkeypatch.setattr(api,'_ensure_workflow_runtime',lambda _:rt)
    lease=threading.Lock();monkeypatch.setattr(api,'_get_conv_lock',lambda *a:lease)
    original=rt.prepare_cdft_partial_recovery;entered=threading.Event();release=threading.Event()
    if control:
        def delayed(*args,**kwargs):
            entered.set();assert release.wait(3);return original(*args,**kwargs)
        monkeypatch.setattr(rt,'prepare_cdft_partial_recovery',delayed)
    async def scenario():
        request=api.CDFTPartialPrepareRequest(timeout_seconds=24,walltime='00:01:00')
        task=asyncio.create_task(api.prepare_cdft_partial_retry(rt.conv_id,'cdft',request,username=rt.username))
        if control:
            assert await asyncio.to_thread(entered.wait,2)
            if control=='cancel':task.cancel()
            else:conv.current_uid='new-turn'
            release.set()
            with pytest.raises(asyncio.CancelledError if control=='cancel' else api.HTTPException):await task
            return None
        return await task
    result=asyncio.run(scenario())
    records=[call for call in rt.main.memory.tool_call_log if call['tool']=='prepare_cdft_partial_recovery']
    if control:
        assert records==[]
    else:
        assert len(records)==1 and records[0]['agent']=='operations' and 'result' not in records[0]
        manifest=load_recovery_manifest(result['manifest_path'],root=rt.root,state=rt.snapshot(),
            goal=rt.main.goal_contract.to_dict(),jobs={'42':rt.job_watch.get('42')})
        assert manifest['original']['retry_materials']==['cedar']
    assert len(calls)==1 and not lease.locked()


@pytest.mark.parametrize('change',['saved_output_summary','current_contract','protected_goal'])
def test_native_binding_summary_drift_does_not_replace_current_approved_graph_authority(partial_cdft,change):
    rt,work,calls,source=partial_cdft
    with json_transaction(rt.store.path) as data:
        state=data['workflows'][rt.workflow_id]
        state['goal_contract']['approved_nodes'][0]['expected_outputs']=[str(rt.root/'old-placeholder/results.csv')]
        if change=='current_contract':state['nodes']['cdft']['contract']['arguments']['temperature']=310
        if change=='protected_goal':state['goal_contract']['parameters']['temperature']=310
    if change=='saved_output_summary':
        assert prepare(rt,source)['retry_materials']==['cedar']
    else:
        with pytest.raises(ValueError):prepare(rt,source)
    assert len(calls)==1


def test_native_lease_can_prepare_while_preserving_user_pause_and_preventing_old_rank_dispatch(partial_cdft):
    rt,work,calls,source=partial_cdft
    rt.store.user_pause(rt.workflow_id,True)
    before=rt.store.path.read_bytes()
    with pytest.raises(ValueError):prepare(rt,source)
    with pytest.raises(PermissionError):prepare(rt,source,allow_paused=True)
    result=prepare(rt,source,allow_paused=True,control_valid=lambda:True)
    assert result['submitted'] is False and rt.store.path.read_bytes()==before and len(calls)==1


@pytest.mark.parametrize('typed',[True,False])
def test_proposal_rebinds_exact_csv_consumers_and_refuses_to_guess_shell_science(partial_cdft,typed):
    from agents.cdft_partial_recovery import bind_downstream_proposal
    rt,work,calls,source=partial_cdft;result=prepare(rt,source)
    state=rt.snapshot()
    contract={'step_id':'ranking','tool':'analyze_cdft_screening' if typed else 'run_bash','agent':'analyst',
        'arguments':{'data_csv':str(work/'results.csv')} if typed else {'command':'scan newest results.csv'},
        'depends_on':['cdft'],'expected_outputs':[]}
    state['nodes']['ranking']={'status':'pending','contract':contract}
    before=copy.deepcopy(state)
    proposal=bind_downstream_proposal(result,state,'cdft')['proposal']
    if typed:
        updated=proposal['changes'][-1]['node']
        assert updated['arguments']['data_csv']==result['merged_output_csv']
        assert proposal['merge_step_id'] in updated['depends_on'] and proposal['complete'] is True
    else:
        assert proposal['complete'] is False and len(proposal['changes'])==2
        assert proposal['downstream_recompile_required'][0]['step_id']=='ranking'
    assert state==before and len(calls)==1


@pytest.mark.parametrize('binding',[{'step_id':'different_source','output_index':0},{'step_id':'cdft','output_index':9}])
def test_literal_csv_cannot_override_a_different_or_unproved_authoritative_input_binding(partial_cdft,binding):
    from agents.cdft_partial_recovery import bind_downstream_proposal
    rt,work,calls,source=partial_cdft;receipt=prepare(rt,source);state=rt.snapshot()
    state['nodes']['rank']={'status':'pending','contract':{'step_id':'rank','tool':'analyze_cdft_screening',
        'arguments':{'data_csv':str(work/'results.csv')},'depends_on':['cdft'],
        'input_bindings':{'data_csv':binding},'expected_outputs':[]}}
    result=bind_downstream_proposal(receipt,state,'cdft')
    assert result['proposal']['complete'] is False
    assert all(change['step_id']!='rank' for change in result['proposal']['changes'])
