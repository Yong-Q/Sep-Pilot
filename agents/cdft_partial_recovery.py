"""Owned timeout-only cDFT recovery preparation and immutable result merging.

This module never runs a solver, submits a job, alters a prior result, changes
workflow state, or certifies scientific convergence. Submission belongs to an
explicitly approved new DAG node and its ordinary RecoveryGate.
"""
import copy
import hashlib
import json
import os
import re
from pathlib import Path

from .report_recovery import _OwnedFiles, _same, _canonical
from .local_preparation_recovery import checkpoint_call_witness
from .recovery import RecoveryGate, input_fingerprint, result_object
from .workload_budget import collect_workload_facts
from .workload_context import _read_regular
from .native_timing import timed_phase, timed_call


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def walltime_seconds(value):
    match = re.fullmatch(r'([0-9]+):([0-5][0-9]):([0-5][0-9])', str(value))
    if not match: raise ValueError('explicit HH:MM:SS walltime is required')
    return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3])


@timed_phase('original_attempt')
def prove_completed_attempt(state, step_id, root, goal, jobs, *, inspection=False):
    """Authenticate one returned attempt; only read-only inspection is branch-local.

    Preparation, submission and merging retain the full approved graph guard.
    The private inspection mode never grants admission or accepts any result.
    """
    from .parallel_workflow import RESOURCE_ARGUMENTS, same_scientific_contract
    owner = {key: state.get(key) for key in ('username', 'conv_id')}
    node = state.get('nodes', {}).get(step_id, {});contract = node.get('contract', {})
    args = node.get('effective_arguments') or contract.get('arguments', {})
    approvals = [item for item in goal.get('approved_nodes', []) if item.get('step_id') == step_id]
    protected=('version','method','gases','parameters','execution_mode','execution_authorized','approved_plan_version')
    saved=state.get('goal_contract',{})
    approved={item.get('step_id'):item for item in goal.get('approved_nodes',[])}
    current_graph=state.get('nodes',{})
    from .native_resource_carry import approved_graph_matches, proof_matches
    current_graph_matches=(inspection is True or (len(approved)==len(goal.get('approved_nodes',[])) and set(approved)==set(current_graph)
        and (all(same_scientific_contract(item.get('contract',{}),approved[key]) for key,item in current_graph.items())
             or approved_graph_matches(state,goal.get('approved_nodes',[]),root,goal))))
    source_matches=(len(approvals)==1 and (same_scientific_contract(contract,approvals[0])
        or proof_matches(state,step_id,approvals[0],root,goal,node.get('approved_resource_carry'))))
    if (contract.get('tool') != 'run_cdft' or args.get('action') != 'submit'
            or node.get('status') not in {'succeeded', 'prefinish', 'validation_failed'}
            or node.get('tool_returned') is not True or not node.get('attempt_id') or not node.get('token')
            or node.get('dispatch_phase') != 'entered_tool' or goal.get('execution_authorized') is not True
            or state.get('plan_version') != goal.get('approved_plan_version')
            or not _same({key:saved.get(key) for key in protected},{key:goal.get(key) for key in protected})
            or not current_graph_matches
            or not source_matches):
        raise ValueError('partial recovery requires the exact approved owned completed cDFT attempt')
    files = _OwnedFiles(root, owner)
    try:
        ref = node.get('result_ref') or {};call_id = ref.get('call_id')
        if not isinstance(call_id, str) or not call_id.isalnum(): raise ValueError('original execution call is required')
        path = files.root / 'evidence' / (call_id + '.json')
        if ref.get('evidence_path') != str(path): raise ValueError('execution evidence path is not owned')
        call, evidence = files.read(path, as_json=True);native = result_object(call.get('result'))
        science = lambda values: {k: v for k, v in values.items() if k not in RESOURCE_ARGUMENTS}
        if (call.get('call_id') != call_id or call.get('tool') != 'run_cdft' or call.get('failed') is not False
                or not _same(science(call.get('params') or {}), science(args))
                or not _same(native, result_object(node.get('result')))):
            raise ValueError('original native execution receipt or scientific inputs changed')
        witness = timed_call('recorded_preparation_witness', checkpoint_call_witness,
                             files, ref.get('checkpoint_path', ''), call, step_id=step_id)
        job_id = str(native.get('job_id') or '')
        entry = RecoveryGate(path=files.root / 'recovery_state.json').snapshot().get(node.get('recovery_key'), {})
        job = jobs.get(job_id, {})
        if (not job_id.isdigit() or set(map(str, node.get('job_ids') or [])) != {job_id}
                or entry.get('attempt_id') != node['attempt_id'] or entry.get('tool') != 'run_cdft'
                or entry.get('status') not in {'completed', 'completed_unverified'}
                or set(map(str, entry.get('job_ids') or [])) != {job_id}
                or not _same(entry.get('params'), call.get('params'))
                or entry.get('fingerprint') != timed_call('original_input_sha', input_fingerprint,
                    entry.get('params') or {}, files.root.parents[2])
                or (job.get('username'), job.get('conv_id')) != (owner['username'], owner['conv_id'])
                or job.get('tool') != 'run_cdft' or job.get('attempt_id') != node['attempt_id']
                or job.get('recovery_key') != node.get('recovery_key')
                or job.get('terminal') is not True or job.get('state') != 'COMPLETED' or job.get('failed') is not False):
            raise ValueError('original scheduler job, attempt and recovery ledger do not agree')
        work = files.owned(native.get('work_dir', ''))
        approved = Path(contract['arguments'].get('job_work_dir') or files.root / 'cdft')
        if not approved.is_absolute(): approved = files.root.parents[2] / approved
        if (not work.is_relative_to(approved.resolve()) or str(work) != job.get('work_dir')
                or result_object(entry.get('last_result')).get('work_dir') != str(work)):
            raise ValueError('original result root is outside its approved owned receipt')
        script = files.owned(native.get('script_path') or native.get('script') or '')
        if script.parent != work: raise ValueError('batch script belongs to another result root')
        _, script_fact = files.read(script)
        return {'owner': owner, 'step_id': step_id, 'token': node['token'], 'attempt_id': node['attempt_id'],
            'job_id': job_id, 'work_dir': str(work), 'contract': copy.deepcopy(contract),
            'arguments': copy.deepcopy(args), 'execution_call_id': call_id, 'execution_evidence': evidence,
            'producer_checkpoint': witness, 'script': script_fact,
            'input_dir': str(files.owned(args.get('input_dir', '')))}
    finally: files.close()


@timed_phase('timeout_subset')
def prove_timeout_subset(attempt, root):
    """Every retry case must have an explicit native timeout, never a blank CSV alone."""
    from .cdft_result_facts import inspect_cdft_batch_quality
    work = Path(attempt['work_dir']);args = attempt['arguments']
    gases = args.get('gases') or [args.get('gas')]
    quality = timed_call('material_quality', inspect_cdft_batch_quality, work, owned_root=root, gases=gases,
                         source_input_dir=attempt['input_dir'])
    if not quality.get('expected_materials_verified'):
        raise ValueError('original owned input material identities are not verified')
    inputs = quality['original_input_facts'];names = set(quality['expected_materials'])
    if set(quality.get('input_materials', [])) != names or set(quality.get('run_materials', [])) != names:
        raise ValueError('original copied input/run inventory is incomplete or ambiguous')
    for item in inputs['inputs']:
        observed = quality['materials'][Path(item['path']).stem]['artifacts'].get(str(work / 'input' / item['path']))
        if not observed or observed['sha256'] != item['sha256'] or observed['size'] != item['bytes']:
            raise ValueError('original copied input does not match the submitted scientific input')
    script, script_sha = timed_call('timeout_script_joblog', _read_regular, Path(attempt['script']['path']), limit=1024*1024)
    if script_sha != attempt['script']['sha256']: raise ValueError('original script changed')
    limits = re.findall(rb'^export CDFT_TIMEOUT=[\"\x27]?([1-9][0-9]*)[\"\x27]?\s*$', script, re.MULTILINE)
    if len(limits) != 1: raise ValueError('original native per-case timeout is not explicit')
    timeout = int(limits[0]);log = work / ('slurm_' + attempt['job_id'] + '.log')
    raw, log_sha = timed_call('timeout_script_joblog', _read_regular, log, limit=8*1024*1024)
    text = raw.decode('utf-8', errors='strict')
    successful = re.findall(r'^  SUCCESS: ([^\r\n]+)$', text, re.MULTILINE)
    timed_out = re.findall(r'^  TIMEOUT: ([^\r\n]+) exceeded ([1-9][0-9]*)s, killing\.$', text, re.MULTILINE)
    failed = [name for name, _ in timed_out]
    if (len(successful) != len(set(successful)) or len(failed) != len(set(failed))
            or set(successful) & set(failed) or set(successful) | set(failed) != names
            or not successful or not failed or any(int(limit) != timeout for _, limit in timed_out)):
        raise ValueError('batch has unknown, duplicate or non-timeout failures; no inferred retry permission')
    for name in successful:
        material = quality['materials'][name]
        if material['errors'] or material.get('output_valid') is not True:
            raise ValueError('a preserved material lacks complete finite native output')
    return {**attempt, 'schema_version': 1, 'kind': 'cdft_timeout_subset',
        'original_input_facts': inputs, 'quality': quality, 'gases': gases,
        'successful_materials': sorted(successful), 'retry_materials': sorted(failed),
        'previous_timeout_seconds': timeout, 'joblog': {'path': str(log), 'sha256': log_sha, 'size': len(raw)},
        'scientific_convergence': None}


def stage_recovery(proof, root, timeout_seconds, walltime, *, control_source, wait_job_ids=()):
    """Only immutable local preparation. Existing outputs and state are untouched."""
    from .raspa_assets import _publish
    if (type(timeout_seconds) is not int or timeout_seconds <= proof['previous_timeout_seconds']
            or timeout_seconds >= walltime_seconds(walltime)):
        raise ValueError('retry requires an explicitly longer evidenced timeout inside its walltime')
    manifest = {'schema_version': 1, 'kind': 'cdft_timeout_subset_recovery', 'original': copy.deepcopy(proof),
        'timeout_seconds': timeout_seconds, 'walltime': walltime,
        'timeout_basis': 'authorized_operations_extension_after_verified_native_timeout',
        'control_source': copy.deepcopy(control_source), 'wait_job_ids': sorted(set(wait_job_ids))}
    identity = digest(manifest)
    directory = Path(root) / 'cdft_recovery' / identity
    subset = directory / 'inputs'
    expected = {Path(item['path']).stem: item for item in proof['original_input_facts']['inputs']}
    for name in proof['retry_materials']:
        item = expected[name];path = Path(proof['input_dir']) / item['path']
        raw, checksum = _read_regular(path, limit=16*1024*1024)
        if checksum != item['sha256'] or len(raw) != item['bytes']:
            raise ValueError('retry input changed since its native timeout proof')
        _publish(subset, item['path'], raw)
    manifest.update(input_dir=str(subset), recovery_id=identity)
    raw = _canonical(manifest);path = directory / 'recovery_manifest.json'
    _publish(directory, path.name, raw)
    retry_args = {key:value for key,value in proof['contract']['arguments'].items()
                  if key not in {'memory_mb','num_processes','cpus_per_task','nodelist','partition','resource_review_id','output'}}
    retry_args.update(action='submit',input_dir=str(subset),job_work_dir=str(directory/'retry'),
        partial_recovery_manifest=str(path),timeout=timeout_seconds,walltime=walltime,
        wait_job_ids=manifest['wait_job_ids'],completion_policy='strict')
    retry_id='cdft_retry_'+identity[:24];merge_id='cdft_merge_'+identity[:24]
    merge_args={key:value for key,value in proof['contract']['arguments'].items()
        if key not in {'action','input_dir','job_work_dir','output','memory_mb','num_processes','cpus_per_task',
            'nodelist','partition','resource_review_id','walltime','timeout','wait_job_ids'}}
    merge_args.update(action='collect_recovery',partial_recovery_manifest=str(path),retry_step_id=retry_id,
        job_work_dir=retry_args['job_work_dir'],output=str(directory/'merged/results.csv'),completion_policy='strict')
    nodes=[{'step_id':retry_id,'agent':proof['contract']['agent'],'tool':'run_cdft','arguments':retry_args,
            'depends_on':[],'expected_outputs':[{'kind':'file','path':str(directory/'retry/results.csv')}]},
           {'step_id':merge_id,'agent':proof['contract']['agent'],'tool':'run_cdft','arguments':merge_args,
            'depends_on':[proof['step_id'],retry_id],
            'expected_outputs':[{'kind':'file','path':str(directory/'merged/results.csv')}]}]
    return {'ok': True, 'submitted': False, 'accepted_result': False,
        'kind': 'cdft_partial_recovery_preparation', 'recovery_id': identity,
        'manifest_path': str(path), 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'input_dir': str(subset), 'original_material_count': len(expected),
        'retry_materials': proof['retry_materials'], 'preserved_materials': proof['successful_materials'],
        'timeout_seconds': timeout_seconds, 'walltime': walltime, 'wait_job_ids': manifest['wait_job_ids'],
        'retry_arguments':retry_args,'merged_output_csv':str(directory/'merged/results.csv'),
        'proposal':{'changes':[{'operation':'upsert','step_id':node['step_id'],'node':node} for node in nodes],
            'retry_step_id':retry_id,'merge_step_id':merge_id,'preserved_source_step':proof['step_id'],
            'requires_formal_approval':True},
        'scientific_convergence': None, 'requires_approved_recovery_dag': True}


def bind_downstream_proposal(receipt,state,source_step):
    """Retarget exact typed CSV inputs only; never rewrite arbitrary shell code."""
    proposal=receipt['proposal'];merge_id=proposal['merge_step_id']
    source=state['nodes'][source_step]
    old_csv=str(Path(result_object(source.get('result')).get('work_dir',''))/'results.csv')
    updates=[];unresolved=[]
    for step,item in state.get('nodes',{}).items():
        contract=item.get('contract',{})
        if source_step not in contract.get('depends_on',[]) or item.get('status')=='succeeded':continue
        if item.get('status')!='pending':
            unresolved.append({'step_id':step,'reason':'consumer has already entered execution; explicit owned reconciliation is required'})
            continue
        revised=copy.deepcopy(contract);rebound=[]
        ambiguous=False
        for key,value in revised.get('arguments',{}).items():
            if key in {'data_csv','input_path'} and value==old_csv:
                binding=revised.get('input_bindings',{}).get(key)
                if binding is not None:
                    from .output_contract import output_path
                    outputs=source.get('contract',{}).get('expected_outputs',[])
                    index=binding.get('output_index') if isinstance(binding,dict) else None
                    valid=(isinstance(binding,dict) and binding.get('step_id')==source_step
                        and type(index) is int and 0<=index<len(outputs))
                    if valid:
                        try:valid=str(output_path(outputs[index],Path(old_csv).parent))==old_csv
                        except (OSError,ValueError,KeyError,TypeError):valid=False
                    if not valid:ambiguous=True;continue
                revised['arguments'][key]=receipt['merged_output_csv'];rebound.append(key)
        if ambiguous or not rebound:
            unresolved.append({'step_id':step,'tool':contract.get('tool'),'requires_dependency':merge_id,
                'original_csv':old_csv,'required_csv':receipt['merged_output_csv'],
                'reason':'no exact typed CSV input; recompile the existing scientific analysis with registered typed tools, never scan newest results or rewrite shell heuristically'})
            continue
        revised['depends_on']=list(dict.fromkeys([*revised['depends_on'],merge_id]))
        for key in rebound:
            if key in revised.get('input_bindings',{}):
                revised['input_bindings'][key]={'step_id':merge_id,'output_index':0}
        updates.append({'operation':'upsert','step_id':step,'node':revised})
    proposal['changes'].extend(updates)
    proposal['downstream_recompile_required']=unresolved
    proposal['complete']=not unresolved
    return receipt


def load_recovery_manifest(path, *, root, state, goal, jobs):
    """Revalidate preparation against a real compact Session call and original job."""
    files = _OwnedFiles(root, {key: state.get(key) for key in ('username', 'conv_id')})
    try:
        path = files.owned(path);relative = path.relative_to(files.root)
        if (len(relative.parts) != 3 or relative.parts[0] != 'cdft_recovery'
                or not re.fullmatch('[0-9a-f]{64}', relative.parts[1]) or relative.parts[2] != 'recovery_manifest.json'):
            raise ValueError('recovery manifest is not a native owned preparation path')
        manifest, fact = files.read(path, as_json=True)
        core = {key: value for key, value in manifest.items() if key not in {'input_dir', 'recovery_id'}}
        if (manifest.get('kind') != 'cdft_timeout_subset_recovery' or manifest.get('schema_version') != 1
                or manifest.get('recovery_id') != relative.parts[1] or digest(core) != relative.parts[1]
                or manifest.get('input_dir') != str(path.parent / 'inputs')):
            raise ValueError('recovery manifest identity or local input scope changed')
        original = manifest['original'];step = original['step_id']
        if state.get('nodes',{}).get(step,{}).get('status')!='succeeded':
            raise ValueError('strict partial source requires an explicit native subsumption workflow; no inferred original success')
        expected_params = {'step_id': step, 'timeout_seconds': manifest['timeout_seconds'], 'walltime': manifest['walltime']}
        checkpoint, _ = files.read(files.root / 'session_checkpoint.json', as_json=True)
        candidates = []
        for item in checkpoint.get('memory', {}).get('tool_call_log', [])[-4096:]:
            if (item.get('tool') != 'prepare_cdft_partial_recovery' or item.get('failed') is not False
                    or not _same(item.get('params'), expected_params)): continue
            call_id = item.get('call_id')
            if not isinstance(call_id, str) or not call_id.isalnum(): continue
            call, _ = files.read(files.root / 'evidence' / (call_id + '.json'), as_json=True)
            result = result_object(call.get('result'))
            if result.get('manifest_path') != str(path) or result.get('manifest_sha256') != fact['sha256']: continue
            timed_call('recorded_preparation_witness', checkpoint_call_witness, files,
                       files.root / 'session_checkpoint.json', call, require_result_hash=True)
            if (call.get('tool') != 'prepare_cdft_partial_recovery' or call.get('failed') is not False
                    or not _same(call.get('params'), expected_params)
                    or result.get('kind') != 'cdft_partial_recovery_preparation'
                    or result.get('retry_materials') != original['retry_materials']
                    or result.get('preserved_materials') != original['successful_materials']
                    or result.get('input_dir') != manifest['input_dir']): continue
            candidates.append(call_id)
        if not candidates: raise ValueError('no authentic recorded native preparation supports this manifest')
        current = prove_timeout_subset(prove_completed_attempt(state, step, root, goal, jobs), root)
        if not _same(current, original): raise ValueError('original partial results or scientific inputs changed after preparation')
        if (type(manifest.get('timeout_seconds')) is not int
                or not original['previous_timeout_seconds'] < manifest['timeout_seconds'] < walltime_seconds(manifest['walltime'])):
            raise ValueError('retry timeout no longer matches its approved evidence')
        waiting = manifest.get('wait_job_ids')
        if (not isinstance(waiting, list) or waiting != sorted(set(waiting))
                or any(not isinstance(job, str) or not job.isdigit() for job in waiting)):
            raise ValueError('recovery scheduler dependencies must be owned native job identities')
        owned_jobs = {str(job): node for node in state.get('nodes', {}).values() for job in node.get('job_ids', [])}
        for jid in waiting:
            node = owned_jobs.get(jid, {});watched = jobs.get(jid, {})
            if ((watched.get('username'), watched.get('conv_id')) != (state['username'], state['conv_id'])
                    or not node or watched.get('attempt_id') != node.get('attempt_id')
                    or watched.get('recovery_key') != node.get('recovery_key')):
                raise ValueError('recovery dependency escaped its current owned workflow')
        expected = [item for item in original['original_input_facts']['inputs']
                    if Path(item['path']).stem in set(original['retry_materials'])]
        subset = timed_call('retry_subset_sha', collect_workload_facts, 'run_cdft', {'action': 'submit', 'input_dir': manifest['input_dir']},
            owned_root=files.root, manifest={'schema_version': 1,
                'items': [{key: item[key] for key in ('path', 'sha256')} for item in expected]})
        if subset.get('status') != 'ready' or not _same(subset.get('inputs'), expected):
            raise ValueError('retry subset does not exactly equal the verified timeout input hashes')
        return manifest
    finally: files.close()


def owned_workflow_context(root):
    """Read existing owned state; never construct an executor or change controls."""
    from .parallel_workflow import WorkflowStore
    from .job_watch import get_watch
    root = Path(root)
    workflow_id = WorkflowStore.identity(root.parent.name, root.name)
    state = WorkflowStore(root.parents[2] / 'data/state/parallel_workflows.json').snapshot(workflow_id)
    watch = get_watch()
    jobs = {str(job): watch.get(str(job)) or {} for node in state.get('nodes', {}).values()
            for job in node.get('job_ids', [])}
    return state, state.get('goal_contract', {}), jobs


def validate_subset_submission(arguments, *, root):
    """Bind approved retry parameters; arbitrary externally supplied IDs grant nothing."""
    state, goal, jobs = owned_workflow_context(root)
    manifest = load_recovery_manifest(arguments['partial_recovery_manifest'], root=root, state=state, goal=goal, jobs=jobs)
    from .parallel_workflow import RESOURCE_ARGUMENTS
    original = manifest['original']['arguments']
    scientific = lambda params: {key: value for key, value in params.items() if key not in RESOURCE_ARGUMENTS
        | {'input_dir', 'job_work_dir', 'output', 'timeout', 'walltime', 'wait_job_ids',
           'partial_recovery_manifest', 'completion_policy', 'min_success_ratio'}}
    if (not _same(scientific(arguments), scientific(original)) or arguments.get('input_dir') != manifest['input_dir']
            or arguments.get('timeout') != manifest['timeout_seconds'] or arguments.get('walltime') != manifest['walltime']
            or arguments.get('wait_job_ids', []) != manifest['wait_job_ids']):
        raise ValueError('retry submit parameters differ from the recorded failed-only preparation')
    approved = [node for node in goal.get('approved_nodes', []) if node.get('tool') == 'run_cdft'
                and node.get('arguments', {}).get('partial_recovery_manifest') == arguments['partial_recovery_manifest']
                and node.get('arguments', {}).get('action') == 'submit']
    if len(approved) != 1:
        raise ValueError('exactly one formally approved DAG retry must own this preparation')
    from .watch_context import get_context
    from .workflow_patch import execution_contract
    step = get_context().get('step_id');current = state.get('nodes', {}).get(step, {})
    if (step != approved[0]['step_id'] or current.get('status') != 'running'
            or current.get('dispatch_phase') != 'entered_tool' or not current.get('token')
            or execution_contract(current.get('contract', {})) != execution_contract(approved[0])
            or not _same(current.get('effective_arguments') or current['contract']['arguments'], arguments)
            or state.get('user_paused')):
        raise ValueError('failed-only submission must run through its actual approved DAG claim')
    for node in state.get('nodes', {}).values():
        for job in node.get('job_ids', []):
            observed = jobs.get(str(job), {})
            if observed.get('terminal') is not True and str(job) not in manifest['wait_job_ids']:
                raise ValueError('new active workflow job requires fresh same-budget recovery admission')
    return manifest


def merge_results(manifest, retry_attempt, *, root, control_valid=None):
    """Merge one exact source per material, retaining native convergence unknowns.

    The caller must authenticate both scheduler attempts. File checks here
    independently bind every selected material to its original input SHA.
    """
    import csv
    import io
    from .cdft_result_facts import inspect_cdft_batch_quality
    from .raspa_assets import _publish
    original = manifest['original'];gases = original['gases']
    retry = inspect_cdft_batch_quality(retry_attempt['work_dir'], owned_root=root, gases=gases,
                                      source_input_dir=manifest['input_dir'])
    if (not retry.get('expected_materials_verified') or retry.get('complete_material_outputs') is not True
            or retry.get('collection_errors') or set(retry['expected_materials']) != set(original['retry_materials'])):
        raise ValueError('retry does not provide every failed material with complete finite native outputs')
    expected = {Path(item['path']).stem: item for item in original['original_input_facts']['inputs']}
    observed = {Path(item['path']).stem: item for item in retry['original_input_facts']['inputs']}
    if any(not _same(observed[name], expected[name]) for name in original['retry_materials']):
        raise ValueError('retry changed the original scientific input SHA or material identity')
    materials = {name: copy.deepcopy(original['quality']['materials'][name]) for name in original['successful_materials']}
    materials.update({name: copy.deepcopy(retry['materials'][name]) for name in original['retry_materials']})
    if set(materials) != set(expected): raise ValueError('merged material inventory is incomplete or ambiguous')
    facts = {};budget = 0
    for material in materials.values():
        if material['errors'] or not material.get('output_valid'): raise ValueError('merged native result remains invalid')
        for path, fact in material['artifacts'].items():
            candidate = Path(path);candidate.relative_to(Path(root))
            if path not in facts: budget += fact['size']
            facts[path] = fact
    if budget > 128*1024*1024: raise ValueError('merged provenance exceeds bounded native evidence size')
    rows = []
    for name in sorted(materials):
        item = materials[name];components = item['output_facts']['components']
        row = {'MOF': name}
        for gas in gases: row[gas+'_density_mol_L'] = components[gas]['density_mol_L']
        for gas in gases: row[gas+'_percentage'] = components[gas]['percentage']
        # Keep absent total-density information unknown instead of deriving a
        # different quantity from framework mass or a guessed unit conversion.
        row['total_density_molec_A3'] = ''
        for gas in gases: row[gas+'_henry_mol_L_atm'] = item['henry_mol_L_atm'][gas]
        rows.append(row)
    quality = {'schema_version': 1, 'status': 'observed', 'expected_materials_verified': True,
        'expected_materials': sorted(expected), 'gases': gases, 'materials': materials,
        'input_materials':sorted(expected),'run_materials':sorted(materials),
        'inventory_layout':'provenance_bound_composite_attempts',
        'expected_materials_source':'immutable_original_inputs_and_owned_retry_material_sha',
        'original_input_facts': original['original_input_facts'], 'complete_material_outputs': True,
        'collection_errors': [], 'scientific_convergence': False if any(item.get('native_convergence') is False
            for item in materials.values()) else None,
        'technical_completion': True if all(item.get('technical_completion') is True for item in materials.values()) else None,
        'job_ownership_verified': True, 'hamiltonian_comparison_verified': False,
        'adsorption_heat_verified': False, 'producer_authentication_required': False,
        'limitations': ['Unique material/input SHA merge is not proof of scientific or discretization convergence.',
            'Original legacy optimizer or per-material process status remains unknown when not recorded.']}
    provenance = {'schema_version': 1, 'kind': 'cdft_partial_result_merge',
        'recovery_id': manifest['recovery_id'], 'source_job_ids': [original['job_id'], retry_attempt['job_id']],
        'source_execution_call_ids': [original['execution_call_id'], retry_attempt['execution_call_id']],
        'material_sources': {name: {'job_id': original['job_id'] if name in original['successful_materials'] else retry_attempt['job_id'],
            'input_sha256': expected[name]['sha256'], 'native_artifacts': materials[name]['artifacts']} for name in sorted(materials)},
        'cdft_quality': quality}
    def current():
        if control_valid is not None and control_valid() is not True:
            raise ValueError('result collection control or native ownership changed')
    from .workload_budget import _owned_fd
    observed_stats={}
    identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    current()
    for path, fact in facts.items():
        handle=_owned_fd(Path(root),Path(path))
        try:before=os.fstat(handle)
        finally:os.close(handle)
        raw, checksum = _read_regular(Path(path), limit=8*1024*1024)
        if checksum != fact['sha256'] or len(raw) != fact['size']:
            raise ValueError('original or retry native result changed during immutable merging')
        handle=_owned_fd(Path(root),Path(path))
        try:after=os.fstat(handle)
        finally:os.close(handle)
        if identity(before)!=identity(after):raise ValueError('native result path changed while hashing')
        observed_stats[path]=identity(after)
    for path,expected_stat in observed_stats.items():
        handle=_owned_fd(Path(root),Path(path))
        try:actual_stat=identity(os.fstat(handle))
        finally:os.close(handle)
        if actual_stat!=expected_stat:raise ValueError('previously hashed native result changed before publication')
    current()
    directory = Path(manifest['input_dir']).parent / 'merged'
    if not directory.is_absolute() or '..' in directory.parts or not directory.is_relative_to(Path(root)):
        raise ValueError('merged output escaped its owned recovery root')
    stream = io.StringIO(newline='');writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);csv_bytes = stream.getvalue().encode()
    provenance['csv_sha256'] = hashlib.sha256(csv_bytes).hexdigest()
    quality['collected_csv_sha256']=provenance['csv_sha256']
    from .cdft_result_facts import collected_cdft_row_errors
    quality['collection_errors']=collected_cdft_row_errors(rows,quality,gases)
    if quality['collection_errors']:raise ValueError('merged CSV differs from independently observed native values')
    _publish(directory, 'results.csv', csv_bytes)
    current()
    _publish(directory, 'merge_provenance.json', _canonical(provenance))
    current()
    return {'submitted': False, 'validation_passed': True, 'rows': rows,
        'output_csv': str(directory / 'results.csv'), 'output_files': [str(directory / 'results.csv')],
        'merge_manifest': str(directory / 'merge_provenance.json'), 'cdft_quality': quality,
        'source_job_ids': provenance['source_job_ids'], 'scientific_convergence': quality['scientific_convergence']}


def collect_recovery(arguments, *, root):
    """Registry entry: only a current approved collector can combine two owned jobs."""
    from .watch_context import get_context
    from .workflow_patch import execution_contract
    state, goal, jobs = owned_workflow_context(root)
    step = get_context().get('step_id');node = state.get('nodes', {}).get(step, {})
    if (state.get('user_paused') or node.get('contract', {}).get('tool') != 'run_cdft'
            or not _same(node.get('effective_arguments') or node.get('contract', {}).get('arguments'), arguments)
            or node.get('status') != 'running' or node.get('dispatch_phase') != 'entered_tool'):
        raise ValueError('result merge requires its own current approved DAG collector')
    approvals = [item for item in goal.get('approved_nodes', []) if item.get('step_id') == step]
    if len(approvals) != 1 or execution_contract(approvals[0]) != execution_contract(node['contract']):
        raise ValueError('result merge contract is not currently approved')
    manifest = load_recovery_manifest(arguments['partial_recovery_manifest'], root=root, state=state, goal=goal, jobs=jobs)
    retry_id = arguments.get('retry_step_id');retry = state.get('nodes', {}).get(retry_id, {})
    if (retry_id not in node['contract'].get('depends_on', [])
            or manifest['original']['step_id'] not in node['contract'].get('depends_on', [])
            or retry.get('contract', {}).get('arguments', {}).get('partial_recovery_manifest') != arguments['partial_recovery_manifest']):
        raise ValueError('merge dependencies do not bind the original and exact failed-only retry')
    attempt = prove_completed_attempt(state, retry_id, root, goal, jobs)
    approved_retry_root=Path(arguments.get('job_work_dir') or '')
    if (attempt['input_dir'] != manifest['input_dir'] or not approved_retry_root.is_absolute()
            or not Path(attempt['work_dir']).is_relative_to(approved_retry_root)):
        raise ValueError('merge result root does not match the actual owned retry scheduler receipt')
    target = str(Path(manifest['input_dir']).parent / 'merged/results.csv')
    if arguments.get('output') != target: raise ValueError('merge output must use its new immutable recovery directory')
    claim_identity=lambda value:{key:item for key,item in value.items() if key not in {'heartbeat','updated_at'}}
    captured = digest([goal, claim_identity(node)])
    def current():
        current_state, current_goal, current_jobs = owned_workflow_context(root)
        visible = current_state.get('nodes', {}).get(step, {})
        if current_state.get('user_paused') or digest([current_goal, claim_identity(visible)]) != captured:
            changed=[key for key in set(node)|set(visible) if node.get(key)!=visible.get(key)]
            raise ValueError('collector pause, goal or exact native claim changed during merge: '+','.join(sorted(changed)))
        original_attempt = prove_completed_attempt(current_state, manifest['original']['step_id'], root, current_goal, current_jobs)
        retry_attempt = prove_completed_attempt(current_state, retry_id, root, current_goal, current_jobs)
        return _same(original_attempt['execution_evidence'], manifest['original']['execution_evidence']) and _same(retry_attempt, attempt)
    return merge_results(manifest, attempt, root=root, control_valid=current)
