"""JSON-only native MD producer -> real packet, sandbox and report workers.

Scientific files/scheduler receipts are explicitly synthetic engineering data;
the actual source probe separately checks the completed real MD calculation.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock

from agents.config import AgentConfig
from agents.session import Session, SessionMemory
from agents.state_io import compact_checkpoint_evidence, write_checkpoint
from agents.output_contract import artifact_fact
from agents.recovery import RecoveryGate, input_fingerprint
from agents.report_authoring import build_result_packet, sources_unchanged, _read
from agents.report_recovery import _OwnedFiles
from agents.local_preparation_recovery import checkpoint_call_witness
from tests.test_parallel_workflow import factory, node
from tests.test_md_thermo_history import block


def cif(a):
    return f'''data_fixture
_cell_length_a {a}
_cell_length_b 12
_cell_length_c 13
_cell_angle_alpha 90
_cell_angle_beta 100
_cell_angle_gamma 110
'''


def native_log():
    text = 'LAMMPS controlled engineering fixture\n'
    for count in (14, 26, 2):
        text += block('minimization', steps=count, headers='Step Temp TotEng',
                      rows=f'0 280 10\n{count} 281 3\n')
    for stage, count in (('warmup', 0), ('equilibration', 0), ('production', 20)):
        text += 'SEPPILOT_STAGE ' + stage + ' begin\n'
        text += block(steps=count, headers='Step Temp TotEng', rows=f'0 282 10\n{count} 290 4\n')
        text += 'SEPPILOT_STAGE ' + stage + ' end\n'
    return (text + block(steps=0, headers='Temp Press PotEng KinEng', rows='290 2 3 1\n')).encode()


def producer_evidence(root, owner, step, tool, params, result):
    memory = SessionMemory()
    memory.record_tool_call('analyst', tool, params, json.dumps(result, sort_keys=True))
    call = memory.tool_call_log[-1]
    checkpoint = root / 'workers' / (step + '-producer') / 'session_checkpoint.json'
    state = {'schema_version': 2, 'checkpoint_at': call['time']+0.01,
             'owner': {**owner, 'step_id': step}, 'memory': memory.to_dict()}
    compact_checkpoint_evidence(state, root / 'evidence')
    write_checkpoint(checkpoint, state)
    return {'call_id': call['call_id'], 'checkpoint_path': str(checkpoint),
            'evidence_path': str(root / 'evidence' / (call['call_id']+'.json'))}


def source_fixture(root, project, contract=None):
    root.mkdir(parents=True, exist_ok=True)
    work = root / 'native-md'; work.mkdir()
    input_cif = root / 'input.cif'; input_cif.write_text(cif(10))
    output_cif = work / 'optimized.cif'; output_cif.write_text(cif(11))
    log = work / 'solver.log'; log.write_bytes(native_log())
    args = {'cif_path': str(input_cif), 'output_dir': str(work), 'mode': 'md',
            'temperature': 300, 'production_steps': 20, 'warmup_steps': 0, 'equilibration_steps': 0}
    value = {'success': True, 'work_dir': str(work), 'output_cif': str(output_cif), 'n_procs': 4,
             'energy': 3, 'stage_steps_source': 'solver.log stage-delimited native Loop time summaries',
             'actual_steps': {'warmup': 0, 'equilibration': 0, 'production': 20, 'total': 20},
             'equilibration_verified': False}
    result_json = work / 'md_result.json'; result_json.write_text(json.dumps(value))
    receipt = {'submitted': True, 'job_id': '42', 'work_dir': str(work), 'output_dir': str(work),
               'result_json_path': str(result_json), 'expected_structure_path': str(output_cif)}
    owner = {'username': root.parent.name, 'conv_id': root.name}
    key = RecoveryGate().key('run_md_optimize', args, 1, 'md', str(project))
    source = {'contract': contract or {'step_id': 'md', 'agent': 'adsorption', 'tool': 'run_md_optimize',
        'arguments': args, 'depends_on': [], 'expected_outputs': [str(result_json)]},
        'status': 'succeeded', 'phase': 'finish', 'token': 'source-token', 'attempt_id': 'source-attempt',
        'recovery_key': key, 'job_ids': ['42'], 'jobs_confirmed_terminal': True,
        'effective_arguments': copy.deepcopy(args), 'tool_returned': True, 'result': receipt,
        'artifacts': {str(result_json): artifact_fact(str(result_json), root)}}
    source['result_ref'] = producer_evidence(root, owner, 'md', 'run_md_optimize', args, receipt)
    entry = {'key': key, 'tool': 'run_md_optimize', 'params': args, 'job_ids': ['42'],
             'attempt_id': 'source-attempt', 'fingerprint': input_fingerprint(args, project), 'status': 'completed'}
    (root / 'recovery_state.json').write_text(json.dumps({key: entry}))
    from agents.result_validation import build_validation_dossier
    source['validation_dossier'] = build_validation_dossier(source, source['attempt_id'], root/'evidence')
    return source, owner


@pytest.fixture
def md_case(tmp_path):
    root = tmp_path / 'runs/demo-user/demo-case'
    source, owner = source_fixture(root, tmp_path)
    target = {'contract': {'step_id': 'report', 'agent': 'communicator', 'tool': 'generate_scientific_report',
        'arguments': {'source_steps': ['md'], 'output_path': str(root / 'report.md'), 'title': 'Native MD'},
        'depends_on': ['md']}, 'status': 'running', 'token': 'report-token'}
    state = {**owner, 'nodes': {'md': source, 'report': target}, 'plan_version': 1}
    def attest(node, call, raw):
        files = _OwnedFiles(root, owner)
        try:
            path = node['result_ref']['checkpoint_path']
            fact = checkpoint_call_witness(files, path, call, step_id=node['contract']['step_id'], require_result_hash=True)
            return {'checkpoint_verified': True, 'checkpoint_sha256': fact['sha256'], 'checkpoint_path': path}
        finally: files.close()
    return root, owner, state, attest


def packet(fixture):
    root, owner, state, attest = fixture
    return build_result_packet(root, owner, state, 'report', attest=attest)


def update_json(fixture, edit):
    root, _, state, _ = fixture
    path = Path(state['nodes']['md']['result']['result_json_path'])
    value = json.loads(path.read_text()); edit(value); path.write_text(json.dumps(value))
    state['nodes']['md']['artifacts'][str(path)] = artifact_fact(str(path), root)


def test_real_packet_expands_only_json_native_artifact_into_exact_seven_log_blocks_and_cifs(md_case):
    root, _, state, _ = md_case
    before = {path: path.read_bytes() for path in (root/'input.cif', root/'native-md/optimized.cif', root/'native-md/solver.log', root/'native-md/md_result.json')}
    assert list(state['nodes']['md']['artifacts']) == [str(root/'native-md/md_result.json')]
    data = packet(md_case)
    history = [table for table in data['tables'] if table.get('format') == 'native_thermo']
    assert len(history) == 7
    assert [table['native_context']['native_steps'] for table in history] == [14,26,2,0,0,20,0]
    assert all(table['column_units']['Temp'] == 'K' for table in history)
    geometry = [table for table in data['tables'] if table.get('format') == 'native_cif_geometry']
    assert [table['rows'][0]['a_A'] for table in geometry] == [10,11]
    assert geometry[0]['native_context']['historical_sha_verified'] is True
    assert geometry[1]['native_context']['historical_sha_verified'] is False
    assert data['sources'][0]['conditions']['temperature'] == 300
    assert history[-1]['rows'][0]['Temp'] == 290
    assert sources_unchanged(root, data)
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize('bad', ['json_bytes','log_sha','cif_sha','foreign_log','foreign_output','symlink_log','symlink_cif','changed_input','different_attempt'])
def test_bad_hash_owner_symlink_or_attempt_cannot_supply_native_md_report_facts(md_case, bad):
    root, _, state, _ = md_case
    if bad == 'json_bytes': (root/'native-md/md_result.json').write_text('{}')
    elif bad == 'log_sha': update_json(md_case, lambda value: value.update(thermo_history={'source_log':str(root/'native-md/solver.log'),'source_sha256':'0'*64}))
    elif bad == 'cif_sha': update_json(md_case, lambda value: value.update(output_cif_sha256='0'*64))
    elif bad == 'foreign_log': update_json(md_case, lambda value: value.update(thermo_history={'source_log':str(root.parent/'foreign.log')}))
    elif bad == 'foreign_output': update_json(md_case, lambda value: value.update(output_cif=str(root.parent/'foreign.cif')))
    elif bad in {'symlink_log','symlink_cif'}:
        path = root/('native-md/solver.log' if bad=='symlink_log' else 'native-md/optimized.cif')
        foreign = root.parent/'foreign-data'; foreign.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(foreign)
    elif bad == 'changed_input': (root/'input.cif').write_text(cif(15))
    else: state['nodes']['md']['attempt_id'] = 'different-attempt'
    with pytest.raises((ValueError, PermissionError, OSError)): packet(md_case)


def test_missing_declared_log_is_disclosed_and_never_replaced_by_configured_steps_or_temperature(md_case):
    root, _, _, _ = md_case
    (root/'native-md/solver.log').unlink()
    data = packet(md_case)
    assert not any(table.get('format')=='native_thermo' for table in data['tables'])
    assert any('unavailable' in text for text in data['sources'][0]['comparison_limits'])
    assert data['sources'][0]['conditions']['temperature'] == 300


def test_packet_byte_guards_cover_added_log_and_input_output_cif(md_case):
    root, _, _, _ = md_case
    data = packet(md_case)
    assert sources_unchanged(root, data)
    (root/'native-md/optimized.cif').write_text(cif(18))
    assert not sources_unchanged(root, data)


def test_projected_dossier_can_omit_display_inventory_but_not_native_facts(md_case):
    source = md_case[2]['nodes']['md']
    source['validation_dossier']['inventory'].pop('files')
    source['validation_dossier'].pop('verification_owner')
    data = packet(md_case)
    assert len([table for table in data['tables'] if table.get('format') == 'native_thermo']) == 7


@pytest.mark.parametrize('field', ['attempt_id', 'job_ids', 'contract', 'method_facts'])
def test_projected_native_dossier_still_rejects_identity_or_observation_tampering(md_case, field):
    dossier = md_case[2]['nodes']['md']['validation_dossier']
    if field == 'attempt_id': dossier[field] = 'different-attempt'
    elif field == 'job_ids': dossier[field] = ['43']
    elif field == 'contract': dossier[field] = {**dossier[field], 'agent': 'foreign-agent'}
    else:
        dossier[field] = copy.deepcopy(dossier[field])
        dossier[field]['md_native']['artifacts']['solver.log']['sha256'] = '0'*64
    with pytest.raises(ValueError): packet(md_case)


def test_no_declared_or_native_inspected_log_does_not_guess_neighbor_filename(md_case, monkeypatch):
    root, _, state, _ = md_case
    update_json(md_case, lambda value: value.update(stage_steps_source='unverified until native summaries are collected'))
    state['nodes']['md'].pop('validation_dossier')
    import agents.report_authoring as module
    original = module._read
    def reject_unannounced_log(owner_root, path, budget):
        assert Path(path).name != 'solver.log', 'undeclared neighboring log was guessed'
        return original(owner_root, path, budget)
    monkeypatch.setattr(module, '_read', reject_unannounced_log)
    data = packet(md_case)
    assert not any(table.get('format') == 'native_thermo' for table in data['tables'])
    assert any('does not declare' in value for value in data['sources'][0]['comparison_limits'])


def test_native_json_only_md_to_real_os_plot_to_native_report_with_verified_image(factory, tmp_path, monkeypatch):
    from tests.test_runtime_authoring_visual import advance
    cid = 'md-closure'
    root = tmp_path/'runs/tester'/cid
    work = root/'native-md'
    args = {'cif_path':str(root/'input.cif'), 'output_dir':str(work), 'mode':'md',
            'temperature':300, 'production_steps':20, 'warmup_steps':0, 'equilibration_steps':0}
    md = node('md', tool='run_md_optimize', args=args, outputs=[str(work/'md_result.json')])
    md['agent']='adsorption'
    figure = node('figure', tool='generate_result_figures', args={'source_steps':['md'],
        'title':'Actual native MD', 'instructions':'Plot original native temperature and CIF cell lengths with declared units.',
        'output_dir':str(root/'figures')}, deps=['md'], outputs=[str(root/'figures/manifest.json')])
    report = node('report', tool='generate_scientific_report', args={'source_steps':['md','figure'],
        'title':'Native MD engineering test', 'instructions':'Use all native evidence and disclose unknown convergence.',
        'output_path':str(root/'report.md'), 'report_mode':'fragment', 'report_requirements':{'minimum_figures':1, 'independent_review': True}},
        deps=['md','figure'], outputs=[str(root/'report.md')]); report['agent']='communicator'
    rt = factory([md,figure,report], cid=cid)
    source, owner = source_fixture(root, tmp_path, contract=md)
    from agents import registry, workspace
    monkeypatch.setattr(registry,'get_config',lambda:rt.main.config)
    monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
    calls=[]
    def provider(self, role):
        content=self.messages[-1]['content'];request=json.loads(content[0]['text'] if isinstance(content,list) else content)
        phase=request['stage'];payload=request['request'];calls.append(phase)
        assert not role.functions and not role.handoff_to
        if phase=='plot':
            choices=payload['available_inputs']; log=next(item for item in choices if item.get('native_structure'))
            assert len(log['native_structure']['summary']['blocks'])==7
            cifs=[item for item in choices if item.get('format_hints',{}).get('native_format')=='CIF']
            assert len(cifs)==2
            script='''import json,re,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with open(%r) as stream: history=json.load(stream)
assert len(history['blocks'])==7
assert [b['native_steps'] for b in history['blocks']]==[14,26,2,0,0,20,0]
production=next(b for b in history['blocks'] if b['stage']=='production')
table=production['tables'][0];step=table['headers'].index('Step');temp=table['headers'].index('Temp')
assert table['column_units']['Temp']=='K'
cells=[]
for path in %r:
 with open(path) as stream: text=stream.read()
 cells.append(float(re.search(r'^_cell_length_a\\s+(\\S+)',text,re.M)[1]))
fig,axes=plt.subplots(1,2,figsize=(7,3))
axes[0].plot([r['values'][step] for r in table['rows']],[r['values'][temp] for r in table['rows']],marker='o')
axes[0].set(xlabel='Native step',ylabel='Temperature (K)')
axes[1].bar(['Input','Output'],cells);axes[1].set(ylabel='Cell length a (angstrom; CIF)')
fig.tight_layout();fig.savefig('/output/native-md.png')
''' % (log['native_structure']['sandbox_path'],[item['sandbox_path'] for item in cifs])
            body={'plot':{'script':script,'input_ids':[log['id'],*[item['id'] for item in cifs]],
                'rationale':'Read actual native blocks and configurations.','limitations':['Convergence unknown.']}}
        elif phase=='visual':
            assert isinstance(content,list) and any(item.get('type')=='image' for item in content)
            body={'visual':{'supported':True,'verified':True,'issues':[], 'limitations':['Controlled engineering image transport fixture.']}}
        elif phase=='author':
            data=payload['packet']; assert len([table for table in data['tables']if table.get('format')=='native_thermo'])==7
            blocks=[{'kind':'narrative','claim_type':'method','source_ids':[item['id']for item in data['sources']],
                'text':'The native observations retain their source units. Configured targets are not measured temperature or integration steps.'}]
            blocks += [{'kind':'table','table_id':table['id']}for table in data['tables']]
            blocks += [{'kind':'figure','figure_id':item['id']}for item in data['figures']]
            blocks += [{'kind':'narrative','claim_type':'limitation','source_ids':[item['id']for item in data['sources']],
                'text':'Scientific convergence remains unknown. Output geometry is a current native-declared configuration, not a time average.'}]
            body={'draft':{'blocks':blocks}}
        elif phase=='review':
            body={'review':{'passed':True,'draft_sha256':payload['draft_sha256'],
                'checked_blocks':list(range(len(payload['draft']['blocks']))),'blocking_issues':[],'notes':[]}}
        else: raise AssertionError('Unexpected native report phase '+phase)
        return SimpleNamespace(id='controlled-md-'+phase,content=[TextBlock(type='text',text=json.dumps(body))])
    monkeypatch.setattr(Session,'_call_api',provider)
    rt.start(1)
    ticket=rt.store.claim(rt.workflow_id,'md')
    source.update(token=ticket['token'])
    rt.store.update(rt.workflow_id,'md',ticket['token'],source,release=True)
    before={path:path.read_bytes()for path in [root/'input.cif',work/'optimized.cif',work/'solver.log',work/'md_result.json']}
    finished=advance(rt,'report',timeout=60)
    assert finished['status']=='succeeded',{'error':finished.get('error'),'result':finished.get('result')}
    image=rt.snapshot()['nodes']['figure']['result']
    assert image['visual_review']['verified'] is True and image['scientific_acceptance'] is False
    assert image['visual_review']['image_receipts'][0]['sha256']==image['artifacts'][0]['sha256']
    assert image['sandbox_receipt']['ok'] is True
    assert (root/'figures/attempt-1/script.py').is_file()
    text=(root/'report.md').read_text()
    assert 'native CIF' in text and 'native log' in text and 'independent visual review verified' in text
    assert '| configured\\_input | 10.0 |' in text and '| native\\_output | 11.0 |' in text
    assert 'Historical CIF bytes were not sealed' in text
    assert 'convergence remains unknown' in text
    assert calls==['plot','visual','author','review']
    assert all(path.read_bytes()==raw for path,raw in before.items())
