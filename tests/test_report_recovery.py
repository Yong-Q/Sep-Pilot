"""Synthetic owned files only: recovery proves receipts, never executes tools."""
import copy
import hashlib
import json
from pathlib import Path

import pytest


def build_report_recovery_case(base):
    root=Path(base)/'runs'/'owner'/'conv';(root/'evidence').mkdir(parents=True)
    (root/'reports').mkdir();report=root/'reports'/'fragment.md';report.write_text('# measured report\n')
    def write(path,value):path.write_text(json.dumps(value,sort_keys=True));return hashlib.sha256(path.read_bytes()).hexdigest()
    source_path=root/'evidence'/'source_call.json'
    source_sha=write(source_path,{'call_id':'source_call','tool':'run_pore_analysis','failed':False,'result':{'finite':True}})
    contract={'step_id':'report','agent':'communicator','tool':'generate_scientific_report',
        'arguments':{'source_steps':['source'],'report_mode':'fragment','output_path':str(report),'title':'Measured','instructions':'Interface only.'},
        'depends_on':['source'],'expected_outputs':[str(report)]}
    receipt={'status':'success','report_file':str(report),'output_path':str(report),'output_files':[str(report)],
        'source_steps':['source'],'source_step':'source','source_evidence_call_id':'source_call',
        'source_attempt_fingerprint':'a'*64,'evidence_fingerprint':source_sha,
        'fragment_fingerprint':hashlib.sha256(report.read_bytes()).hexdigest(),'report_preview':report.read_text()}
    call_path=root/'evidence'/'report_call.json'
    write(call_path,{'call_id':'report_call','tool':'generate_scientific_report','agent':'communicator','time':1,
        'params':contract['arguments'],'result':json.dumps(receipt),'failed':False})
    reference={'call_id':'report_call','evidence_path':str(call_path)}
    artifact={str(report):{'size':report.stat().st_size,'mtime_ns':report.stat().st_mtime_ns}}
    history={'status':'succeeded','phase':'finish','token':'historical_token','contract':copy.deepcopy(contract),
        'origin_goal_version':5,'execution_plan_version':6,'started_at':1,'result':receipt,'result_ref':reference,
        'output_baseline':{str(report):None},'artifacts':artifact,'execution_fingerprint':'report-fingerprint'}
    current={'status':'uncertain','token':'unknown_token','owner_pid':123,'dispatch_phase':'entered_tool',
        'contract':copy.deepcopy(contract),'origin_goal_version':5,'execution_plan_version':7,'started_at':2,
        'attempt_history':[history],'job_ids':[]}
    goal={'version':5,'approved_plan_version':7,'execution_authorized':True,'execution_mode':'workflow',
        'gases':['CO2'],'parameters':{'temperature_K':298},'approved_nodes':[copy.deepcopy(contract)]}
    source={'status':'succeeded','token':'source_token','execution_fingerprint':'a'*64,
        'result_ref':{'call_id':'source_call','evidence_path':str(source_path)}}
    state={'username':'owner','conv_id':'conv','plan_version':7,'goal_contract':goal,'nodes':{'report':current,'source':source}}
    return {'root':root,'state':state,'report':report,'history':history,'call_path':call_path,'source_path':source_path}


def test_archived_report_proof_is_deterministic_readonly_and_keeps_old_baseline(tmp_path):
    from agents.report_recovery import verify_archived_report_fragment
    case=build_report_recovery_case(tmp_path);before=copy.deepcopy(case['state'])
    files={str(p):p.read_bytes() for p in case['root'].rglob('*') if p.is_file()}
    proof=verify_archived_report_fragment(case['state'],'report',case['root'])
    assert proof==verify_archived_report_fragment(case['state'],'report',case['root'])
    assert proof['kind']=='archived_verified_report_fragment'
    assert proof['current_token']=='unknown_token' and proof['historical_token']=='historical_token'
    assert proof['goal_version']==5 and proof['plan_version']==7
    assert proof['result_ref']['call_id']=='report_call' and proof['started_at']==1
    assert proof['output_baseline']==case['history']['output_baseline']
    assert proof['artifacts']==case['history']['artifacts']
    assert proof['no_resubmission'] is True and proof['no_output_write'] is True
    assert len(proof['proof_sha256'])==64 and 'report_preview' not in proof['result']
    assert not any(key in proof for key in ('may_finish','retry_allowed','execution_authorized'))
    assert case['state']==before
    assert files=={str(p):p.read_bytes() for p in case['root'].rglob('*') if p.is_file()}


@pytest.mark.parametrize('change',['source_pending','source_attempt','source_call','source_file','report_bytes','history_contract',
    'current_contract','call_params','call_failed','call_result','wrong_goal','wrong_plan','not_fragment','no_success','artifact_size'])
def test_archived_report_rejects_incomplete_stale_or_tampered_proof(tmp_path,change):
    from agents.report_recovery import verify_archived_report_fragment
    c=build_report_recovery_case(tmp_path);s=c['state'];n=s['nodes']['report'];source=s['nodes']['source']
    if change=='source_pending':source['status']='pending'
    if change=='source_attempt':source['execution_fingerprint']='b'*64
    if change=='source_call':source['result_ref']['call_id']='new_call'
    if change=='source_file':c['source_path'].write_text('{}')
    if change=='report_bytes':c['report'].write_text('# changed report!\n')
    if change=='history_contract':c['history']['contract']['arguments']['title']='Different'
    if change=='current_contract':n['contract']['arguments']['title']='Different'
    if change.startswith('call_'):
        call=json.loads(c['call_path'].read_text())
        if change=='call_params':call['params']['title']='Different'
        if change=='call_failed':call['failed']=True
        if change=='call_result':call['result']='{"status":"success"}'
        c['call_path'].write_text(json.dumps(call))
    if change=='wrong_goal':s['goal_contract']['execution_authorized']=False
    if change=='wrong_plan':s['plan_version']=8
    if change=='not_fragment':n['contract']['arguments']['report_mode']='assembly'
    if change=='no_success':c['history']['status']='pending'
    if change=='artifact_size':c['history']['artifacts'][str(c['report'])]['size']+=1
    with pytest.raises((ValueError,PermissionError)):
        verify_archived_report_fragment(s,'report',c['root'])


@pytest.mark.parametrize('target',['report','source_path','call_path','parent'])
def test_archived_report_rejects_symlink_components(tmp_path,target):
    from agents.report_recovery import verify_archived_report_fragment
    c=build_report_recovery_case(tmp_path)
    if target=='parent':
        old=c['root']/'reports';moved=tmp_path/'outside_reports';old.rename(moved);old.symlink_to(moved,target_is_directory=True)
    else:
        path=c[target];moved=tmp_path/('outside_'+path.name);path.rename(moved);path.symlink_to(moved)
    with pytest.raises((ValueError,PermissionError,OSError)):
        verify_archived_report_fragment(c['state'],'report',c['root'])


def test_proof_hash_changes_when_source_token_changes(tmp_path):
    from agents.report_recovery import verify_archived_report_fragment
    c=build_report_recovery_case(tmp_path);a=verify_archived_report_fragment(c['state'],'report',c['root'])
    c['state']['nodes']['source']['token']='reconciled_source_token'
    b=verify_archived_report_fragment(c['state'],'report',c['root'])
    assert a['proof_sha256']!=b['proof_sha256']


def build_returned_readonly_case(base):
    c=build_report_recovery_case(base);s=c['state'];contract={'step_id':'qc','agent':'analyst','tool':'query_literature',
        'arguments':{'query':'bounded fixture'},'depends_on':[],'expected_outputs':[]}
    result={'ok':True,'evidence_complete':True,'results':[{'title':'Synthetic fixture'}]}
    path=c['root']/'evidence'/'readonly_call.json';path.write_text(json.dumps({'call_id':'readonly_call','tool':'query_literature',
        'params':contract['arguments'],'result':json.dumps(result),'failed':False,'time':3}))
    s['nodes']={'qc':{'contract':contract,'status':'pending','token':'returned_token','origin_goal_version':5,'execution_plan_version':7,
        'tool_returned':True,'result':result,'result_ref':{'call_id':'readonly_call','evidence_path':str(path)},'job_ids':[],
        'started_at':2,'output_baseline':{},'artifacts':{}}}
    s['goal_contract']['approved_nodes']=[copy.deepcopy(contract)]
    return c


def test_returned_readonly_uses_current_exact_evidence_not_a_new_execution(tmp_path):
    from agents.report_recovery import verify_returned_readonly
    c=build_returned_readonly_case(tmp_path);before=copy.deepcopy(c['state'])
    proof=verify_returned_readonly(c['state'],'qc',c['root'])
    assert proof['kind']=='current_returned_evidence' and proof['current_token']=='returned_token'
    assert proof['result']==before['nodes']['qc']['result']
    assert proof['result_ref']['call_id']=='readonly_call' and proof['started_at']==2
    assert c['state']==before and proof['no_resubmission'] and proof['no_output_write']


@pytest.mark.parametrize('change',['not_returned','different_result','different_call','different_params','mutator','foreign_scope','goal_changed'])
def test_returned_readonly_rejects_unproven_or_nonreadonly_state(tmp_path,change):
    from agents.report_recovery import verify_returned_readonly
    c=build_returned_readonly_case(tmp_path);s=c['state'];n=s['nodes']['qc'];goal=None
    if change=='not_returned':n['tool_returned']=False
    if change=='different_result':n['result']['ok']=False
    if change=='different_call':n['result_ref']['call_id']='unknown'
    if change=='different_params':n['contract']['arguments']['query']='changed'
    if change=='mutator':n['contract']['tool']='run_string_tst';s['goal_contract']['approved_nodes']=[copy.deepcopy(n['contract'])]
    if change=='foreign_scope':s['username']='other'
    if change=='goal_changed':goal=copy.deepcopy(s['goal_contract']);goal['version']+=1
    with pytest.raises((ValueError,PermissionError)):
        verify_returned_readonly(s,'qc',c['root'],goal)


@pytest.mark.parametrize('field',['origin_goal_version','execution_plan_version'])
def test_archived_report_requires_current_dispatch_versions(tmp_path,field):
    from agents.report_recovery import verify_archived_report_fragment
    c=build_report_recovery_case(tmp_path);c['state']['nodes']['report'][field]-=1
    with pytest.raises(ValueError):verify_archived_report_fragment(c['state'],'report',c['root'])


def test_recovery_rejects_duplicate_approved_node_identity(tmp_path):
    from agents.report_recovery import verify_archived_report_fragment
    c=build_report_recovery_case(tmp_path);goal=c['state']['goal_contract']
    goal['approved_nodes'].append(copy.deepcopy(goal['approved_nodes'][0]))
    with pytest.raises(ValueError):verify_archived_report_fragment(c['state'],'report',c['root'])


@pytest.mark.parametrize('target',['report','source_path','call_path'])
def test_fifo_evidence_is_rejected_without_waiting_for_a_writer(tmp_path,target):
    import os
    import threading
    from agents.report_recovery import verify_archived_report_fragment
    c=build_report_recovery_case(tmp_path);path=c[target];path.rename(path.with_name(path.name+'.original'));os.mkfifo(path)
    errors=[]
    def check():
        try:verify_archived_report_fragment(c['state'],'report',c['root'])
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=check,daemon=True);thread.start();thread.join(.15)
    blocked=thread.is_alive()
    if blocked:
        # Unblock only this fixture's FIFO, so a RED test leaves no reader.
        fd=os.open(path,os.O_WRONLY|os.O_NONBLOCK);os.close(fd);thread.join(1)
    assert not thread.is_alive()
    assert not blocked,'recovery reader waited for a FIFO writer before fstat'
    assert errors


def test_oversized_regular_evidence_is_rejected_before_any_read(tmp_path,monkeypatch):
    import agents.report_recovery as recovery
    c=build_report_recovery_case(tmp_path)
    with c['report'].open('r+b') as stream:stream.truncate(64*1024*1024+1)
    reads=[];original=recovery.os.read
    def observed(*args):reads.append(1);return original(*args)
    monkeypatch.setattr(recovery.os,'read',observed)
    with pytest.raises(ValueError):recovery.verify_archived_report_fragment(c['state'],'report',c['root'])
    assert reads==[],'oversized evidence was hashed instead of rejected at fstat'
