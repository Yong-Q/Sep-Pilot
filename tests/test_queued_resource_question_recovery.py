"""Native same-job resource proposals release only obsolete infrastructure questions."""
import copy,json,time
from pathlib import Path
from types import SimpleNamespace
import pytest
from agents.lifecycle import obsolete_internal_question_for_recovery


def queued_case(tmp_path):
    now=time.time();root=tmp_path/'runs/owner/conv';(root/'evidence').mkdir(parents=True)
    contracts=[{'step_id':step,'agent':'adsorption','tool':'run_cdft','arguments':{'action':'submit','gas':gas,'memory_mb':16384},'expected_outputs':[]} for step,gas in (('n2','N2'),('co2','CO2'))]
    goal={'version':6,'method':'CDFT','gases':['N2','CO2'],'parameters':{'temperature_K':298},'execution_mode':'workflow','execution_authorized':True,'approved_plan_version':9,'approved_nodes':copy.deepcopy(contracts)}
    saved={**copy.deepcopy(goal),'version':5}
    nodes={c['step_id']:{'contract':c,'status':'waiting_jobs','token':c['step_id']+'token','attempt_id':job+'attempt','recovery_key':job+'key','job_ids':[job]} for c,job in zip(contracts,('123','456'))}
    runtime={'username':'owner','conv_id':'conv','plan_version':9,'goal_contract':saved,'status':'active','user_paused':False,'nodes':nodes}
    jobs=[{'job_id':job,'username':'owner','conv_id':'conv','tool':'run_cdft','attempt_id':job+'attempt','recovery_key':job+'key','state':'PENDING','terminal':False,'failed':False,'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')} for job in ('123','456')]
    target={'nodelist':'node25','partition':'compute','memory_mb':16384,'cpus':1,'num_processes':1,'walltime':'3-00:00:00','observed_at':now}
    assessment={'ok':True,'read_only':True,'job_id':'123','observed_at':now,'execution_started':False,
        'effective_constraints':{'JobState':'PENDING','NumCPUs':'1','ReqTRES':'cpu=1,mem=16G,node=1','TimeLimit':'3-00:00:00'},
        'compatible_targets':[target]}
    call={'call_id':'assessment123','agent':'monitor','tool':'assess_job_resources','failed':False,'time':now,
        'params':{'job_id':'123'},'result':json.dumps(assessment)}
    evidence=root/'evidence/assessment123.json';evidence.write_text(json.dumps(call))
    review={'status':'propose_resource_patch','readonly':True,'not_resubmitted':True,'reviewed_at':now,
        'resource_review_id':'review123','evidence_call_ids':['assessment123'],
        'evidence_calls':[{'call_id':'assessment123','tool':'assess_job_resources','evidence_path':str(evidence)}],
        'suggested_resources':{'nodelist':'node25','partition':'compute','memory_mb':16384,'num_processes':1}}
    event={'event_id':'pending-warning-native','kind':'job_pending_warning','time':now-1,'main_chat':'waiting_user','supervisor':'delivered',
        'supervisor_receipt':{'resource_review':review},'payload':{'job':copy.deepcopy(jobs[0]),'goal_contract':copy.deepcopy(goal),'notification':{'job_id':'123','state':'PENDING'}}}
    checkpoints=[]
    s=SimpleNamespace(_evidence_root=root/'evidence',goal_contract=SimpleNamespace(version=6,to_dict=lambda:copy.deepcopy(goal)),
        _pending_user_interaction={'tool':'user_decision','params':{'related_nodes':['n2','co2']}},_waiting_for_user_input=True,
        _awaiting_plan_approval=False,_pending_workflow_patch=None,_chain_jobs=lambda:copy.deepcopy(jobs),context={},_checkpoint=lambda reason:checkpoints.append(reason))
    decision_params={'question':'Choose a compatible node', 'reason':'native retarget rejected',
        'related_nodes':['n2','co2'],'candidate_job_id':'123','recovery_key':'123retarget'}
    s._pending_user_interaction={'tool':'user_decision','params':{**decision_params,'decision_id':'decision123'}}
    calls=[]
    for jid in ('123','456'):
        calls.append({'call_id':'rejected'+jid,'agent':'lead-orchestrator','tool':'retarget_queued_job',
            'failed':False,'time':now-12,'params':{'job_id':jid,'nodelist':'oldnode','partition':'compute'},
            'result':json.dumps({'blocked':True,'executed':False,'requires_user':True,'reason':'Fresh CPU/RAM/partition facts do not support the requested node; preserve original job'})})
    calls.append({'call_id':'decisioncall','agent':'lead-orchestrator','tool':'request_user_decision',
        'failed':False,'time':now-10,'params':decision_params,
        'result':json.dumps({'ok':True,'status':'needs_user','decision_id':'decision123'})})
    for native in calls:
        path=root/'evidence'/(native['call_id']+'.json');path.write_text(json.dumps(native));native['evidence_path']=str(path)
    s.memory=SimpleNamespace(tool_call_log=calls)
    return s,event,runtime,assessment,evidence,jobs,checkpoints


def test_fresh_same_job_resource_proposal_allows_metadata_goal_version_progress(tmp_path):
    s,e,r,a,p,jobs,checkpoints=queued_case(tmp_path);before=copy.deepcopy(r)
    assert obsolete_internal_question_for_recovery(s,e,r)
    assert s._pending_user_interaction is None and not s._waiting_for_user_input
    assert r==before and jobs[0]['job_id']=='123' and len(checkpoints)==1


@pytest.mark.parametrize('change',['owner','science','approval','pause','stale','ram','walltime','target','other_running','unrelated','wrong_job','old_attempt','old_tool','future_watch'])
def test_resource_question_recovery_rejects_unproven_or_changed_scope(tmp_path,change):
    s,e,r,a,p,jobs,checkpoints=queued_case(tmp_path)
    if change=='owner':jobs[0]['username']='other'
    elif change=='science':r['goal_contract']['parameters']['temperature_K']=300
    elif change=='approval':s._awaiting_plan_approval=True
    elif change=='pause':r['user_paused']=True
    elif change=='stale':a['observed_at']=time.time()-600
    elif change=='ram':e['supervisor_receipt']['resource_review']['suggested_resources']['memory_mb']=8192
    elif change=='walltime':e['supervisor_receipt']['resource_review']['suggested_resources']['walltime']='1:00:00'
    elif change=='target':e['supervisor_receipt']['resource_review']['suggested_resources']['nodelist']='unsupported'
    elif change=='other_running':jobs[1]['state']='RUNNING'
    elif change=='unrelated':s._pending_user_interaction['params']['related_nodes'].append('other')
    elif change=='wrong_job':a['job_id']='999'
    elif change=='old_attempt':jobs[0]['attempt_id']='anotherattempt'
    elif change=='old_tool':jobs[0]['tool']='run_gcmc_batch'
    elif change=='future_watch':jobs[0]['updated_at']='2030-01-01 00:00:00'
    call=json.loads(p.read_text());call['result']=json.dumps(a);p.write_text(json.dumps(call))
    pending=copy.deepcopy(s._pending_user_interaction)
    assert not obsolete_internal_question_for_recovery(s,e,r)
    assert s._pending_user_interaction==pending and s._waiting_for_user_input and not checkpoints


def test_pending_warning_policy_upgrade_rechecks_original_job_once(tmp_path,monkeypatch):
    from agents.job_watch import JobWatch
    from agents import slurm
    watch=JobWatch(str(tmp_path/'watch.json'));watch.register('123',username='owner',conv_id='conv')
    record=watch._jobs['123'];record.update(pending_since_epoch=time.time()-600,pending_warning_at=time.time(),pending_warning_reason='Priority',pending_warning_policy_version=1)
    monkeypatch.setattr(slurm,'batch_fetch_states',lambda ids:{'123':{'status':'PENDING','terminal':False,'failed':False,'pending_reason':'Priority'}})
    watch._poll_once();notifications=watch.drain_notifications()
    assert len(notifications)==1 and notifications[0]['job_id']=='123'
    assert watch.get('123')['pending_warning_policy_version']==2
    watch.ack_notification('123',notifications[0]['notification_id'])
    watch._poll_once();assert watch.drain_notifications()==[]
    assert watch.get('123')['state']=='PENDING'


def test_resource_warning_naturally_reserves_main_after_clearing_old_question(tmp_path,monkeypatch):
    import asyncio,api
    s,e,r,a,p,jobs,checkpoints=queued_case(tmp_path)
    s._runtime_snapshot=lambda:copy.deepcopy(r)
    conv=SimpleNamespace(username='owner',conv_id='conv',session=s,is_processing=False,interrupt_requested=False,
        interrupt_message='',last_progress_at=0,current_uid='old-main',current_status='waiting')
    receipts=[];scheduled=[]
    class Store:
        def recover_claims(self):pass
        def heartbeat(self):pass
        def pending(self):return [copy.deepcopy(e)]
        def receipt(self,*args):receipts.append(args)
        def claim(self,*args):return True
    monkeypatch.setattr(api,'_ensure_workflow_runtime',lambda conv:SimpleNamespace(tick=lambda *args:None))
    monkeypatch.setattr(api._watch,'list',lambda:copy.deepcopy(jobs))
    monkeypatch.setattr(api,'_schedule_lifecycle',lambda coroutine:(scheduled.append(coroutine),coroutine.close()))
    asyncio.run(api._dispatch_conversation_lifecycle('owner',conv,Store()))
    assert s._pending_user_interaction is None and conv.is_processing
    assert conv.current_uid.startswith('lifecycle:') and len(scheduled)==1
    assert any(row[1:3]==('main_chat','retry') for row in receipts)


def test_supervisor_resource_proposal_is_not_a_new_user_approval(tmp_path,monkeypatch):
    import asyncio,api
    s,e,r,a,p,jobs,checkpoints=queued_case(tmp_path);s._current_line_id='conv'
    observed=[];receipts=[]
    class Store:
        def provisional(self,*args):pass
        def receipt(self,*args):receipts.append(args)
        def snapshot(self):return {'events':{e['event_id']:{'main_chat':'waiting_user'}}}
    s._lifecycle_store=Store()
    conv=SimpleNamespace(username='owner',conv_id='conv',session=s,execution_logs=[])
    review=copy.deepcopy(e['supervisor_receipt']['resource_review'])
    def observe(event):
        observed.append(event['payload']['requires_user'])
        return {'next_action':'diagnose_and_fix','reason':'review same original job'}
    monkeypatch.setattr(api,'_review_resources',lambda *args:copy.deepcopy(review))
    monkeypatch.setattr(api,'_ensure_supervisor',lambda conv:SimpleNamespace(observe_lifecycle_event=observe))
    asyncio.run(api._supervise_lifecycle_event(conv,e))
    assert observed==[False]
    delivered=receipts[-1][3]
    assert delivered['resource_review']['status']=='propose_resource_patch'
    assert delivered['not_resubmitted'] is True
    # Consume the exact persisted supervisor shape produced above, not a hand-built receipt.
    e['supervisor_receipt'] = delivered
    e['supervisor'] = 'delivered'
    assert obsolete_internal_question_for_recovery(s, e, r)
    assert s._pending_user_interaction is None


def test_current_approved_execution_graph_is_authority_over_stale_saved_goal_labels(tmp_path):
    s,e,r,*_=queued_case(tmp_path)
    r['goal_contract']['method']='MULTI'
    r['goal_contract']['approved_nodes'][0]['expected_outputs']=[{'kind':'file','path':'old/unbound/results.csv'}]
    assert obsolete_internal_question_for_recovery(s,e,r)


@pytest.mark.parametrize('change',['missing_decision','different_decision','unproven_rejection'])
def test_resource_proposal_preserves_question_without_exact_native_question_origin(tmp_path,change):
    s,e,r,*_=queued_case(tmp_path)
    if change=='missing_decision':s.memory.tool_call_log=[]
    elif change=='different_decision':s._pending_user_interaction['params']['decision_id']='another'
    else:
        path=Path(s.memory.tool_call_log[0]['evidence_path']);native=json.loads(path.read_text());native['result']=json.dumps({'executed':True});path.write_text(json.dumps(native))
    assert not obsolete_internal_question_for_recovery(s,e,r)
    assert s._waiting_for_user_input


@pytest.mark.parametrize('touch_related',[False,True])
def test_unrelated_unapplied_patch_does_not_block_same_queued_job_review(tmp_path,touch_related):
    s,e,r,*_=queued_case(tmp_path)
    goal=s.goal_contract.to_dict()
    other={'step_id':'other','tool':'run_bash','agent':'lead-orchestrator','arguments':{'command':'echo report'},'depends_on':['n2','co2']}
    goal['approved_nodes'].append(copy.deepcopy(other));r['nodes']['other']={'status':'pending','contract':copy.deepcopy(other)}
    s.goal_contract.to_dict=lambda:copy.deepcopy(goal)
    patch={'base_version':9,'new_version':10,'affected_nodes':['other'],
        'changes':[{'operation':'upsert','step_id':'other','node':copy.deepcopy(other)}],
        'proposed_nodes':copy.deepcopy(r['goal_contract']['approved_nodes'])}
    if touch_related:patch['changes'][0]={'operation':'remove','step_id':'n2'}
    s._pending_workflow_patch=patch
    before=copy.deepcopy(patch)
    assert obsolete_internal_question_for_recovery(s,e,r) is (not touch_related)
    assert s._pending_workflow_patch==before
