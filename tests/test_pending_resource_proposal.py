import copy
import json
import time
from types import SimpleNamespace
import pytest
from agents.resource_review import run_resource_review
from agents.watch_context import set_context, clear_context
from tests.test_job_retarget_feedback import assessment


def run_review(tmp_path, monkeypatch, native=None, decision=None):
    from anthropic.types import ToolUseBlock
    from agents.session import Session
    from agents.config import AgentConfig
    a=assessment();a.update(ok=True,read_only=True,job_id='123',observed_at=time.time())
    a['effective_constraints']['script_has_memory_directive']=True
    a['node_inventory']['policy']['preferred_nodes']=['good']
    a['compatible_targets']=[{'nodelist':'good','partition':'compute','memory_mb':16384,'num_processes':1,'cpus':1,'walltime':'3-00:00:00'}]
    if native: native(a)
    session=Session(config=AgentConfig(api_key='test',project_root=tmp_path))
    monkeypatch.setattr('agents.job_watch.get_watch',lambda:SimpleNamespace(get=lambda _: {'tool':'run_cdft','username':'a','conv_id':'c'}))
    monkeypatch.setattr(session.registry,'execute_dict',lambda *args,**kw:a)
    step=0
    def response(agent):
        nonlocal step
        step+=1
        if step==1:name,args='assess_job_resources',{'job_id':'123'}
        else:
            name='resource_review_decision';args={'status':'needs_user','reason':'pick a node','evidence_call_ids':[c['call_id'] for c in session.memory.tool_call_log if c['tool']=='assess_job_resources']}
            if decision:args.update(decision)
        return SimpleNamespace(content=[ToolUseBlock(type='tool_use',id=str(step),name=name,input=args)])
    session._call_api=response
    set_context('a','c','monitor')
    try:return run_resource_review(session,{'kind':'job_pending_warning','job_id':'123'})
    finally:clear_context()


def test_native_candidates_replace_model_user_pick_without_changing_budget(tmp_path,monkeypatch):
    result=run_review(tmp_path,monkeypatch)
    assert result['status']=='propose_resource_patch'
    assert result['suggested_resources']=={'nodelist':'good','partition':'compute','memory_mb':16384,'num_processes':1,'walltime':'3-00:00:00'}
    assert result['resource_review_id'] and result['reviewed_at']
    assert result['readonly'] and result['not_resubmitted'] and result['requires_user'] is False
    assert result['evidence_call_ids'] and result['evidence_calls'][0]['tool']=='assess_job_resources'


@pytest.mark.parametrize('bad',['stale','foreign_job','failed','running','different_budget','no_targets','planned'])
def test_unproven_assessment_never_turns_into_permission_or_user_pick(tmp_path,monkeypatch,bad):
    def mutate(a):
        if bad=='stale':a['observed_at']=time.time()-121
        if bad=='foreign_job':a['job_id']='other'
        if bad=='failed':a['error']='native assessment failed'
        if bad=='running':a['effective_constraints']['JobState']='RUNNING'
        if bad=='different_budget':a['compatible_targets'][0]['memory_mb']=8192
        if bad=='no_targets':a['compatible_targets']=[]
        if bad=='planned':a['node_inventory']['nodes']['good']['states']=['mix-']
    result=run_review(tmp_path,monkeypatch,native=mutate)
    assert result['status'] in {'waiting_resources','review_retry'}
    assert result.get('requires_user') is not True


def test_model_proposal_without_fresh_native_assessment_stays_waiting(tmp_path,monkeypatch):
    result=run_review(tmp_path,monkeypatch,native=lambda a:a.update(observed_at=time.time()-300),
        decision={'status':'propose_resource_patch','suggested_resources':{'nodelist':'good','partition':'compute','memory_mb':16384,'num_processes':1}})
    assert result['status']=='waiting_resources' and not result['requires_user']


def test_valid_model_target_is_preserved_over_operator_soft_preference(tmp_path,monkeypatch):
    def mutate(a):
        alternate=copy.deepcopy(a['node_inventory']['nodes']['good']);alternate['name']='other'
        a['node_inventory']['nodes']['other']=alternate
        a['compatible_targets'].append({**a['compatible_targets'][0],'nodelist':'other'})
    result=run_review(tmp_path,monkeypatch,native=mutate,
        decision={'suggested_resources':{'nodelist':'other','partition':'compute','memory_mb':16384,'num_processes':1}})
    assert result['status']=='propose_resource_patch' and result['suggested_resources']['nodelist']=='other'
