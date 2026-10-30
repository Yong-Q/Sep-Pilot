import copy
import pytest
from agents.parallel_workflow import _approval_digest
from agents.preparation_transition import preparation_transition


def fixture():
 prep={'step_id':'prep','agent':'analyst','tool':'run_cdft','arguments':{'action':'inputs','cif_path':'x.cif','temperature':298},'depends_on':[],'expected_outputs':['x.dat']}
 calc={'step_id':'calc','agent':'analyst','tool':'run_cdft','arguments':{'action':'submit'},'depends_on':['prep'],'expected_outputs':['result']}
 fig={'step_id':'fig','agent':'analyst','tool':'generate_result_figures','arguments':{'output_dir':'fig'},'depends_on':['calc']}
 old={'version':2,'method':'CDFT','gases':['CH4'],'parameters':{'temperature_K':298},'execution_mode':'workflow','execution_authorized':True,'original_goal':'scientific task','approved_plan_version':5,'approved_nodes':[prep,calc,fig]}
 state={'username':'u','conv_id':'c','plan_version':5,'goal_contract':copy.deepcopy(old),'nodes':{n['step_id']:{'contract':copy.deepcopy(n),'status':'succeeded','phase':'finish','token':'tok'} for n in [prep,calc,fig]}}
 goal=copy.deepcopy(old);goal['version']=3;goal['approved_plan_version']=7;goal['approved_nodes'][2]['arguments']['output_dir']='newfig'
 line={'username':'u','conv_id':'c','plan_version':7,'plan_history':[{'version':5}]}
 approval={'schema_version':1,'kind':'main_model_applied_plan','owner':{'username':'u','conv_id':'c'},'goal_version':3,'plan_version':7,'base_plan_version':5,'approval_sha256':_approval_digest(goal),'control_source':{'origin':'user','owner':{'username':'u','conv_id':'c'},'turn_id':'turn'}}
 return state,goal,line,approval

def context(items,step='prep'):
 return preparation_transition(*items,step,owner={'username':'u','conv_id':'c'})

def test_unrelated_figure_change_retains_exact_preparation():
 items=fixture();before=copy.deepcopy(items);proof=context(items)
 assert proof['kind']=='approved_local_preparation_history' and proof['no_execution_authority']
 assert items==before

@pytest.mark.parametrize('mutation',['other_owner','pause','goal_temperature','goal_gas','goal_original','unapproved','old_contract','new_contract','new_dependency','unfinished','stale_base','missing_token','unsupported','duplicate'])
def test_invalid_scope_never_carries(mutation):
 s,g,l,a=items=fixture()
 if mutation=='other_owner':s['username']='other'
 if mutation=='pause':s['user_paused']=True
 if mutation=='goal_temperature':g['parameters']['temperature_K']=308
 if mutation=='goal_gas':g['gases']=['CO2']
 if mutation=='goal_original':g['original_goal']='other'
 if mutation=='unapproved':a['control_source']['origin']='model'
 if mutation=='old_contract':s['nodes']['prep']['contract']['arguments']['temperature']=300
 if mutation=='new_contract':g['approved_nodes'][0]['arguments']['temperature']=300
 if mutation=='new_dependency':g['approved_nodes'][0]['depends_on']=['calc']
 if mutation=='unfinished':s['nodes']['prep']['status']='prefinish'
 if mutation=='stale_base':a['base_plan_version']=4
 if mutation=='missing_token':s['nodes']['prep']['token']=None
 if mutation=='unsupported':
  s['nodes']['prep']['contract']['tool']='run_bash';s['goal_contract']['approved_nodes'][0]['tool']='run_bash';g['approved_nodes'][0]['tool']='run_bash'
 if mutation=='duplicate':g['approved_nodes'].append(copy.deepcopy(g['approved_nodes'][0]))
 a['approval_sha256']=_approval_digest(g)
 assert context(items) is None


def test_claim_waits_for_unfinished_indirect_ancestor(tmp_path):
    from agents.parallel_workflow import WorkflowStore
    from agents.state_io import write_checkpoint
    store=WorkflowStore(tmp_path/'workflows.json');wid='owned'
    def node(status, deps):
        return {'status':status,'contract':{'depends_on':deps,'resources':[]}}
    nodes={'prep':node('running',[]),'science':node('succeeded',['prep']),
           'report':node('pending',['science']),'independent':node('pending',[])}
    workflow={'status':'active','username':'u','conv_id':'c','plan_version':1,'goal_contract':{'version':3},'nodes':nodes}
    write_checkpoint(store.path,{'workflows':{wid:workflow},'leases':{}})
    assert store.claim(wid,'report') is None
    assert any('prep' in x for x in store.claim_blockers(wid,'report'))
    assert store.claim(wid,'independent') is not None
    from agents.state_io import json_transaction
    with json_transaction(store.path) as data:data['workflows'][wid]['nodes']['prep']['status']='succeeded'
    assert store.claim(wid,'report') is not None

@pytest.mark.parametrize('deps', [['missing'], ['report']])
def test_claim_fails_closed_for_missing_or_cyclic_ancestry(tmp_path,deps):
    from agents.parallel_workflow import WorkflowStore
    from agents.state_io import write_checkpoint
    store=WorkflowStore(tmp_path/'workflows.json');wid='owned'
    node={'status':'pending','contract':{'depends_on':deps,'resources':[]}}
    write_checkpoint(store.path,{'workflows':{wid:{'status':'active','goal_contract':{'version':1},'nodes':{'report':node}}},'leases':{}})
    assert store.claim(wid,'report') is None


def self_recorded_fixture():
    import hashlib,json
    fields=('status','token','updated_at','result_ref','execution_fingerprint','artifacts','node_verification','origin_goal_version','execution_plan_version')
    old={'status':'succeeded','token':'original','updated_at':10.0,'result_ref':{'call_id':'original'},'execution_fingerprint':'fingerprint','artifacts':{'x':'hash'},'node_verification':{'ok':True},'origin_goal_version':1,'execution_plan_version':1}
    current=copy.deepcopy(old);current.update(updated_at=25.0,reuse_basis={'kind':'verified_completed_submission','no_resubmission':True,'original_execution_fingerprint':'fingerprint','checked_at':24.0})
    digest=hashlib.sha256(json.dumps({'science':{f:old.get(f) for f in fields}},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    state={'nodes':{'science':current},'plan_history':[{'nodes':{'science':old}}]}
    return state,{'scientific_source_steps':['science'],'source_receipts_sha256':digest},{'approved_at':20.0}


def test_only_own_postapproval_reuse_timestamp_can_match_original_receipt():
    from agents.presentation_transition import _self_recorded_reuse_receipt
    state,receipt,approval=self_recorded_fixture();before=copy.deepcopy(state)
    assert _self_recorded_reuse_receipt(state,receipt,approval)
    assert state==before

@pytest.mark.parametrize('mutation',['token','result','artifacts','verification','fingerprint','late_old','early_reuse','wrong_kind','resubmission','missing_archive','digest'])
def test_reuse_timestamp_proof_rejects_source_or_authority_change(mutation):
    from agents.presentation_transition import _self_recorded_reuse_receipt
    state,receipt,approval=self_recorded_fixture();n=state['nodes']['science']
    if mutation=='token':n['token']='other'
    if mutation=='result':n['result_ref']={'call_id':'other'}
    if mutation=='artifacts':n['artifacts']={'x':'other'}
    if mutation=='verification':n['node_verification']={'ok':False}
    if mutation=='fingerprint':n['execution_fingerprint']='other'
    if mutation=='late_old':state['plan_history'][0]['nodes']['science']['updated_at']=21
    if mutation=='early_reuse':n['reuse_basis']['checked_at']=19
    if mutation=='wrong_kind':n['reuse_basis']['kind']='model_asserted'
    if mutation=='resubmission':n['reuse_basis']['no_resubmission']=False
    if mutation=='missing_archive':state['plan_history']=[]
    if mutation=='digest':receipt['source_receipts_sha256']='0'*64
    assert not _self_recorded_reuse_receipt(state,receipt,approval)
