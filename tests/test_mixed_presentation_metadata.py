import copy
from types import SimpleNamespace
import pytest
from agents.workflow_metadata_alignment import metadata_history_context

def fixture(monkeypatch):
 import agents.parallel_workflow as pw,agents.native_resource_carry as carry
 monkeypatch.setattr(pw,'_valid_execution_approval',lambda *a:True)
 monkeypatch.setattr(carry,'proof_matches',lambda *a,**k:True)
 def c(id,tool,args,deps=()):return {'step_id':id,'agent':'analyst','tool':tool,'arguments':args,'depends_on':list(deps),'expected_outputs':[]}
 prep=c('prep','run_cdft',{'action':'inputs'});calc=c('calc','run_cdft',{'action':'submit','resource_review_id':'old'},['prep']);fig=c('fig','generate_result_figures',{},['calc']);report=c('report','generate_scientific_report',{'instructions':'old'},['fig'])
 nodes={'prep':{'contract':prep,'status':'succeeded','phase':'finish'},'calc':{'contract':calc,'status':'succeeded','phase':'finish','jobs_confirmed_terminal':True},'fig':{'contract':fig,'status':'succeeded','phase':'finish'},'report':{'contract':report,'status':'failed','tool_returned':True,'dispatch_phase':'entered_tool'}}
 old={'version':1,'approved_plan_version':1,'execution_authorized':True,'method':'CDFT','gases':['CH4'],'parameters':{},'execution_mode':'workflow'}
 state={'username':'u','conv_id':'c','plan_version':1,'goal_contract':old,'nodes':nodes};goal={**copy.deepcopy(old),'approved_plan_version':2,'approved_nodes':[copy.deepcopy(x['contract'])for x in nodes.values()]}
 goal['approved_nodes'][1]['arguments']['resource_review_id']='actual';goal['approved_nodes'][3]['arguments']['instructions']='new'
 runtime=SimpleNamespace(username='u',conv_id='c',root='/unused');line={'plan_version':2};return runtime,state,goal,line,{'calc':{'proof':'receipt'}}

def test_failed_report_patch_preserves_attested_science_and_accepted_figure(monkeypatch):
 rt,s,g,line,proofs=fixture(monkeypatch);p=metadata_history_context(rt,s,g,line,{},proofs)
 assert p['prepared_steps']==['prep'] and p['retained_presentations']==['fig'] and p['no_execution_authority']

@pytest.mark.parametrize('tamper',['active_report','unreturned_report','changed_figure','changed_science','foreign','paused','missing_proof','changed_goal'])
def test_incompatible_or_active_changes_are_not_reused(monkeypatch,tamper):
 rt,s,g,line,proofs=fixture(monkeypatch)
 if tamper=='active_report':s['nodes']['report']['status']='running'
 if tamper=='unreturned_report':s['nodes']['report']['tool_returned']=False
 if tamper=='changed_figure':g['approved_nodes'][2]['arguments']['different']='yes'
 if tamper=='changed_science':g['approved_nodes'][0]['arguments']['gas']='N2'
 if tamper=='foreign':s['username']='other'
 if tamper=='paused':s['user_paused']=True
 if tamper=='missing_proof':proofs={}
 if tamper=='changed_goal':g['parameters']={'temperature':308}
 assert metadata_history_context(rt,s,g,line,{},proofs) is None
