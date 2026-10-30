import copy,pytest
from agents.parallel_workflow import _approval_digest
from agents.presentation_transition import report_only_history_context
from tests.test_report_noop_rebase import rebased

def setup():
 state,goal,line,approval=rebased();state['pause_kind']='failure';state['pause_reason']='failed node requires diagnosis and plan review';return state,goal,line,approval

def test_owned_failed_rebase_retains_presentation_only_with_proof_still_required():
 state,goal,line,approval=setup();before=copy.deepcopy((state,goal,line,approval))
 result=report_only_history_context(state,goal,line,approval,owner={'username':'owner','conv_id':'conv'})
 assert result['retained_steps']==['figure'] and result['changed_report_steps']==[]
 assert result['no_execution_authority'] and result['no_scientific_change'] and before==(state,goal,line,approval)

def pending_report_edit():
 state,goal,line,approval=setup();step=next(x for x in line['plan_history'][1]['steps'] if x['step_id']=='report')
 step['arguments']['instructions']='unentered edit';step.update(status='pending',done=False,job_ids=[],output_files=[])
 return state,goal,line,approval,step

def test_intermediate_never_started_report_edit_can_revert_without_regenerating_accepted_figures():
 state,goal,line,approval,step=pending_report_edit()
 result=report_only_history_context(state,goal,line,approval,owner={'username':'owner','conv_id':'conv'})
 assert result['retained_steps']==['figure'] and result['no_execution_authority']

@pytest.mark.parametrize('change',['succeeded','entered','job','receipt','output','execution_plan','dependency','figure_edit'])
def test_intermediate_report_edit_requires_strict_unexecuted_proof(change):
 state,goal,line,approval,step=pending_report_edit()
 if change=='succeeded':state['nodes']['report']['status']='succeeded'
 elif change=='entered':step['status']='running'
 elif change=='job':step['job_ids']=['123']
 elif change=='receipt':step['result_ref']={'call_id':'actual'}
 elif change=='output':step['output_files']=['report.md']
 elif change=='execution_plan':state['nodes']['report']['attempt_history']=[{'execution_plan_version':7}]
 elif change=='dependency':step['depends_on']=['science']
 elif change=='figure_edit':next(x for x in line['plan_history'][1]['steps'] if x['step_id']=='figure')['arguments']['instructions']='changed'
 assert report_only_history_context(state,goal,line,approval,owner={'username':'owner','conv_id':'conv'}) is None

@pytest.mark.parametrize('kind',['different_failure','wrong_pause','history','owner','unsigned','change','new_file','stop','unapproved_node'])
def test_failed_rebase_rejects_unproved_authority_or_changed_scope(kind):
 state,goal,line,approval=setup()
 if kind=='different_failure':state['pause_reason']='different failure'
 elif kind=='wrong_pause':state['pause_kind']='user_change'
 elif kind=='history':line['plan_history'][1]['steps'][0]['arguments']['gas']='CO2'
 elif kind=='owner':approval['owner']['username']='other'
 elif kind=='unsigned':approval['approval_sha256']='bad'
 elif kind=='change':goal['approved_nodes'][0]['arguments']['gas']='CO2'
 elif kind=='new_file':goal['directives'][-1]='继续原任务，新增 other.csv。'
 elif kind=='stop':goal['directives'][-1]='不要继续原任务。'
 elif kind=='unapproved_node':state['nodes']['figure']['contract']['arguments']['instructions']='different'
 if kind!='unsigned':approval['approval_sha256']=_approval_digest(goal)
 assert report_only_history_context(state,goal,line,approval,owner={'username':'owner','conv_id':'conv'}) is None
