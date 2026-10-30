"""Only native-approved report deltas may retain identical accepted presentation."""
import copy
import pytest
from agents.parallel_workflow import _approval_digest
from agents.presentation_transition import report_only_history_context


def fixture():
    science={'step_id':'science','agent':'adsorption','tool':'run_string_tst','arguments':{'gas':'CH4'},'depends_on':[],'expected_outputs':['native.json']}
    figure={'step_id':'figure','agent':'analyst','tool':'generate_result_figures','arguments':{'source_steps':['science'],'output_dir':'figs','instructions':'original figure'},'depends_on':['science'],'expected_outputs':['figs/manifest.json']}
    report={'step_id':'report','agent':'communicator','tool':'generate_scientific_report','arguments':{'source_steps':['figure'],'output_path':'report.md','instructions':'original report'},'depends_on':['figure'],'expected_outputs':['report.md']}
    old={'version':2,'method':'TST','gases':['CH4'],'parameters':{'temperature':298},'execution_mode':'workflow','execution_authorized':True,'approved_plan_version':6,'approved_nodes':[science,figure,report],'active_goal':'original goal','directives':['original goal']}
    state={'username':'owner','conv_id':'conv','plan_version':6,'goal_contract':copy.deepcopy(old),'nodes':{n['step_id']:{'contract':copy.deepcopy(n),'status':'succeeded' if n['step_id']!='report' else 'failed','phase':'finish' if n['step_id']!='report' else None,'token':n['step_id']+'token'}for n in [science,figure,report]}}
    goal=copy.deepcopy(old);goal['approved_plan_version']=7;goal['approved_nodes'][-1]['arguments']['instructions']='Correctly bind actual facts.'
    line={'username':'owner','conv_id':'conv','plan_version':7,'plan_history':[{'version':6}]}
    approval={'schema_version':1,'kind':'main_model_applied_plan','owner':{'username':'owner','conv_id':'conv'},'goal_version':2,'plan_version':7,'base_plan_version':6,'approval_sha256':_approval_digest(goal),'control_source':{'origin':'user','owner':{'username':'owner','conv_id':'conv'},'turn_id':'real-owned-user-turn','goal_version':2}}
    return state,goal,line,approval


def test_report_only_approval_retains_exact_figure_without_granting_execution():
    state,goal,line,approval=fixture();before=copy.deepcopy((state,goal,line,approval))
    context=report_only_history_context(state,goal,line,approval,owner={'username':'owner','conv_id':'conv'})
    assert context['kind']=='approved_report_only_history' and context['retained_steps']==['figure']
    assert context['changed_report_steps']==['report'] and context['no_execution_authority']
    assert context['from_plan_version']==6 and context['to_plan_version']==7
    assert (state,goal,line,approval)==before


@pytest.mark.parametrize('change',['science','science_removed','new_science','figure_arguments','figure_outputs','figure_actor','new_user_version','new_user_goal','new_user_directive','foreign_approval','stale_base','unsigned','paused','ambiguous_ids','old_approval_changed','metadata_authority_kind'])
def test_report_only_reuse_never_ignores_science_render_intent_or_user_authority(change):
    state,goal,line,approval=fixture()
    if change=='science':goal['approved_nodes'][0]['arguments']['gas']='CO2'
    elif change=='science_removed':goal['approved_nodes'].pop(0)
    elif change=='new_science':goal['approved_nodes'].append({'step_id':'new-science','tool':'run_md_optimize'})
    elif change=='figure_arguments':goal['approved_nodes'][1]['arguments']['instructions']='Explicitly redraw.'
    elif change=='figure_outputs':goal['approved_nodes'][1]['expected_outputs']=['new/manifest.json']
    elif change=='figure_actor':goal['approved_nodes'][1]['agent']='communicator'
    elif change=='new_user_version':goal['version']=3;approval['goal_version']=3
    elif change=='new_user_goal':goal['active_goal']='Please redraw the figure.'
    elif change=='new_user_directive':goal['directives'].append('Please redraw.')
    elif change=='foreign_approval':approval['control_source']['owner']['username']='other'
    elif change=='stale_base':approval['base_plan_version']=5
    elif change=='unsigned':approval={}
    elif change=='paused':state['user_paused']=True
    elif change=='ambiguous_ids':goal['approved_nodes'].append(copy.deepcopy(goal['approved_nodes'][1]))
    elif change=='old_approval_changed':state['goal_contract']['approved_nodes'][0]['arguments']['gas']='CO2'
    elif change=='metadata_authority_kind':approval['kind']='native_metadata_continuation'
    if approval:approval['approval_sha256']=_approval_digest(goal)
    assert report_only_history_context(state,goal,line,approval,owner={'username':'owner','conv_id':'conv'}) is None
