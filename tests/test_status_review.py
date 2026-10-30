import copy,json
import pytest
from agents.goal_contract import scope_preserving_continuation
from agents.workflow_execution_identity import metadata_only_goal_revision
from agents.parallel_workflow import WorkflowStore
from tests.test_metadata_control_continuation import fixture,old_parser_revision
from tests.test_parallel_workflow import factory,node
from agents.state_io import json_transaction

@pytest.mark.parametrize('text',[
 '请根据当前执行记录复核并更正之前关于任务状态的说明，保留现有结果，不要重新计算。',
 '核对运行状态并澄清之前的说明，保留已有数据。',
 '请根据现有记录检查任务状态并更正此前表述。',
])
def test_status_review_preserves_original_approved_goal_and_recovers_legacy_revision(tmp_path,text):
 root,goal,state=fixture(tmp_path);before=goal.to_dict();text=text.replace('此前','之前')
 assert scope_preserving_continuation(text,goal.approved_nodes)
 assert goal.apply_user_message(text) is False and goal.version==before['version']
 assert goal.active_goal==before['active_goal']
 after=old_parser_revision(type('Goal',(),{'to_dict':lambda self:copy.deepcopy(before)})(),text)
 assert metadata_only_goal_revision(before,after)
 path=tmp_path/'store.json';path.write_text(json.dumps({'workflows':{'w':state},'leases':{}}));store=WorkflowStore(path)
 assert store.reconcile_goal('w',after,owner=('tester','case'),require_unpaused=True)
 new=store.snapshot('w');assert new['goal_reconciliation_history'][-1]['metadata_only_verified'] is True
 assert new['nodes']==state['nodes'] and new['goal_contract']['version']==after['version']

@pytest.mark.parametrize('suffix',[
 '把温度改为300 K。','增加材料。','重新计算。','更换输入文件 foo.csv。',
 '并绘图。','并给别人发消息。','删除已有结果。','但是修改压力。','同时再提交任务。',
 '用另一种方法。','不要复核状态，改算另一体系。',
])
def test_review_mixed_with_new_scope_is_never_metadata(tmp_path,suffix):
 _,goal,_=fixture(tmp_path);text='请复核任务状态并更正说明。'+suffix
 assert not scope_preserving_continuation(text,goal.approved_nodes)
 assert not metadata_only_goal_revision(goal.to_dict(),old_parser_revision(goal,text))

def test_completed_runtime_tick_reconciles_status_review_without_dispatch(factory,tmp_path,monkeypatch):
 root=tmp_path/'runs/tester/case';read=node('read',tool='read_file',args={'path':str(root/'source.txt')})
 rt=factory([read],cid='case');rt.start(1)
 with json_transaction(rt.store.path) as data:
  w=data['workflows'][rt.workflow_id];w['nodes']['read']['status']='succeeded';w['nodes']['read']['phase']='finish';w['nodes']['read']['token']='owned-token';w['status']='completed'
 before=rt.snapshot();after=old_parser_revision(rt.main.goal_contract,'请根据当前执行记录复核并更正之前关于任务状态的说明，保留现有结果，不要重新计算。')
 from agents.goal_contract import GoalContract
 rt.main.goal_contract=GoalContract.from_dict(after);calls=[]
 monkeypatch.setattr(rt.main.registry.get('read_file'),'execute',lambda params:calls.append(params))
 rt.tick();updated=rt.snapshot()
 assert updated['goal_contract']['version']==after['version'] and updated['nodes']==before['nodes']
 assert updated['status']=='completed' and updated['goal_reconciliation_history'][-1]['metadata_only_verified'] and not calls
