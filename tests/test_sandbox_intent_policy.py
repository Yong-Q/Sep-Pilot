import copy
import pytest
from agents.goal_contract import GoalContract
from agents.sandbox_execution_policy import sandbox_requested,safe_readonly_shell,sandbox_violation
from agents.workflow_patch import WorkflowContractError
from tests.test_parallel_workflow import factory,node,eventually

@pytest.mark.parametrize('text',['请在隔离环境自主编写分析代码','请使用沙盒处理文件','run analysis in an isolated environment','沙盒里读取输入并分析'])
def test_explicit_isolation_blocks_only_unsafe_shell(text):
 goal=GoalContract.from_user_message(text)
 assert sandbox_requested(goal)
 assert sandbox_violation(goal,'run_bash',{'command':'python3 analysis.py'})
 assert not sandbox_violation(goal,'run_bash',{'command':'ls -l /tmp'})
 assert not sandbox_violation(goal,'run_analysis_sandbox',{'script':'pass'})
 assert not goal.guard_tool_call('run_bash',{'command':'python3 analysis.py'})[0]

@pytest.mark.parametrize('command',['python3 -c "print(1)"','bash script.sh','sh -c "ls"','ls | python3','ls; touch x','cat /tmp/x > /tmp/y','env python3 x.py','$(python3 x.py)','./analysis.py'])
def test_nonisolated_execution_cannot_disguise_as_diagnostic(command):
 assert not safe_readonly_shell(command)

@pytest.mark.parametrize('text',['常规分析已有文件并写报告','沙盒不可用，请解释原因','无需使用沙盒，正常执行'])
def test_other_tasks_do_not_acquire_isolation_requirement(text):
 goal=GoalContract.from_user_message(text);assert not sandbox_requested(goal)
 assert goal.guard_tool_call('run_bash',{'command':'python3 analysis.py'})[0]


def test_proposal_rejects_nonisolated_code_with_actionable_alternative(tmp_path,monkeypatch):
 from agents.session import Session
 from agents.config import AgentConfig
 from agents.task_line import TaskLineStore
 from agents.defns import ORCHESTRATOR
 from agents.watch_context import set_context,clear_context
 store=TaskLineStore(str(tmp_path/'lines.json'));store.begin_line('c1',username='tester',conv_id='c1');monkeypatch.setattr('agents.task_line.get_store',lambda:store)
 s=Session(config=AgentConfig(api_key='test',project_root=tmp_path));s.current_agent=ORCHESTRATOR;s._current_line_id='c1';s.goal_contract=GoalContract.from_user_message('请在隔离环境自主编写分析代码并输出 JSON，直接执行')
 change={'operation':'upsert','step_id':'bad','node':node('bad',tool='run_bash',args={'command':'python3 analysis.py','cwd':str(tmp_path/'runs/tester/c1')})}
 set_context('tester','c1')
 try:
  with pytest.raises(WorkflowContractError,match='sandbox-required')as error:s._propose_workflow_patch(0,'Run the requested analysis safely',[change])
  assert error.value.details['error_kind']=='workflow_contract' and error.value.details['contract_code']=='sandbox_required' and 'run_analysis_sandbox' in error.value.details['next_action']
 finally:clear_context()


def test_worker_rechecks_user_isolation_after_original_approval(factory,tmp_path):
 from agents.goal_contract import GoalContract
 command=node('shell',tool='run_bash',args={'command':'python3 analysis.py','cwd':str(tmp_path/'runs/tester/policy')})
 rt=factory([command],cid='policy');called=[];rt.main.registry.get('run_bash').execute=lambda args:called.append(args)
 rt.start(1);ticket=rt.store.claim(rt.workflow_id,'shell');assert ticket
 rt.main.goal_contract.apply_user_message('请在隔离环境执行分析代码')
 rt._worker_body('shell',ticket)
 result=rt.snapshot()['nodes']['shell'];assert result['status']=='failed' and not called and not result.get('tool_returned')
 assert 'run_analysis_sandbox' in result['error']


def test_apply_rechecks_isolation_after_user_changes_intent(tmp_path,monkeypatch):
 from agents.session import Session
 from agents.config import AgentConfig
 from agents.defns import ORCHESTRATOR
 from agents.watch_context import set_context,clear_context
 s=Session(config=AgentConfig(api_key='test',project_root=tmp_path));s.current_agent=ORCHESTRATOR;s._current_line_id='c1'
 s.goal_contract=GoalContract.from_user_message('请在沙盒中运行已有数据分析')
 bad=node('bad',tool='run_bash',args={'command':'python3 analysis.py'})
 s._pending_workflow_patch={'new_version':2,'proposed_nodes':[bad]}
 s._operation_control_source=lambda:{'origin':'user','owner':{'username':'tester','conv_id':'c1'},'turn_id':'turn'}
 monkeypatch.setattr('agents.requested_parameters.rebase_confirmed_proposal',lambda patch,goal:patch)
 set_context('tester','c1')
 try:
  with pytest.raises(ValueError,match='隔离环境'):s._apply_workflow_patch(2)
  assert s._pending_workflow_patch['new_version']==2
 finally:clear_context()

@pytest.mark.parametrize('suffix',['.csv','.json','.tsv','.txt','.data'])
def test_sandbox_task_cannot_bypass_analysis_by_direct_data_write(suffix):
 goal=GoalContract.from_user_message('请在隔离环境自主编写数据分析代码并导出结果')
 assert not goal.guard_tool_call('write_file',{'path':'output'+suffix,'content':'precomputed'})[0]
 assert goal.guard_tool_call('write_file',{'path':'report.md','content':'a narrative'})[0]


def test_pending_realistic_direct_data_shortcut_is_rejected():
 from agents.sandbox_execution_policy import validate_sandbox_changes
 goal=GoalContract.from_user_message('请在沙盒中分析数据并导出 CSV 和 JSON')
 changes=[{'operation':'upsert','step_id':'fake','node':node('fake',tool='write_file',args={'path':'result.csv','content':'x\n1'})}]
 with pytest.raises(WorkflowContractError,match='sandbox-required') as error:validate_sandbox_changes(goal,changes)
 assert error.value.details['executed'] is False and 'run_analysis_sandbox' in error.value.details['next_action']

def test_later_explicit_user_releases_isolation_requirement():
 goal=GoalContract.from_user_message('请使用沙盒分析现有数据')
 assert sandbox_requested(goal)
 goal.apply_user_message('不用沙盒，按普通方式继续')
 assert not sandbox_requested(goal)
 assert goal.guard_tool_call('run_bash',{'command':'python3 analysis.py'})[0]
