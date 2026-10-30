"""User execution limits must survive incidental scientific method names."""
import pytest

from agents.goal_contract import GoalContract, is_readonly_request, prohibits_submission


@pytest.mark.parametrize('text', [
    '只读检查当前任务状态。',
    '这次仅只读查看日志。',
    '查看已存在的 cDFT 结果，不启动任何计算。',
    '不运行作业，先检查已有结果。',
    '暂时不要启动新的模拟，解释已有日志。',
    '禁止提交作业，查看当前队列。',
    '请不要执行任何计算，先核对状态。',
    'Inspect the existing MD results in read-only mode.',
    'Check existing results; do not start any calculation.',
    "Check the queue, don't launch any jobs.",
    'Inspect logs without submitting any jobs.',
    'Only inspect existing results; do not run simulations.',
    '不要准备输入，只读查看已有 MD 日志。',
    'Do not prepare inputs; only inspect logs.',
    '不是只读检查，但不运行新的作业，先查看结果。',
])
def test_explicit_investigation_does_not_authorize_submission(text):
    assert is_readonly_request(text)
    assert prohibits_submission(text)
    goal = GoalContract.from_user_message(text)
    assert goal.execution_mode == 'read_only'
    assert not goal.execution_authorized
    assert goal.guard_tool_call('list_my_jobs', {})[0]
    assert not goal.guard_tool_call('run_cdft', {'action': 'pipeline'})[0]


@pytest.mark.parametrize('text', [
    '准备 cDFT 输入文件，但不启动任何计算。',
    '只读检查结构并准备输入脚本，不提交。',
    'Prepare MD input files, but do not execute jobs.',
    'Generate the input files without submitting jobs.',
])
def test_preparing_inputs_preserves_prepare_only_scope(text):
    assert prohibits_submission(text)
    assert not is_readonly_request(text)
    goal = GoalContract.from_user_message(text)
    assert goal.execution_mode == 'prepare_only'
    assert not goal.execution_authorized
    assert not goal.guard_tool_call('run_md_optimize', {'cif_path': 'a.cif'})[0]


@pytest.mark.parametrize('text', [
    '不是只读检查，直接开始执行 cDFT 计算。',
    '并非只读，请直接执行 MD。',
    '不只是只读查看，直接执行 GCMC。',
    '不要只查状态，直接开始执行 cDFT。',
    'This is not read-only; execute now with cDFT.',
    'Not read only. Start now with MD.',
    'Do not only inspect the results; execute now with GCMC.',
    'Not only read-only; execute now with cDFT.',
    '不能不运行作业，直接开始执行 MD。',
])
def test_negated_readonly_scope_does_not_remove_execution_authority(text):
    assert not is_readonly_request(text)
    assert not prohibits_submission(text)
    goal = GoalContract.from_user_message(text)
    assert goal.execution_mode == 'workflow'
    assert goal.execution_authorized


def test_negating_readonly_scope_can_resume_existing_method_authority():
    goal = GoalContract.from_user_message('只读检查 cDFT 的 CO2 结果。')
    goal.apply_user_message('不是只读检查了，直接开始执行。')
    assert goal.execution_mode == 'workflow' and goal.execution_authorized
    assert goal.method == 'CDFT' and goal.gases == ['CO2']


def test_scientific_method_remains_pinned_across_scope_changes():
    goal = GoalContract.from_user_message('直接执行 cDFT 计算 CO2。')
    assert goal.method == 'CDFT' and goal.gases == ['CO2']
    goal.apply_user_message('先只读查看已有结果，不运行作业。')
    assert goal.execution_mode == 'read_only' and not goal.execution_authorized
    assert goal.method == 'CDFT' and goal.gases == ['CO2']
    goal.apply_user_message('直接开始执行。')
    assert goal.execution_mode == 'workflow'
    assert goal.method == 'CDFT' and goal.gases == ['CO2']
    assert not goal.guard_tool_call('run_gcmc_batch', {'gases': ['CO2']})[0]
def test_explicit_lammps_software_keeps_existing_md_contract():
    from agents.goal_contract import GoalContract
    contract = GoalContract.from_user_message('用LAMMPS优化已有框架，做一次快速功能测试')
    assert contract.method == 'MD'
    assert contract.guard_tool_call('run_md_optimize', {'cif_path': 'reference.cif', 'mode': 'single'})[0]
    assert not contract.guard_tool_call('run_cdft', {'action': 'pipeline'})[0]


@pytest.mark.parametrize('text', ['用String方法算扩散能垒', 'Use the String method for a transition path',
                                  '用 String/TST 研究 CO2 跃迁'])
def test_explicit_string_method_keeps_tst_contract(text):
    from agents.goal_contract import GoalContract
    assert GoalContract.from_user_message(text).method == 'TST'


@pytest.mark.parametrize('text', ['Explain Python string methods', '解释字符串string的用法'])
def test_generic_programming_strings_do_not_select_tst(text):
    from agents.goal_contract import GoalContract
    assert GoalContract.from_user_message(text).method != 'TST'
