"""Coordinator/executor custody for an already-approved scientific workflow.

Native role and GoalContract data choose the boundary. No user/material keyword,
tool argument, file extension, or model-provided context grants execution.
"""

_DOMAIN_TOOLS=frozenset({
    'generate_structure','run_pacman_charge','run_cdft','run_gcmc_isotherm',
    'run_gcmc_batch','run_isotherm_chain','run_henry','run_henry_chain',
    'run_vasp','run_md_optimize','run_string_tst','run_external_potential',
    'run_pore_analysis','run_xtb_optimize','calc_binding_energy','extract_features',
    'build_mof_database','build_guest_forcefield','stage_cif_subset','expand_cell',
    'analyze_gcmc_screening','analyze_cdft_screening','analyze_diffusion_msd',
    'generate_result_figures','generate_scientific_report','plot_result_code',
    'ml_train','ml_predict','ml_active_learning','ml_feature_importance',
})


def custody_policy(session,agent):
    role=getattr(agent,'name','')
    # A legacy handoff changes the role on the same coordinating Session; it
    # does not manufacture a native workflow worker. Real workers use fresh
    # Sessions without the main activation callback and execute their contract.
    if role!='lead-orchestrator'and not callable(getattr(session,'_on_workflow_start',None)):return None
    goal=getattr(session,'goal_contract',None)
    nodes=getattr(goal,'approved_nodes',None)
    if(not getattr(session,'_current_line_id',None)or not isinstance(nodes,list)or not nodes
            or not(getattr(goal,'approved_plan_version',0)>0 or getattr(goal,'execution_authorized',False))):
        return None
    scientific=any(isinstance(node,dict)and node.get('status')!='superseded'
        and not node.get('branch_parent')and node.get('tool')in _DOMAIN_TOOLS for node in nodes)
    if not scientific:return None
    from .parallel_workflow import READ_ONLY
    allowed=READ_ONLY|session._CHAIN_CONTROL_TOOLS|{
        'read_result_guidance','handoff_to_operations','handoff_to_patcher','handoff_to_lead-orchestrator'}
    if role in {'patcher','operations'}:
        allowed=allowed|{'restart_backend','run_project_regressions','build_project_frontend'}
    return {'allowed_tools':allowed,'goal_version':getattr(goal,'version',None),
        'plan_version':getattr(goal,'approved_plan_version',None)}


def request_tools(policy,tools):
    if policy is None:return tools
    return [tool for tool in tools if tool.get('name')in policy['allowed_tools']]


def custody_feedback(session,agent,tool):
    policy=custody_policy(session,agent)
    if policy is None or tool in policy['allowed_tools']:return None
    delivery=[name for name in ('generate_result_figures','generate_scientific_report')
              if session.registry.get(name)is not None]
    return {'blocked':True,'executed':False,'planning_required':True,
        'feedback_kind':'coordinator_execution_custody','scientific_failure':False,
        'goal_version':policy['goal_version'],'plan_version':policy['plan_version'],
        'blocked_tool':tool,'execution_owner':'approved_native_workflow_node',
        'no_execution_authority_granted':True,'requires_user':False,
        'reason':'The main coordinator owns planning and acceptance; the approved scientific runtime owns execution and deliverable writes.',
        'registered_delivery_tools':delivery,
        'next_native_actions':['task_line_query','get_tool_schema','propose_workflow_patch',
                               'apply_workflow_patch','execute_workflow'],
        'next_action':('Read the current owned completed sources. Compile presentation nodes with registered figure/report tools and owned new outputs, '
            'then apply/activate under the existing user scope. Do not author a shell/file alternative from main. '
            'If no native function covers the task, propose an owned controlled sandbox or repair node; do not bypass the runtime. '
            'Missing scientific choices still follow the original goal/authorization checks.')}


def prompt_contract():
    return ('\n[NATIVE COORDINATOR / EXECUTOR CUSTODY]\n'
        '当前已有批准的科研DAG。主协调者只读取、协商、编排、验收与派发；写文件、shell、直接计算和科研交付生成由批准节点的执行器负责。'
        '这些执行工具不会在主协调请求中暴露。绘图/报告优先编译已注册generate_result_figures/generate_scientific_report节点，'
        '绑定当前owned source_steps与新输出，再apply/execute；缺功能走受控sandbox/repair节点。'
        '不得把DAG控制或绘图节点替换为主chat手写脚本/SVG/报告，不能因旧图completed就宣称新交付已生成。'
        '技术门禁反馈不是科学计算失败，也不需要用户授权绕过。')


def bounded_stop_reply():
    return '已有任务及结果记录已保留；新的执行安排尚未完成。这些操作未执行，尚未生成对应的新交付物，本轮已停止重复尝试。'
