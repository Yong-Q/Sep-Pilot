"""One read model for approved contracts and executor-owned execution facts."""
import copy

from .workflow_patch import execution_contract


def authoritative_line(line, runtime):
    result = copy.deepcopy(line or {})
    if not runtime:
        result['state_authority'] = 'task_line_legacy'
        return result
    if ((result.get('username') and result['username'] != runtime.get('username'))
            or (result.get('conv_id') and result['conv_id'] != runtime.get('conv_id'))):
        raise PermissionError('runtime and approved line belong to different user/session scopes')
    result.update(state_authority='workflow_executor', execution_plan_version=runtime.get('plan_version'),
                  execution_status=runtime.get('status'), user_paused=runtime.get('user_paused', False))
    for step in result.get('steps', []):
        actual = runtime.get('nodes', {}).get(step['step_id'])
        if not actual:
            continue
        step['reported_state'] = {k: step.get(k) for k in ('status', 'done', 'job_ids')}
        step['execution_contract'] = copy.deepcopy(actual['contract'])
        step['contract_matches_execution'] = execution_contract(step) == execution_contract(actual['contract'])
        if not step['contract_matches_execution']:
            step['previous_execution'] = copy.deepcopy(actual)
            step.update(status='pending', done=False, job_ids=[], output_files=[],
                        path_manifest={}, error=None, node_verification=None)
            continue
        step.update(status=actual['status'], done=actual['status'] == 'succeeded', phase=actual.get('phase'),
                    job_ids=list(actual.get('job_ids', [])), state_authority='workflow_executor',
                    execution_plan_version=actual.get('execution_plan_version', runtime.get('plan_version')),
                    error=actual.get('error'), result_ref=copy.deepcopy(actual.get('result_ref')),
                    node_verification=copy.deepcopy(actual.get('node_verification')),
                    resolved_arguments=copy.deepcopy(actual.get('resolved_arguments')),
                    runtime_input_bindings=copy.deepcopy(actual.get('runtime_input_bindings', [])),
                    runtime_binding_fingerprint=actual.get('runtime_binding_fingerprint'))
        if actual.get('path_manifest'):
            step['path_manifest'] = copy.deepcopy(actual['path_manifest'])
            step['output_files'] = list(actual['path_manifest'].get('output_files', []))
        if actual.get('contract', {}).get('tool') == 'generate_scientific_report':
            receipt = actual.get('result') or {}
            step['report_receipt'] = copy.deepcopy({key: receipt.get(key) for key in (
                'report_file', 'output_path', 'source_steps', 'source_step',
                'source_attempt_fingerprint', 'evidence_fingerprint',
                'fragment_fingerprint', 'fragment_manifest') if receipt.get(key) is not None})
    return result


def line_digest(line):
    rows = [f"Approved plan={line.get('plan_version')}; execution plan={line.get('execution_plan_version')}; state authority={line.get('state_authority')}"]
    for node in line.get('steps', []):
        if 'expected_outputs' not in node:
            continue
        rows.append(f"{node['step_id']}: agent={node.get('agent')} tool={node.get('tool')} status={node.get('status')} "
                    f"done={node.get('done')} dependencies={node.get('depends_on', [])} jobs={node.get('job_ids', [])} "
                    f"contract_matches_execution={node.get('contract_matches_execution', True)} error={str(node.get('error') or '')[:400]}")
    return '\n'.join(rows)


def bind_result_handoff(session, runtime_getter):
    """Production and SDK acceptance use the same parameter-preserving binding."""
    session._on_workflow_revalidate = lambda step_id, expected_outputs, completed_job_id=None: runtime_getter().revalidate_outputs(
        step_id, expected_outputs, completed_job_id=completed_job_id)
    session._on_workflow_finish = lambda step_id, evidence_call_ids, conclusion: runtime_getter().finish_node(
        step_id, evidence_call_ids, conclusion)
    session._on_workflow_runtime_repair = lambda step_id: runtime_getter().repair_runtime_inputs(step_id)


def restore_approved_nodes(goal, runtime, owner):
    """Recover missing contracts, never grant execution or change science."""
    if goal.approved_nodes or not runtime:
        return []
    if {k: runtime.get(k) for k in ('username', 'conv_id')} != owner:
        raise PermissionError('cannot restore a foreign workflow approval')
    saved = runtime.get('goal_contract', {})
    if not goal.approved_plan_version or runtime.get('plan_version') != goal.approved_plan_version:
        return []
    current = goal.to_dict()
    if (saved.get('approved_plan_version') != goal.approved_plan_version
            or not saved.get('execution_authorized')
            or any(current.get(k) != saved.get(k) for k in
                   ('version', 'gases', 'method', 'parameters', 'execution_mode'))):
        return []
    approved_ids = {n.get('step_id') for n in saved.get('approved_nodes', [])}
    if not approved_ids or approved_ids != set(runtime.get('nodes', {})):
        return []
    return [{'step_id': key, **execution_contract(node['contract'])}
            for key, node in runtime['nodes'].items()]


def workflow_completion(line, runtime, approved_nodes, approved_version):
    """Executable contracts alone determine completion; diagnostics are history."""
    from .recovery import is_submission
    view = authoritative_line(line, runtime)  # Also enforces scope isolation.
    if runtime and runtime.get('plan_version') != approved_version:
        return {'ok': False, 'reason': 'plan_version_mismatch', 'blockers': []}
    ids = {n['step_id'] for n in approved_nodes}
    if runtime:
        if ids and ids != set(runtime.get('nodes', {})):
            return {'ok': False, 'reason': 'approval_node_set_mismatch', 'blockers': []}
        ids = ids or set(runtime.get('nodes', {}))
        nodes = runtime.get('nodes', {})
        blockers = []
        contracts = {n['step_id']: n for n in approved_nodes}
        for key in sorted(ids):
            node = nodes.get(key)
            if not node:
                blockers.append({'step_id': key, 'status': 'missing'})
            elif key in contracts and execution_contract(contracts[key]) != execution_contract(node['contract']):
                blockers.append({'step_id': key, 'status': 'contract_mismatch'})
            elif node.get('status') != 'succeeded' or (is_submission(
                    node['contract'].get('tool'), node['contract'].get('arguments', {}))
                    and not node.get('node_verification')):
                blockers.append({'step_id': key, 'status': node.get('status', 'missing')})
    else:
        contracts = {n['step_id']: n for n in view.get('steps', [])
                     if ('expected_outputs' in n or not approved_version and (
                         n.get('status', 'pending') in {'pending', 'running', 'submitted'} or n.get('job_ids')))
                     and not n.get('branch_parent')
                     and n.get('status') != 'superseded'
                     and int(n.get('plan_version', 0) or 0) in {0, approved_version}}
        ids = ids or set(contracts)
        blockers = [{'step_id': key, 'status': contracts.get(key, {}).get('status', 'missing')}
                    for key in sorted(ids) if contracts.get(key, {}).get('status') != 'completed']
    if not ids:
        return {'ok': not approved_version, 'reason': 'approved_contract_missing', 'blockers': []}
    return {'ok': not blockers, 'reason': 'unfinished_nodes' if blockers else 'verified', 'blockers': blockers}
