"""Recognize only typed parameter edits already fixed by the user goal.

This is an authorization comparison, not a text classifier or a second planner.
The normal compiler, output contracts, running-job leases and goal guards remain
responsible for validating and applying a complete candidate DAG.
"""
import copy

from .workflow_patch import operational_patch


def proposal_goal_scope(goal):
    """Typed authority and approved graph, independent of conversation wording."""
    from .workflow_patch import execution_contract
    data=goal.to_dict()
    return {**{key:copy.deepcopy(data.get(key)) for key in (
        'method','gases','parameters','execution_mode','execution_authorized','approved_plan_version')},
        'approved_contracts':[execution_contract(node) for node in data.get('approved_nodes',[])]}


def rebase_confirmed_proposal(patch, goal):
    """Bind an authenticated apply to the same typed scope after a text revision.

    This function grants no approval. Session first checks the executing user,
    then the normal graph, scope, resource and runtime validators still run.
    Legacy proposals without a captured scope cannot be cross-version rebased.
    """
    result=copy.deepcopy(patch)
    version=result.get('goal_version',goal.version)
    if type(version) is not int or type(goal.version) is not int or version>goal.version:
        raise ValueError('user goal changed after this proposal; recompile against current constraints')
    if version==goal.version:
        return result
    if (result.get('goal_scope')!=proposal_goal_scope(goal)
            or result.get('base_version')!=goal.approved_plan_version):
        raise ValueError('user goal changed after this proposal; recompile against current constraints')
    result['goal_revision_rebase']={'from_version':result['goal_version'],'to_version':goal.version,
        'basis':'authenticated controller applied unchanged typed scope and approved base graph'}
    result['goal_version']=goal.version
    return result


def parameter_reconciliation(goal, runtime=None):
    """Show typed goal/plan differences to the planner without changing state."""
    from .goal_contract import _TOOL_METHOD
    runtime = runtime or {}
    differences = []
    for node in goal.approved_nodes:
        tool, arguments = node.get('tool'), node.get('arguments') or {}
        bindings = {}
        if _TOOL_METHOD.get(tool) in {'GCMC', 'HENRY'}:
            bindings['cycles'] = 'cycles'
        if 'temperature' in arguments:
            bindings['temperature'] = 'temperature_K'
        if tool == 'generate_structure':
            bindings.update(n_structures='n_structures', max_atoms='max_atoms')
        for argument, field in bindings.items():
            expected = goal.parameters.get(field)
            if expected is None or arguments.get(argument) == expected:
                continue
            state = runtime.get('nodes', {}).get(node['step_id'], {})
            differences.append({'step_id': node['step_id'], 'tool': tool, 'argument': argument,
                                'planned_value': copy.deepcopy(arguments.get(argument)),
                                'current_user_constraint': copy.deepcopy(expected),
                                'status': state.get('status', node.get('status', 'pending')),
                                'job_ids': copy.deepcopy(state.get('job_ids', []))})
    if not differences:
        return None
    return {'goal_version': goal.version, 'approved_plan_version': goal.approved_plan_version,
            'status': 'goal_plan_mismatch', 'applied': False, 'differences': differences}


def requested_parameter_patch(existing_steps, changes, goal, project_root=None):
    old = {node['step_id']: node for node in existing_steps}
    reset = copy.deepcopy(changes)
    proof = []
    for change in reset:
        before = old.get(change.get('step_id'))
        after = change.get('node', {})
        if change.get('operation') != 'upsert' or not before or before.get('tool') != after.get('tool'):
            return None
        if (before.get('done') or before.get('job_ids')
                or before.get('status') in {'completed', 'succeeded', 'running', 'submitted',
                                           'waiting_jobs', 'uncertain', 'prefinish'}):
            return None
        # A parameter instruction does not authorize replacing the node owner,
        # data-flow edges, input provenance or resource-lock semantics.
        for field, default in (('agent', None), ('depends_on', []),
                               ('resource_locks', []), ('input_bindings', {})):
            if before.get(field, default) != after.get(field, default):
                return None
        prior_args, args = before.get('arguments', {}), after.get('arguments', {})
        for argument in set(prior_args) | set(args):
            if prior_args.get(argument) == args.get(argument):
                continue
            field = None
            if argument == 'cycles' and after.get('tool') in {'run_gcmc_batch', 'run_gcmc_isotherm'}:
                field = 'cycles'
            elif argument == 'temperature':
                field = 'temperature_K'
            elif argument in {'n_structures', 'max_atoms'} and after.get('tool') == 'generate_structure':
                field = argument
            if field is None:
                continue  # The remaining edit must pass the operational check.
            expected = goal.parameters.get(field)
            if expected is None or isinstance(expected, (list, dict, bool)) or args.get(argument) != expected:
                return None
            if field in {'cycles', 'n_structures', 'max_atoms'} and (
                    type(args.get(argument)) is not int or args[argument] <= 0):
                return None
            proof.append({'step_id': change['step_id'], 'argument': argument,
                          'before': copy.deepcopy(prior_args.get(argument)),
                          'after': copy.deepcopy(args[argument]), 'goal_field': field})
            if argument in prior_args:
                args[argument] = copy.deepcopy(prior_args[argument])
            else:
                args.pop(argument, None)
    if not proof or not operational_patch(existing_steps, reset, project_root):
        return None
    return {'goal_version': goal.version, 'changes': sorted(proof, key=lambda item: (item['step_id'], item['argument']))}
