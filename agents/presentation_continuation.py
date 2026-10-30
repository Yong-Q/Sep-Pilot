"""Classify a native DAG delta; never authorize, apply or execute it.

Call after normal compilation/schema/capability/scope checks, with snapshots
read by the owning controller (never model-supplied runtime/goal evidence).
The receipt is advisory. Existing authenticated control, current user scope,
pause/version/CAS and invocation/acceptance checks remain mandatory.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

PRESENTATION_TOOLS = frozenset({'generate_result_figures', 'generate_scientific_report'})
PRESENTATION_AGENTS = frozenset({'analyst', 'communicator'})
SCIENTIFIC_SOURCE_TOOLS = frozenset({
    'generate_structure', 'run_pacman_charge', 'validate_framework_charges',
    'run_pore_analysis', 'run_md_optimize', 'run_string_tst', 'run_external_potential',
    'run_gcmc_isotherm', 'run_gcmc_batch', 'run_henry', 'run_vasp', 'run_cdft',
    'validate_gcmc_results', 'analyze_gcmc_screening', 'analyze_cdft_screening',
    'extract_features',
})
CONTRACT_FIELDS = ('agent', 'tool', 'arguments', 'input_bindings', 'depends_on',
                   'expected_outputs', 'resource_locks')
PATH_FIELDS = frozenset({'input_dir', 'input_path', 'cif_dir', 'work_dir', 'output_dir',
    'job_work_dir', 'output_path', 'output_csv', 'trajectory_path', 'manifest_path',
    'model_dir', 'potential_dir', 'structure_path'})
_IDENTITY = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')


class _Reject(ValueError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _contract(node):
    return {key: copy.deepcopy(node.get(key, [] if key in {
        'depends_on', 'expected_outputs', 'resource_locks'} else {} if key in {
        'arguments', 'input_bindings'} else None)) for key in CONTRACT_FIELDS}


def _require(condition, reason):
    if not condition:
        raise _Reject(reason)


def _ids(values):
    _require(isinstance(values, list) and 1 <= len(values) <= 16, 'invalid_source_steps')
    _require(all(isinstance(x, str) and _IDENTITY.fullmatch(x) for x in values), 'invalid_step_identity')
    _require(len(set(values)) == len(values), 'duplicate_source_steps')
    return values


def _path(raw, root, *, fresh=False):
    _require(isinstance(raw, str) and raw == raw.strip() and raw and
        not any(c in raw for c in ('\x00', '\n', '\r', '{', '}', '*', '?', '[')), 'invalid_literal_output_path')
    target = Path(raw)
    _require(target.is_absolute() and '..' not in target.parts and target != root and
        target.is_relative_to(root), 'output_outside_owned_conversation')
    # Reject links in every existing ancestor, including a linked owner root.
    _require(not any(part.is_symlink() for part in (target, *target.parents)), 'symlink_output_path')
    relative = target.relative_to(root)
    _require(relative.parts[0] not in {'workers', 'evidence', 'state', 'scientific-owners'} and
        not any(part.startswith('.') for part in relative.parts) and
        target.name not in {'session_checkpoint.json', 'task_manifest.json', 'lifecycle.json',
                            'orchestration_chain.json', 'orchestration_current.json', 'recovery_state.json'},
        'output_overlaps_control_state')
    if fresh:
        _require(not target.exists(), 'output_already_exists')
    return target


def valid_report_requirements(requirements):
    return (isinstance(requirements,dict) and set(requirements)<={'minimum_figures','minimum_citations','independent_review','require_explanatory_narrative','source_only','section_titles'}
        and all(type(requirements.get(k,0)) is int and 0<=requirements.get(k,0) for k in ('minimum_figures','minimum_citations'))
        and all(type(requirements.get(k,False)) is bool for k in ('independent_review','require_explanatory_narrative','source_only'))
        and isinstance(requirements.get('section_titles',[]),list)
        and all(isinstance(v,str) and v.strip() and '\n' not in v for v in requirements.get('section_titles',[]))
        and len(set(requirements.get('section_titles',[])))==len(requirements.get('section_titles',[])))


def classify_presentation_continuation(existing_steps, changes, *, project_root, owner, goal, runtime):
    """Return technical_delivery_only/no_execution_authority, with no writes.

    Only fresh outputs and explicit native source_steps are accepted. Legacy
    work_dir reports, arbitrary scripts/input paths, old presentation results,
    unverified sources and unknown schema fields conservatively use the normal
    controller review route. A positive result is never an approval receipt.
    """
    receipt = {'schema_version': 1, 'technical_delivery_only': False,
        'no_execution_authority': True, 'executed': False, 'applied': False}
    try:
        _require(isinstance(owner, dict) and set(owner) == {'username', 'conv_id'} and
            all(isinstance(x, str) and _IDENTITY.fullmatch(x) and x not in {'.', '..'}
                for x in owner.values()), 'invalid_owner')
        _require(isinstance(goal, dict) and isinstance(runtime, dict), 'missing_native_context')
        _require({k: runtime.get(k) for k in owner} == owner, 'foreign_runtime_owner')
        # A proposal advances only pending_plan_version before application.
        # That bookkeeping is not an approved goal/scientific revision.
        native_goal = runtime.get('goal_contract')
        _require(isinstance(native_goal, dict) and
            _canonical({k:v for k,v in native_goal.items() if k != 'pending_plan_version'}) ==
            _canonical({k:v for k,v in goal.items() if k != 'pending_plan_version'}), 'stale_or_foreign_goal')
        version = goal.get('approved_plan_version')
        _require(type(version) is int and version > 0 and runtime.get('plan_version') == version,
            'stale_plan_version')
        _require(isinstance(existing_steps, list) and 1 <= len(existing_steps) <= 256 and
            isinstance(changes, list) and 1 <= len(changes) <= 64, 'invalid_patch_size')
        project = Path(project_root)
        _require(project.is_absolute(), 'project_root_must_be_absolute')
        root = project / 'runs' / owner['username'] / owner['conv_id']
        old = {}
        for node in existing_steps:
            _require(isinstance(node, dict) and isinstance(node.get('step_id'), str) and
                _IDENTITY.fullmatch(node['step_id']) and node['step_id'] not in old, 'invalid_existing_graph')
            old[node['step_id']] = copy.deepcopy(node)
        approved = {n['step_id']: n for n in goal.get('approved_nodes', [])}
        native = runtime.get('nodes')
        _require(isinstance(native, dict), 'missing_native_nodes')
        science = {key: node for key, node in old.items() if node.get('tool') not in PRESENTATION_TOOLS}
        _require(bool(science), 'no_existing_scientific_graph')
        for key, node in science.items():
            _require(key in approved and key in native and
                _canonical(_contract(node)) == _canonical(_contract(approved[key])) ==
                _canonical(_contract(native[key].get('contract', {}))), 'scientific_contract_not_current')
        graph = copy.deepcopy(old)
        edited = set(); retired = set()
        for change in changes:
            if isinstance(change, dict) and change.get('operation') == 'remove':
                key = change.get('step_id')
                _require(set(change) == {'operation', 'step_id'} and key in old and key not in science
                    and old[key].get('tool') in PRESENTATION_TOOLS and key not in retired and key not in edited,
                    'only_historical_presentation_may_be_archived')
                _require(native.get(key, {}).get('status') not in {'running', 'submitted', 'waiting_jobs',
                    'waiting_prerequisite', 'uncertain', 'prefinish'}, 'presentation_node_still_owns_execution')
                graph.pop(key); retired.add(key)
                continue
            _require(isinstance(change, dict) and set(change) == {'operation', 'step_id', 'node'} and
                change.get('operation') == 'upsert', 'only_presentation_upserts_allowed')
            key, node = change['step_id'], change['node']
            _require(isinstance(key, str) and _IDENTITY.fullmatch(key) and key not in edited and key not in retired,
                'duplicate_or_invalid_patch_identity')
            _require(isinstance(node, dict) and node.get('step_id', key) == key and
                set(node) <= set(CONTRACT_FIELDS) | {'step_id', 'report_role', 'report_parent', 'report_source'},
                'invalid_presentation_contract_fields')
            _require(node.get('tool') in PRESENTATION_TOOLS and node.get('agent') in PRESENTATION_AGENTS,
                'not_a_presentation_tool_or_role')
            _require(key not in science, 'cannot_replace_scientific_node')
            _require(not node.get('resource_locks') and not node.get('input_bindings'), 'custom_bindings_or_locks_forbidden')
            if key in old:
                _require(native.get(key, {}).get('status') not in {'running', 'submitted', 'waiting_jobs',
                    'waiting_prerequisite', 'uncertain', 'prefinish'},
                    'presentation_node_still_owns_execution')
            graph[key] = {**copy.deepcopy(node), 'step_id': key}
            edited.add(key)
        _require(bool(edited), 'archival_requires_new_or_updated_presentation')
        added = edited - set(old)
        protected = []
        def protect(raw):
            if not isinstance(raw, str) or not raw:
                return
            path = Path(raw)
            if path.is_absolute():
                protected.append(path)
            else:
                # Canonical compiler input is absolute; conservative aliases
                # also protect older project/conversation-relative contracts.
                protected.extend([project / path, root / path])
        for node in science.values():
            for raw in node.get('expected_outputs', []):
                raw = raw.get('path') if isinstance(raw, dict) else raw
                protect(raw)
            for key, value in node.get('arguments', {}).items():
                if key in PATH_FIELDS:
                    protect(value)
        for key, node in old.items():
            if key not in science and key not in edited:
                for field in ('output_dir', 'output_path'):
                    protect(node.get('arguments', {}).get(field))
        destinations = []
        sources_used = set()
        for key in edited:
            node = graph[key];args = node.get('arguments')
            _require(isinstance(args, dict), 'missing_presentation_arguments')
            sources = _ids(args.get('source_steps'))
            _require(node.get('depends_on') == sources, 'source_steps_must_equal_direct_dependencies')
            if any(field in node for field in ('report_role', 'report_parent', 'report_source')):
                _require(node['tool'] == 'generate_scientific_report', 'report_metadata_on_non_report')
                if node.get('report_role') == 'fragment':
                    parent = graph.get(node.get('report_parent'), {})
                    _require(args.get('report_mode') == 'fragment' and len(sources) == 1 and
                        node.get('report_source') == sources[0] and parent.get('tool') == 'generate_scientific_report' and
                        parent.get('report_role') == 'assembly' and key in parent.get('depends_on', []),
                        'invalid_compiler_fragment_association')
                else:
                    _require(node.get('report_role') == 'assembly' and args.get('report_mode') == 'assembly' and
                        not node.get('report_parent') and not node.get('report_source') and
                        all(graph.get(s, {}).get('report_parent') == key and
                            graph.get(s, {}).get('report_role') == 'fragment' for s in sources),
                        'invalid_compiler_assembly_association')
            for source in sources:
                _require(source in graph and source != key, 'missing_or_self_source')
                if source in added:
                    _require(graph[source].get('tool') in PRESENTATION_TOOLS, 'new_source_must_be_presentation')
                else:
                    _require(source in science and science[source].get('tool') in SCIENTIFIC_SOURCE_TOOLS and
                        science[source].get('agent') not in {'harness-maintainer', 'monitor', 'supervisor', 'operations'} and
                        native[source].get('status') == 'succeeded', 'source_not_current_successful_science')
                    for raw in science[source].get('expected_outputs', []):
                        raw = raw.get('path') if isinstance(raw, dict) else raw
                        _require(isinstance(raw, str) and Path(raw).is_absolute() and
                            Path(raw).is_relative_to(root) and '..' not in Path(raw).parts and
                            not any(p.is_symlink() for p in (Path(raw), *Path(raw).parents)),
                            'scientific_source_output_not_owned')
                    reference = native[source].get('result_ref')
                    _require(isinstance(reference, dict) and isinstance(reference.get('call_id'), str) and
                        bool(reference['call_id']), 'source_missing_native_evidence_reference')
                    for field in ('checkpoint_path', 'evidence_path'):
                        raw = reference.get(field)
                        _require(isinstance(raw, str) and Path(raw).is_absolute() and
                            Path(raw).is_relative_to(root) and '..' not in Path(raw).parts and
                            not any(p.is_symlink() for p in (Path(raw), *Path(raw).parents)),
                            'source_evidence_outside_owned_conversation')
                    sources_used.add(source)
            common = {'source_steps', 'title', 'instructions'}
            if node['tool'] == 'generate_result_figures':
                _require(set(args) <= common | {'output_dir', 'backend'} and args.get('backend', 'python') == 'python',
                    'unsupported_figure_argument')
                _require(isinstance(args.get('instructions'), str) and 1 <= len(args['instructions']) <= 8000,
                    'missing_figure_instructions')
                target = _path(args.get('output_dir'), root, fresh=True)
            else:
                _require(set(args) <= common | {'output_path', 'report_mode', 'report_requirements'},
                    'unsupported_report_argument')
                _require(args.get('report_mode', 'fragment') in {'fragment', 'assembly'}, 'unsupported_report_mode')
                requirements = args.get('report_requirements', {})
                _require(valid_report_requirements(requirements),'unsupported_report_requirements')
                target = _path(args.get('output_path'), root, fresh=True)
                _require(target.suffix.lower() == '.md', 'report_must_be_markdown')
                _require('title' in args, 'report_requires_title')
            if 'title' in args:
                _require(isinstance(args['title'], str) and 1 <= len(args['title']) <= 500, 'invalid_title')
            if 'instructions' in args:
                _require(isinstance(args['instructions'], str) and 1 <= len(args['instructions']) <= 8000,
                    'invalid_instructions')
            _require(not any(target == p or target.is_relative_to(p) or p.is_relative_to(target)
                for p in protected + destinations), 'output_overlaps_science_or_other_delivery')
            destinations.append(target)
            outputs = node.get('expected_outputs')
            _require(isinstance(outputs, list) and 1 <= len(outputs) <= 16, 'missing_output_contract')
            for output in outputs:
                if isinstance(output, dict):
                    _require(set(output) <= {'kind', 'path', 'pattern', 'min_count'} and
                        output.get('kind') in {'file', 'directory'}, 'unsupported_output_contract')
                    pattern = output.get('pattern', '*')
                    _require(isinstance(pattern, str) and '/' not in pattern and '\\' not in pattern and pattern not in {'.', '..'},
                        'invalid_output_pattern')
                    _require(type(output.get('min_count', 1)) is int and output.get('min_count', 1) >= 1,
                        'invalid_output_count')
                    _require(output['kind'] != 'file' or not set(output) & {'pattern', 'min_count'},
                        'file_contract_cannot_have_directory_options')
                    raw = output.get('path')
                else:
                    raw = output
                path = _path(raw, root)
                _require(path == target or node['tool'] == 'generate_result_figures' and path.is_relative_to(target),
                    'output_contract_outside_tool_destination')
        # The entire graph remains acyclic. No scientific consumer may receive
        # an edited presentation producer (even through unchanged intermediate nodes).
        visiting, done = set(), set()
        def visit(key):
            _require(key in graph and key not in visiting, 'missing_dependency_or_cycle')
            if key in done:
                return set()
            visiting.add(key)
            ancestors = set()
            for dep in graph[key].get('depends_on', []):
                ancestors.add(dep);ancestors.update(visit(dep))
            visiting.remove(key);done.add(key)
            if key in science:
                _require(not ancestors & edited, 'presentation_changes_feed_scientific_execution')
            return ancestors
        # Do not memoize ancestry: each science node requires its full closure.
        for key in graph:
            done.clear();visit(key)
        receipt.update(technical_delivery_only=True, reason='same_owned_goal_science_unchanged',
            changed_steps=sorted(edited | retired), added_steps=sorted(added), retired_presentation_steps=sorted(retired), scientific_source_steps=sorted(sources_used),
            owner=copy.deepcopy(owner), goal_version=goal.get('version'), base_plan_version=version,
            changes_sha256=hashlib.sha256(_canonical(changes).encode()).hexdigest(),
            scientific_contracts_sha256=hashlib.sha256(_canonical({key:_contract(value)
                for key,value in science.items()}).encode()).hexdigest(),
            source_receipts_sha256=hashlib.sha256(_canonical({key:{field:native[key].get(field)
                for field in ('status','token','updated_at','result_ref','execution_fingerprint','artifacts',
                              'node_verification','origin_goal_version','execution_plan_version')}
                for key in sorted(sources_used)}).encode()).hexdigest())
    except (_Reject, KeyError, TypeError, ValueError, AttributeError, OSError, OverflowError, RecursionError) as error:
        receipt['reason'] = str(error) if isinstance(error, _Reject) else 'invalid_native_context_or_contract'
    return receipt
