"""Read-only installation facts and executor provenance, never scientific jobs."""
import hashlib
from itertools import islice
import json
import os
import re
from pathlib import Path


CDFT_RECOVERY_DEPENDENCIES = (
    'agents/executor_repair_controller.py',
    'agents/workflow_execution_identity.py',
    'agents/__init__.py', 'agents/placement_budget.py', 'agents/node_inventory.py',
    'agents/cdft_submission_receipt.py',
    'bimem_agent/python_child.py', 'agents/runtime_health_probe.py',
    'agents/cdft_partial_recovery.py', 'agents/cdft_failure_facts.py', 'agents/cdft_grid_guard.py', 'agents/native_resource_carry.py', 'agents/native_timing.py', 'agents/report_recovery.py',
    'agents/local_preparation_recovery.py', 'agents/recovery.py',
    'agents/workload_budget.py', 'agents/workload_context.py',
    'agents/cdft_result_facts.py', 'agents/parallel_workflow.py',
    'agents/control_policy.py', 'agents/raspa_assets.py',
    'agents/job_watch.py', 'agents/watch_context.py', 'agents/workflow_patch.py',
    'agents/result_validation.py', 'agents/state_io.py', 'agents/goal_contract.py',
    'agents/output_contract.py',
)


from .plot_dependencies import PLOT_IMPLEMENTATIONS

IMPLEMENTATIONS = {
    'ml_active_learning_plan': ['agents/registry.py','agents/ml_templates.py'],
    'ml_template_plan': ['agents/registry.py','agents/ml_templates.py'],

    'read_scientific_document':['agents/scientific_documents.py','agents/scientific_document_worker.py','agents/workspace.py',
        'agents/document_source_facts.py','agents/literature_report_bridge.py'],
    'download_scientific_file':['agents/scientific_documents.py','agents/scientific_http.py','agents/workspace.py'],
    'plot_result_code': list(PLOT_IMPLEMENTATIONS),

    'analyze_gcmc_screening': ['agents/adsorption_summary.py', 'agents/workload_budget.py', 'agents/workspace.py', 'agents/slurm.py'],
    'run_gcmc_isotherm': ['agents/slurm.py', 'agents/raspa_assets.py', 'agents/raspa_cif_normalization.py', 'agents/gcmc_geometry.py'],
    'run_gcmc_batch': ['agents/slurm.py', 'agents/raspa_assets.py', 'agents/raspa_cif_normalization.py', 'agents/gcmc_geometry.py', 'agents/gcmc_workload.py', 'agents/workload_budget.py'],
    'generate_structure': ['pormake_generate_topo.py', 'tools/stage_pormake_outputs.py'],
    'run_pacman_charge': ['tools/pacmof_batch.py'],
    'run_cdft': ['tools/cdft/cDFT_Initialization/data_input.py',
                 'tools/cdft/cDFT_Initialization/cdft_submit.py',
                 'tools/cdft/cDFT_Initialization/cdft_scratch.py',
                 'tools/cdft/cDFT_Initialization/collect_cdft_results.py', *CDFT_RECOVERY_DEPENDENCIES],
    'run_pore_analysis': ['tools/zeopp.py', 'bimem_agent/paths.py', 'agents/installations.py', 'agents/registry.py'],
    'run_md_optimize': ['tools/md_runtime.py'],
    'run_string_tst': ['tools/run_string_tst_batch.py'],
    'run_xtb_optimize': [],
    'run_external_potential': [],
}


def cdft_installation_files(project_root, arguments):
    """Public adapters execute; operator-owned installation supplies FF data."""
    from bimem_agent import paths
    paths.load_runtime_config.cache_clear()
    software = paths.get_runtime_section('software')
    adapter = Path(project_root) / 'tools/cdft/cDFT_Initialization'
    assets = paths.get_config_path('software', 'cdft_init_dir') if 'cdft_init_dir' in software else adapter
    python = paths.get_config_path('software', 'cdft_python') if 'cdft_python' in software else paths.get_config_path('paths', 'agent_env_python')
    action = arguments.get('action', 'pipeline')
    files = {'python': (python, 'executable')}
    if action in {'inputs', 'pipeline', 'expand'}:
        files['input_adapter'] = (adapter/'data_input.py', 'file')
        files['framework_forcefield'] = (assets/'data_ff_UFF', 'data')
    if action in {'inputs', 'pipeline'}:
        files['coarse_forcefield'] = (assets/'Molecule_FF/data_ff_Coarsening_gas', 'data')
        gases = arguments.get('gases') or [arguments.get('gas', 'CO2')]
        if not isinstance(gases, list) or not all(isinstance(gas, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_+\-]*', gas) for gas in gases):
            raise ValueError('cDFT gases must be explicit safe gas identifiers')
        for gas in gases:
            files['gas_forcefield:' + gas] = (assets/'Molecule_FF/UFF'/f'{gas}.def', 'data')
    if action in {'submit', 'pipeline'}:
        files['submit_adapter'] = (adapter/'cdft_submit.py', 'file')
        files['execution_runner'] = (adapter/'cdft_scratch.py', 'file')
        if (adapter/'cdft_config.json').exists():
            files['operator_execution_config'] = (adapter/'cdft_config.json', 'data')
        executable = paths.get_config_path('software', 'cdft_executable') if 'cdft_executable' in software else assets/'cDFT/DM_cdft'
        files['cdft_executable'] = (executable, 'executable')
    if action in {'submit', 'pipeline', 'collect'}:
        files['collect_adapter'] = (adapter/'collect_cdft_results.py', 'file')
    return files


def installation_file_available(path, kind='file'):
    target = Path(path) if path is not None else None
    available = bool(target and (target.is_dir() if kind == 'directory' else target.is_file()))
    if kind == 'executable':
        available = available and os.access(target, os.X_OK)
    if kind == 'data':
        available = available and target.stat().st_size > 0
    return bool(available)


def _gcmc_asset_selectors(tool, arguments):
    from .slurm import _gas_preset, is_supported_gas
    gases = arguments.get('gases') if tool == 'run_gcmc_batch' else [arguments.get('gas')]
    if (not isinstance(gases, list) or not 1 <= len(gases) <= 64
            or not all(isinstance(g, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_+\-]*', g) for g in gases)):
        raise ValueError('GCMC verification requires explicit safe gas identifiers')
    selected = []
    for gas in gases:
        preset = _gas_preset(gas.upper())
        if not is_supported_gas(gas):
            raise ValueError('GCMC gas has no reviewed molecule preset')
        ff = arguments.get('force_field') or preset['forcefield']
        definition = arguments.get('guest_ff') or preset['molecule_definition']
        if not all(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_+\-]+', value)
                   for value in (ff, definition)):
            raise ValueError('GCMC force field and molecule definition must be named assets')
        selected.append({'gas': gas.upper(), 'forcefield': ff,
                         'molecule_name': preset['molecule_name'], 'molecule_def': definition})
    return selected


def _zeopp_installation():
    """Read the exact SDK resolver used by submission; never run network."""
    from .registry import _load_sdk_zeopp
    adapter=_load_sdk_zeopp()
    resolver=getattr(adapter,'resolve_zeopp',None)
    resolution=resolver()if callable(resolver)else {'executable':adapter.find_zeopp(),'source':'sdk:find_zeopp','configuration_files':[]}
    if not isinstance(resolution,dict):raise ValueError('Zeo resolver returned invalid installation facts')
    value=resolution.get('executable')
    if value is not None and (not isinstance(value,(str,os.PathLike))or not Path(value).is_absolute()):
        raise ValueError('Zeo installation resolver must provide an absolute executable')
    return resolution


def installation_profile(tool, arguments):
    """Only selectors that change the installation being verified."""
    if tool in {'run_gcmc_isotherm', 'run_gcmc_batch'}:
        return {'assets': _gcmc_asset_selectors(tool, arguments)}
    if tool == 'generate_structure':
        family = str(arguments.get('material_type', 'MOF')).upper()
        return {'family': 'HOF' if 'HOF' in family else 'COF' if 'COF' in family else 'MOF'}
    if tool == 'run_pacman_charge':
        return {'method': arguments.get('method', 'pacmof')}
    if tool == 'run_cdft':
        return {'requires_engine': arguments.get('action', 'pipeline') in {'pipeline', 'submit'}}
    return {}


def changed_executor(entry, project_root):
    current = executor_fingerprint(entry['tool'], project_root)
    previous = entry.get('runtime_fingerprint')
    if previous:
        return previous != current
    for name in ['agents/registry.py'] + IMPLEMENTATIONS.get(entry['tool'], []):
        file = Path(project_root) / name
        if file.is_file() and file.stat().st_mtime > entry.get('failed_at', entry.get('started_at', float('inf'))):
            return True
    return False


def executor_fingerprint(tool, project_root):
    root = Path(project_root)
    files = [root / name for name in ('agents/registry.py', 'agents/resource_policy.py',
             'agents/resource_review.py', 'agents/node_inventory.py', 'agents/slurm.py', 'env/node_policy.json')] + [root / p for p in IMPLEMENTATIONS.get(tool, [])]
    files.append(Path(os.environ['BIMEM_RUNTIME_CONFIG']).expanduser()
                 if os.environ.get('BIMEM_RUNTIME_CONFIG') else root / 'config/runtime.json')
    files.append(root / 'env/tools.json')
    if tool == 'run_pore_analysis':
        resolution=_zeopp_installation()
        if resolution.get('executable'):files.append(Path(resolution['executable']))
        files.extend(Path(fact['path'])for fact in resolution.get('configuration_files',[]))
    if tool == 'run_cdft':
        configured = cdft_installation_files(root, {'action': 'pipeline'})
        files.extend(path for path, _kind in configured.values())
        uff = configured['coarse_forcefield'][0].parent/'UFF'
        if uff.is_dir():
            files.extend(sorted(uff.glob('*.def'))[:512])
    if tool in {'run_gcmc_isotherm', 'run_gcmc_batch'}:
        # Asset changes are executor changes even when the CIF/arguments stay
        # fixed. Bounded installed mirror scan; never hydrate scientific runs.
        from .slurm import RASPA_SHARE
        asset_roots = {root/'forcefields/raspa', Path(RASPA_SHARE)}
        assets = set()
        for asset_root in asset_roots:
            assets.update(islice(asset_root.glob('*/*/*.def'), 2049))
        if len(assets) > 2048 or sum(p.stat().st_size for p in assets) > 16_000_000:
            raise ValueError('RASPA asset provenance exceeds bounded verification limit')
        files.extend(sorted(assets))
    facts = {}
    if tool == 'run_pore_analysis':
        executable=resolution.get('executable')
        target=Path(executable) if executable else None
        facts['zeopp:installation_selection']={
            'source':resolution.get('source'), 'path':executable,
            'mode':target.stat().st_mode if target and target.is_file() else None,
            'available':installation_file_available(target,'executable') if target else False,
        }
        # The installed network binary exceeds the generic small-source cap.
        # Bind every byte in a separate bounded streaming proof; never publish
        # a known executor identity after skipping an oversized executable.
        if target and installation_file_available(target,'executable'):
            try:
                with target.open('rb') as stream:
                    before=os.fstat(stream.fileno())
                    if not 0<before.st_size<=64*1024*1024:return None
                    digest=hashlib.sha256();count=0
                    while count<=before.st_size:
                        chunk=stream.read(min(1024*1024,before.st_size+1-count))
                        if not chunk:break
                        digest.update(chunk);count+=len(chunk)
                    after=os.fstat(stream.fileno())
                identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode)
                if count!=before.st_size or identity(before)!=identity(after) or identity(after)!=identity(target.stat()):return None
                facts['zeopp:executable_content']={'path':str(target),'size':count,'sha256':digest.hexdigest()}
            except OSError:return None
    for path in files:
        if path.is_file() and path.stat().st_size <= 16_000_000:
            facts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()


def _inspect_internal_report_capability(tool, arguments, root):
    """Prove loaded Python functions match owned source; never run an author."""
    import importlib,stat,types
    from .executor_identity import code_digest
    profiles={
        'generate_scientific_report':[
            ('agents.report_execution','execute_context_report'),
            ('agents.report_authoring','build_result_packet'),
            ('agents.report_authoring','render_validated_draft'),
            ('agents.report_source_metadata','metadata_tables'),
            ('agents.literature_report_bridge','excerpt_block')],
        'generate_result_figures':[
            ('agents.report_execution','execute_context_report'),
            ('agents.figure_authoring','author_figures'),
            ('agents.plot_sandbox_worker','native_png_output_contract')],
        'analyze_gcmc_screening':[
            ('agents.registry','_exec_analyze_gcmc_screening'),
            ('agents.adsorption_summary','collect_adsorption_summary')],
        'validate_framework_charges':[
            ('agents.registry','_exec_validate_framework_charges'),
            ('agents.charge_contract','validate_framework_charges')],
        'read_scientific_document':[
            ('agents.scientific_documents','read_scientific_document'),
            ('agents.literature_report_bridge','excerpt_block')],
        'download_scientific_file':[
            ('agents.scientific_documents','download_scientific_file')],
    }
    checks=[]
    try:
        for module_name,function_name in profiles[tool]:
            module=importlib.import_module(module_name)
            expected_path=Path(root)/Path(*module_name.split('.')).with_suffix('.py')
            actual=Path(module.__file__)
            if actual.is_symlink() or actual.resolve()!=expected_path.resolve():raise ValueError('internal source outside SDK root')
            fd=os.open(actual,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            try:
                before=os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=4*1024*1024:raise ValueError('internal source is not bounded regular data')
                raw=b''
                while len(raw)<=before.st_size:
                    chunk=os.read(fd,min(65536,before.st_size+1-len(raw)))
                    if not chunk:break
                    raw+=chunk
                after=os.fstat(fd);linked=os.stat(actual,follow_symlinks=False)
                identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
                if len(raw)!=before.st_size or identity(before)!=identity(after) or identity(after)!=identity(linked):raise ValueError('internal source changed while reading')
            finally:os.close(fd)
            compiled=compile(raw,str(actual),'exec',dont_inherit=True)
            expected=next((x for x in compiled.co_consts if isinstance(x,types.CodeType) and x.co_name==function_name),None)
            function=getattr(module,function_name,None)
            if expected is None or not callable(function) or not hasattr(function,'__code__'):raise ValueError('internal function unavailable')
            loaded=code_digest(function.__code__);source=code_digest(expected)
            checks.append({'name':module_name+'.'+function_name,'source_sha256':hashlib.sha256(raw).hexdigest(),
                'loaded_matches_source':loaded==source,'available':loaded==source})
        ok=bool(checks) and all(item['available']for item in checks)
        return {'ok':ok,'read_only':True,'verified_tool':tool,'checked_arguments':arguments,
            'checks':checks,'kind':'internal_python_source_identity','execution_authorized':False,
            'scientific_acceptance':False,'provider_connectivity_verified':False,'external_dependencies_verified':False,
            'scope':'Loaded Python capability matches current local source only; external dependencies, provider connectivity, output validity and scientific claims remain unverified.',
            **({} if ok else {'error':'loaded internal capability differs from current source'})}
    except (OSError,ValueError,TypeError,KeyError,ImportError,AttributeError,SyntaxError) as error:
        return {'ok':False,'read_only':True,'verified_tool':tool,'checked_arguments':arguments,
            'checks':checks,'kind':'internal_python_source_identity','execution_authorized':False,
            'scientific_acceptance':False,'provider_connectivity_verified':False,'external_dependencies_verified':False,
            'error':'internal capability identity could not be established','error_type':type(error).__name__}


def inspect_installation(tool, arguments, config):
    from bimem_agent import paths
    paths.load_runtime_config.cache_clear()
    software = paths.get_runtime_section('software')
    root = Path(config.project_root)
    if tool in {'generate_scientific_report','generate_result_figures','analyze_gcmc_screening',
                'validate_framework_charges','read_scientific_document','download_scientific_file'}:
        return _inspect_internal_report_capability(tool,arguments,root)
    checks = []
    def configured(key, fallback=None):
        return paths.get_config_path('software', key) if key in software else fallback
    def check(name, path, kind='file'):
        target = Path(path) if path is not None else None
        available = installation_file_available(target, kind)
        checks.append({'name': name, 'path': str(target) if target else None,
                       'kind': kind, 'available': bool(available)})
    python = paths.get_config_path('paths', 'agent_env_python')
    asset_manifests = []
    if tool in {'run_gcmc_isotherm', 'run_gcmc_batch'}:
        check('raspa_executable', Path(config.raspa_path)/'bin/simulate', 'executable')
        try:
            from .slurm import _raspa_composition
            for selector in _gcmc_asset_selectors(tool, arguments):
                composed = _raspa_composition(selector['forcefield'], selector['molecule_name'], selector['molecule_def'])
                manifest = composed.manifest
                if (not manifest.get('sources') or not manifest.get('file_hashes')
                        or set(manifest['file_hashes']) != set(composed.files)
                        or any(hashlib.sha256(raw).hexdigest() != manifest['file_hashes'][name]
                               for name, raw in composed.files.items())):
                    raise ValueError('RASPA composition manifest does not match verified asset bytes')
                for fact in manifest['sources'].values():
                    source = Path(fact['path'])
                    if (not source.is_file() or source.stat().st_size != fact['size']
                            or source.stat().st_size > 16_000_000
                            or hashlib.sha256(source.read_bytes()).hexdigest() != fact['sha256']):
                        raise ValueError('RASPA composition source changed during verification')
                asset_manifests.append(manifest)
            checks.append({'name': 'raspa_asset_closure', 'kind': 'data', 'available': True})
        except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
            checks.append({'name': 'raspa_asset_closure', 'kind': 'data', 'available': False,
                           'error': str(error)[:2000]})
    elif tool == 'generate_structure':
        check('python', config.pormake_python, 'executable')
        database = configured('pormake_database_dir', paths.LEGACY_ROOT.parent / 'pormake_db')
        family = str(arguments.get('material_type', 'MOF')).upper()
        subdir = 'bbs' if 'HOF' in family else 'bbs_COF' if 'COF' in family else 'bbs_MOF'
        check('building_blocks', database / subdir, 'directory')
    elif tool == 'run_pacman_charge':
        if arguments.get('method', 'pacmof') == 'pacman':
            check('python', configured('pacman_python', python), 'executable')
            directory = configured('pacman_dir')
            check('charge_script', directory / 'pmcharge.py' if directory else None)
        else:
            check('python', configured('pacmof_python', python), 'executable')
    elif tool == 'run_cdft':
        for name, (path, kind) in cdft_installation_files(root, arguments).items():
            check(name, path, kind)
    elif tool == 'run_pore_analysis':
        try:
            resolution=_zeopp_installation()
            check('zeopp_executable',resolution.get('executable'),'executable')
            if checks[-1]['available'] and Path(resolution['executable']).stat().st_size == 0:
                checks[-1].update(available=False,error='Zeo executable is empty')
        except (OSError,ValueError,TypeError,KeyError,ImportError) as error:
            checks.append({'name':'zeopp_executable','kind':'executable','available':False,
                'error':str(error)[:2000]})
    elif tool == 'run_md_optimize':
        for key in ('lammps_executable', 'mpi_executable', 'lammps_interface_executable'):
            check(key, configured(key), 'executable')
    elif tool == 'run_string_tst':
        check('string_executable', configured('string_tst_executable'), 'executable')
        check('input_generator', configured('string_tst_generate_script'))
    elif tool == 'run_xtb_optimize':
        check('xtb_executable', configured('xtb_executable'), 'executable')
    elif tool == 'run_external_potential':
        for key in ('tustrast_generate_script', 'tustrast_input_param', 'tustrast_executable'):
            check(key, configured(key), 'executable' if key.endswith('executable') else 'file')
    else:
        return {'ok': False, 'error': 'No installation profile registered for this tool', 'verified_tool': tool}
    implementations = []
    for relative in IMPLEMENTATIONS.get(tool, []):
        file = root / relative
        check(relative, file)
        if file.is_file():
            raw = file.read_bytes()
            valid = True
            try:
                compile(raw, str(file), 'exec')
            except (SyntaxError, ValueError):
                valid = False
            checks.append({'name': relative + ':syntax', 'available': valid, 'kind': 'syntax'})
            implementations.append({'path': str(file), 'sha256': hashlib.sha256(raw).hexdigest(),
                                    'modified_at': file.stat().st_mtime})
    runtime_fingerprint=(None if tool=='run_pore_analysis'and not all(item['available']for item in checks)
        else executor_fingerprint(tool,root))
    if tool=='run_pore_analysis'and all(item['available']for item in checks)and runtime_fingerprint is None:
        checks.append({'name':'zeopp_executable_identity','kind':'bounded_content_proof','available':False,
            'error':'Zeo executable content cannot be stably verified within the 64 MiB limit'})
    from .runtime_health_probe import inspect_runtime_health
    runtime_health = inspect_runtime_health(tool, arguments, config)
    return {'runtime_health_probe': runtime_health,
            'ok': bool(checks) and all(item['available'] for item in checks),
            'read_only': True, 'verified_tool': tool, 'checked_arguments': arguments,
            'checks': checks, 'implementation_files': implementations, 'asset_manifests': asset_manifests,
            'runtime_fingerprint': runtime_fingerprint,
            'scope': 'local installation and adapter syntax; compute-node execution and scientific validity are not established'}


_DEPENDENCY_UNKNOWN = object()


def _dependency_condition(expression, action, layout_keys):
    """Evaluate only literal action/layout predicates; never execute user code."""
    import ast
    unknown = _DEPENDENCY_UNKNOWN
    if isinstance(expression, ast.Constant): return expression.value
    if isinstance(expression, ast.Name):
        return action if expression.id == 'action' else layout_keys if expression.id == 'files' else unknown
    if isinstance(expression, (ast.Set, ast.List, ast.Tuple)):
        values = [_dependency_condition(item, action, layout_keys) for item in expression.elts]
        return unknown if any(value is unknown for value in values) else tuple(values)
    if (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute)
            and isinstance(expression.func.value, ast.Name) and expression.func.value.id == 'arguments'
            and expression.func.attr == 'get' and expression.args
            and isinstance(expression.args[0], ast.Constant) and expression.args[0].value == 'action'):
        return action
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
        value = _dependency_condition(expression.operand, action, layout_keys)
        return unknown if value is unknown else not value
    if isinstance(expression, ast.BoolOp):
        values = [_dependency_condition(value, action, layout_keys) for value in expression.values]
        if isinstance(expression.op, ast.And):
            if any(value is not unknown and not value for value in values): return False
            return unknown if any(value is unknown for value in values) else True
        if any(value is not unknown and bool(value) for value in values): return True
        return unknown if any(value is unknown for value in values) else False
    if isinstance(expression, ast.Compare) and len(expression.ops) == 1:
        left = _dependency_condition(expression.left, action, layout_keys)
        right = _dependency_condition(expression.comparators[0], action, layout_keys)
        if left is unknown or right is unknown: return unknown
        op = expression.ops[0]
        if isinstance(op, ast.Eq): return left == right
        if isinstance(op, ast.NotEq): return left != right
        if isinstance(op, ast.In): return left in right
        if isinstance(op, ast.NotIn): return left not in right
    return unknown


def _dependency_code(path, names, action, layout_keys):
    """Hash declared reachable same-module functions after safe specialization.

    Unknown conditions keep both branches. Direct same-module helper calls are
    followed recursively; dynamic calls cannot claim a stable cache identity.
    """
    import ast
    import copy
    raw = Path(path).read_bytes()
    if len(raw) > 2_000_000: raise ValueError('dependency source exceeds bound')
    module = ast.parse(raw, filename=str(path))
    functions = {node.name: node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    constants = {target.id: node for node in module.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)}
    class Specialize(ast.NodeTransformer):
        def visit_If(self, node):
            value = _dependency_condition(node.test, action, layout_keys)
            if value is _DEPENDENCY_UNKNOWN: return self.generic_visit(node)
            result = []
            for statement in node.body if value else node.orelse:
                updated = self.visit(statement)
                result.extend(updated if isinstance(updated, list) else [updated] if updated is not None else [])
            return result
    pending, found = list(names), {}
    while pending:
        name = pending.pop()
        if name in found: continue
        if name not in functions: raise ValueError('declared dependency function missing: ' + name)
        original = functions[name]
        for item in ast.walk(original):
            if (isinstance(item, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
                    and isinstance(item.target, ast.Name) and item.target.id == 'action'):
                raise ValueError('action rebinding requires conservative fallback')
            if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'action' for target in item.targets):
                value = item.value
                if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute)
                        or not isinstance(value.func.value, ast.Name) or value.func.value.id not in {'p', 'arguments'}
                        or value.func.attr != 'get' or not value.args
                        or not isinstance(value.args[0], ast.Constant) or value.args[0].value != 'action'):
                    raise ValueError('action binding is dynamic; conservative fallback required')
            if (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                    and isinstance(item.func.value, ast.Name) and item.func.value.id in {'p', 'arguments', 'files'}
                    and item.func.attr in {'pop', 'update', 'clear', 'setdefault', '__setitem__'}):
                raise ValueError('mutable dependency selector requires conservative fallback')
        node = Specialize().visit(copy.deepcopy(original))
        # Comments, locations and function documentation cannot change execution.
        for item in ast.walk(node):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.body and isinstance(item.body[0], ast.Expr) and isinstance(item.body[0].value, ast.Constant) and isinstance(item.body[0].value.value, str):
                item.body = item.body[1:]
            if isinstance(item, ast.Call) and (isinstance(item.func, (ast.Subscript, ast.Lambda))
                    or isinstance(item.func, ast.Name) and item.func.id in {'eval', 'exec', 'globals', 'locals'}):
                raise ValueError('dynamic dependency dispatch requires conservative fallback')
        found[name] = ast.dump(node, include_attributes=False)
        used = {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}
        pending.extend(used.intersection(functions).difference(found))
        for constant in used.intersection(constants):
            found['constant:' + constant] = ast.dump(constants[constant], include_attributes=False)
    return found


def dependency_fingerprint(tool_name, arguments, project_root, *, executor=None, tool_version=None):
    """Versioned completed-result cache identity, separate from retry permission.

    Only the three explicit native cDFT actions have audited dependencies.
    Missing/unsupported facts return None for the caller's conservative policy.
    This never migrates a previous digest or authorizes execution/retry.
    """
    import inspect
    from itertools import islice
    from .recovery import input_fingerprint
    if tool_name == 'plot_result_code':
        # Unlocked arbitrary imports/fonts/randomness cannot authorize image reuse.
        return None
    if tool_name == 'analyze_gcmc_screening':
        try:
            root = Path(project_root)
            if executor is not None and (getattr(executor, '__name__', None) != '_exec_analyze_gcmc_screening'
                    or Path(inspect.getsourcefile(executor)).resolve() != (root/'agents/registry.py').resolve()):
                return None
            from .adsorption_summary import summary_dependency_identity
            observed = summary_dependency_identity(arguments, root)
            hashes = {}
            for name in ('agents/registry.py', 'agents/installations.py', *IMPLEMENTATIONS[tool_name]):
                path = root/name
                if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 16*1024*1024:
                    return None
                # The same stable NOFOLLOW reader used for native evidence.
                from .adsorption_summary import _read
                _read(root, path, hashes, [16*1024*1024], 16*1024*1024)
            payload = {'schema': 'adsorption-summary-deps-v1', 'observed': observed,
                       'source_files': hashes, 'tool_version': tool_version}
            return 'adsorption-summary-deps-v1:' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            return None
    if tool_name != 'run_cdft' or arguments.get('action') not in {'inputs', 'submit', 'collect'}:
        return None
    root, action = Path(project_root), arguments['action']
    try:
        registry_path = root/'agents/registry.py'
        if executor is not None and (getattr(executor, '__name__', None) != '_exec_cdft'
                or Path(inspect.getsourcefile(executor)).resolve() != registry_path.resolve()):
            return None
        layout = cdft_installation_files(root, arguments)
        required = {'inputs': {'python', 'input_adapter', 'framework_forcefield', 'coarse_forcefield'},
                    'submit': {'python', 'submit_adapter', 'execution_runner', 'collect_adapter', 'cdft_executable'},
                    'collect': {'python', 'collect_adapter'}}[action]
        if not required.issubset(layout): return None
        code = {'registry': _dependency_code(registry_path, ['_exec_cdft'], action, set(layout)),
                'installation_layout': _dependency_code(root/'agents/installations.py',
                    ['cdft_installation_files', 'installation_file_available'], action, set(layout)),
                'path_resolution': _dependency_code(root/'bimem_agent/paths.py',
                    ['get_config_path', 'get_runtime_section'], action, set(layout)),
                'receipt_parser': _dependency_code(root/'agents/recovery.py', ['result_object'], action, set(layout)),
                'workspace_paths': _dependency_code(root/'agents/workspace.py', ['session_dir'], action, set(layout))}
        files = {str(path.resolve()): Path(path) for path, _kind in layout.values()}
        timing = root/'agents/native_timing.py'
        files[str(timing.resolve())] = timing
        child_entry = root/'bimem_agent/python_child.py'
        files[str(child_entry.resolve())] = child_entry
        for relative in ('agents/__init__.py', 'agents/placement_budget.py', 'agents/node_inventory.py'):
            dependency = root/relative
            files[str(dependency.resolve())] = dependency
        if action in {'inputs', 'submit'}:
            grid_guard = root/'agents/cdft_grid_guard.py'
            files[str(grid_guard.resolve())] = grid_guard
        membership = {}
        if action == 'inputs':
            source = Path(arguments.get('cif_dir') or arguments.get('cif_path') or '')
            source = (source if source.is_absolute() else root/source).resolve(strict=True)
            candidates = []
            if source.is_file():
                if source.suffix != '.cif': return None
                candidates = [source]
                source_root = source.parent
            elif source.is_dir():
                source_root, pending, seen = source, [source], 0
                # Walk only the same recursive CIF inputs as the native reader,
                # bounding every entry, not merely the matching file count.
                while pending:
                    directory = pending.pop()
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            seen += 1
                            if seen > 512: return None
                            path = Path(entry.path)
                            if entry.name.endswith('.cif'):
                                if not path.is_file(): return None
                                candidates.append(path)
                            if entry.is_dir(follow_symlinks=False): pending.append(path)
            else: return None
            if not candidates: return None
            stems = set()
            for path in sorted(candidates):
                if not path.resolve().is_relative_to(source_root) or path.stem in stems: return None
                stems.add(path.stem)
                membership[str(path.relative_to(source_root))] = str(path.resolve())
                files[str(path.resolve())] = path
            # The native immutable-input publisher loads this reader by file.
            reader = root/'agents/workload_budget.py'
            files[str(reader.resolve())] = reader
        if action == 'submit':
            for relative in CDFT_RECOVERY_DEPENDENCIES:
                dependency=root/relative
                files[str(dependency.resolve())]=dependency
            if arguments.get('partial_recovery_manifest'):
                manifest=Path(arguments['partial_recovery_manifest'])
                manifest=manifest if manifest.is_absolute() else root/manifest
                if (not manifest.is_file() or manifest.is_symlink()
                        or manifest.resolve()!=manifest.absolute() or manifest.stat().st_size>16*1024*1024):
                    return None
                files[str(manifest.resolve())]=manifest
            for relative in ('agents/node_inventory.py', 'agents/watch_context.py', 'agents/job_watch.py', 'agents/task_line.py'):
                path = root/relative
                files[str(path.resolve())] = path
            native_config = root/'tools/cdft/cDFT_Initialization/cdft_config.json'
            code['native_submit_config_present'] = native_config.exists()
            if native_config.exists(): files[str(native_config.resolve())] = native_config
        if action == 'collect':
            for relative in ('agents/cdft_result_facts.py', 'agents/workload_context.py', 'agents/workload_budget.py'):
                dependency = root/relative; files[str(dependency.resolve())] = dependency
            # Without an explicit directory, executor chooses "latest" at run
            # time; no cache identity can prove which native results it reads.
            if not arguments.get('job_work_dir'): return None
            work = Path(arguments['job_work_dir'])
            work = work if work.is_absolute() else root/work
            native = work/'input_dir'
            if not native.is_dir(): return None
            materials = list(islice(native.iterdir(), 513))
            if len(materials) > 512: return None
            membership['run_entries'] = sorted((item.name, item.is_dir()) for item in materials)
            optional = [work/'results.csv']
            input_roots = {'batch_inputs': work/'input'}
            if arguments.get('input_dir'):
                declared = Path(arguments['input_dir'])
                input_roots['declared_inputs'] = declared if declared.is_absolute() else root/declared
            for label, directory in input_roots.items():
                membership[label] = None
                if directory.exists():
                    if not directory.is_dir() or directory.is_symlink(): return None
                    entries = list(islice(directory.iterdir(), 513))
                    if len(entries) > 512: return None
                    membership[label] = sorted((item.name, item.is_dir()) for item in entries)
                    optional.extend(item for item in entries if item.name.endswith('.dat'))
            for material in materials:
                if material.is_dir():
                    if not material.resolve().is_relative_to(native.resolve()) or material.is_symlink(): return None
                    optional.extend(material/name for name in ('input.dat', 'solver.log', 'stderr.log', 'execution.json'))
                    for name in ('output.dat', 'hr.dat'):
                        path = material/name
                        if not path.is_file() or not path.resolve().is_relative_to(native.resolve()): return None
                        membership[str(path.relative_to(native))] = str(path.resolve())
                        files[str(path.resolve())] = path
            for path in optional:
                membership['quality:' + str(path)] = None
                if path.exists() or path.is_symlink():
                    if not path.is_file() or path.is_symlink(): return None
                    membership['quality:' + str(path)] = str(path.resolve())
                    files[str(path.resolve())] = path
        hashes, total = {}, 0
        for name, path in sorted(files.items()):
            size = path.stat().st_size
            total += size
            if not path.is_file() or size > 128*1024*1024 or total > 256*1024*1024: return None
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
            hashes[name] = digest.hexdigest()
        payload = {'schema': 'cdft-deps-v1', 'action': action, 'code': code, 'files': hashes,
                   'inputs': input_fingerprint(arguments, root), 'native_membership': membership, 'tool_version': tool_version}
        return 'cdft-deps-v1:' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    except (OSError, ValueError, TypeError, KeyError, SyntaxError):
        return None
