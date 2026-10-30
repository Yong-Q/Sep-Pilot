"""Source-bound native MD log/CIF inputs, without directory discovery.

Original result JSON stays authoritative and unmodified. Older adapters declare
their solver log through a precise stage-summary protocol; modern adapters may
declare an explicit log path/hash. A path prediction is never a historical SHA.
"""
import hashlib
import json
from pathlib import Path
import re


LEGACY_LOG_DECLARATION = 'solver.log stage-delimited native Loop time summaries'


def _attach_executed_script(root, work, value, summary, sealed_inputs, *, read, budget, file_facts):
    """Expose only adapter-attested exact executed input, never guess legacy paths."""
    receipt=value.get('execution_input')
    if receipt is None:
        summary.setdefault('comparison_limits', []).append(
            'Historical native MD receipt does not attest the executed input script; control settings cannot be established from a merely present file.')
        return None
    if (not isinstance(receipt,dict) or receipt.get('verification')!='sha256_before_and_after_solver'
            or receipt.get('scientific_acceptance') is not False
            or not isinstance(receipt.get('sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',receipt['sha256'])
            or type(receipt.get('bytes')) is not int or not 0<receipt['bytes']<=1024*1024):
        raise ValueError('native executed MD input receipt is invalid')
    path=_owned_path(root,receipt.get('path'),branch=work)
    raw,fact=read(root,path,budget)
    if fact['sha256']!=receipt['sha256'] or fact['size']!=receipt['bytes']:
        raise ValueError('native executed MD input differs from its receipt')
    if str(path) in sealed_inputs and sealed_inputs[str(path)]!=fact['sha256']:
        raise ValueError('inherited native MD input SHA changed')
    from .plot_input_adapter import preview, text_format_hints
    identifier='MDI'+summary['id']+hashlib.sha256(fact['path'].encode()).hexdigest()[:12]
    summary.setdefault('text_documents',[]).append({'id':identifier,'path':fact['path'],
        'sha256':fact['sha256'],'format':'text','preview':preview(raw),
        'format_hints':text_format_hints(raw),'historical_sha_verified':True,
        'observation_scope':'Native adapter recorded matching input hashes immediately before and after the solver; no independent scientific acceptance.'})
    file_facts.append(fact)
    return str(path)


def _object(value):
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, dict): raise ValueError('native MD receipt is not an object')
    return result


def _owned_path(root, value, *, branch=None):
    if not isinstance(value, str) or not value or '..' in Path(value).parts:
        raise ValueError('native MD path is not a declared absolute owned path')
    path = Path(value)
    if not path.is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError('native MD path escapes its owner')
    if branch is not None and (not path.is_relative_to(branch) or path == branch):
        raise ValueError('native MD result escapes its declared calculation branch')
    return path


def _nodes(source, native_result, state):
    """Outer packet builder has already witnessed all authored ancestors."""
    queue = [(source, native_result)]; seen = set(); result = []
    while queue:
        node, receipt = queue.pop()
        contract = node.get('contract') or {}; name = contract.get('step_id')
        if not isinstance(name, str) or name in seen: continue
        seen.add(name)
        if len(seen) > 32: raise ValueError('native MD ancestry exceeds bounded source scope')
        if node.get('status') != 'succeeded': raise ValueError('native MD source ancestry is not completed')
        tool = contract.get('tool')
        if tool == 'run_md_optimize':
            result.append((node, receipt)); continue
        if tool not in {'generate_scientific_report', 'generate_result_figures'}: continue
        declared = (contract.get('arguments') or {}).get('source_steps') or []
        if receipt.get('source_steps') != declared or not set(declared) <= set(contract.get('depends_on') or []):
            raise ValueError('native MD authored ancestry differs from its declared sources')
        for step in declared:
            child = (state.get('nodes') or {}).get(step)
            if not isinstance(child, dict): raise ValueError('native MD source ancestry is missing')
            if (child.get('contract') or {}).get('tool') in {
                    'run_md_optimize', 'generate_scientific_report', 'generate_result_figures'}:
                queue.append((child, _object(child.get('result'))))
    return result


def _accepted_json(source, path, fact):
    for name, artifact in (source.get('artifacts') or {}).items():
        if not isinstance(artifact, dict): continue
        files = artifact.get('files')
        if files is None:
            if name == str(path) and artifact.get('size') == fact['size'] and artifact.get('mtime_ns') == fact['mtime_ns']:
                if artifact.get('sha256') not in (None, fact['sha256']): return False
                return True
        else:
            for item in files:
                if isinstance(item, list) and len(item) >= 3 and item[:3] == [str(path), fact['size'], fact['mtime_ns']]:
                    return True
    return False


def _native_dossier(root, source, work, *, read, budget, file_facts):
    dossier = source.get('validation_dossier')
    if not dossier: return {}
    if (not isinstance(dossier, dict) or dossier.get('step_id') != source['contract'].get('step_id')
            or dossier.get('attempt_id') != source.get('attempt_id')
            or dossier.get('tool') != 'run_md_optimize' or dossier.get('job_ids') != source.get('job_ids')
            or dossier.get('contract') != source['contract']):
        raise ValueError('native MD validation observations belong to a different source attempt')
    identifier = dossier.get('evidence_call_id')
    if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', identifier):
        raise ValueError('native MD validation observations lack native evidence')
    expected = Path(root) / 'evidence' / (identifier + '.json')
    if dossier.get('evidence_path') != str(expected):
        raise ValueError('native MD validation evidence escapes its owner')
    raw, fact = read(root, expected, budget); file_facts.append(fact)
    call = _object(raw.decode()); observed = _object(call.get('result'))
    native = (dossier.get('method_facts') or {}).get('md_native') or {}
    # UI/runtime projections may omit inventory.files and verification_owner.
    # Neither omission may alter the source identities or a single native fact.
    protected = ('schema_version', 'workflow_id', 'plan_version', 'step_id', 'attempt_id',
        'job_ids', 'tool', 'contract', 'generated_at', 'evidence_call_id', 'evidence_path', 'method_facts')
    if (call.get('call_id') != identifier or call.get('tool') != 'inspect_workflow_result'
            or call.get('failed') is not False
            or call.get('params') != {'path': str(work), 'step_id': source['contract']['step_id']}
            or call.get('time') != dossier.get('generated_at')
            or any(observed.get(field) != dossier.get(field) for field in protected)
            or native.get('result_dir') != str(work)):
        raise ValueError('native MD dossier differs from its actual native inspection record')
    return native


def _input_identity(root, source, arguments, path, fact, *, read, budget, file_facts):
    key, attempt = source.get('recovery_key'), source.get('attempt_id')
    ledger = Path(root) / 'recovery_state.json'
    if not key or not attempt or not ledger.exists(): return False
    raw, witness = read(root, ledger, budget); file_facts.append(witness)
    entry = _object(raw.decode()).get(key)
    if not isinstance(entry, dict): return False
    if (entry.get('key') != key or entry.get('tool') != 'run_md_optimize'
            or entry.get('attempt_id') != attempt or entry.get('params') != arguments
            or entry.get('job_ids') != source.get('job_ids')):
        raise ValueError('native MD original input ledger belongs to a different attempt')
    # Exact existing RecoveryGate input_fingerprint protocol, using the already
    # NOFOLLOW-read CIF hash rather than reopening any model-supplied path.
    other = ('cif', 'cif_dir', 'input_path', 'input_dir', 'data_csv', 'path', 'work_dir')
    if any(arguments.get(field) for field in other):
        raise ValueError('unsupported additional MD input fingerprint paths')
    ignored = {'output', 'output_dir', 'output_csv', 'job_work_dir', 'job_name', 'work_dir', 'resource_review_id'}
    params = {key: value for key, value in arguments.items() if key not in ignored}
    blob = json.dumps({'params': params, 'files': {str(path): fact['sha256']}}, sort_keys=True, default=str)
    if hashlib.sha256(blob.encode()).hexdigest() != entry.get('fingerprint'):
        raise ValueError('native MD original input SHA no longer matches its submitted fingerprint')
    return True


def _geometry(root, path, expected_sha):
    from .gcmc_geometry import derive_gcmc_geometry
    # Reuse only the strict six-cell-scalar reader; its minimum-image estimate
    # is not used as an MD parameter, scientific result or execution budget.
    geometry = derive_gcmc_geometry(path, 1.0, owned_root=root)
    if not geometry.get('ok') or geometry.get('source_sha256') != expected_sha:
        raise ValueError('native MD CIF cell geometry is invalid or source SHA changed')
    return {'lengths': geometry['lengths'], 'angles': geometry['angles'],
            'length_unit': geometry['length_unit'], 'angle_unit': geometry['angle_unit']}


def attach_md_inputs(root, owner, state, source, native_result, summary, tables, sealed_inputs, *,
                     report_tool, read, budget, file_facts):
    """Extend an actual witnessed JSON-only MD producer into its named sources.

    Returns paths handled here so inherited source lists do not duplicate them.
    Every current byte identity remains in the packet's normal file_facts; no
    files, metadata envelopes, scientific outputs or completion state are made.
    """
    root = Path(root)
    if (root.parent.name, root.name) != (owner.get('username'), owner.get('conv_id')):
        raise PermissionError('native MD input owner differs from the packet owner')
    handled = set()
    from .plot_input_adapter import preview, text_format_hints
    from .md_result_facts import parse_md_thermo, md_thermo_summary
    from .report_native_inputs import append_thermo_tables
    for producer, receipt in _nodes(source, native_result, state):
        limits = summary.get('comparison_limits') or []
        summary['comparison_limits'] = limits if isinstance(limits, list) else [str(limits)]
        arguments = producer.get('effective_arguments') or producer['contract'].get('arguments') or {}
        if receipt.get('submitted') is not True or not re.fullmatch(r'[1-9][0-9]*', str(receipt.get('job_id', ''))):
            raise ValueError('native MD source lacks its real submitted receipt')
        if producer.get('job_ids') != [str(receipt['job_id'])]:
            raise ValueError('native MD source job identity differs from its receipt')
        approved = _owned_path(root, arguments.get('output_dir'))
        work = _owned_path(root, receipt.get('work_dir'))
        if not work.is_relative_to(approved) or receipt.get('output_dir') != str(work):
            raise ValueError('native MD receipt differs from its approved output branch')
        path = _owned_path(root, receipt.get('result_json_path'), branch=work)
        raw, fact = read(root, path, budget); file_facts.append(fact)
        if not _accepted_json(producer, path, fact):
            raise ValueError('native MD result JSON changed since its accepted producer artifact')
        observed = _native_dossier(root, producer, work, read=read, budget=budget, file_facts=file_facts)
        native_artifacts = observed.get('artifacts') or {}
        if len(native_artifacts) > 16: raise ValueError('native MD declared source inventory exceeds its bound')
        json_observations = [item for item in native_artifacts.values()
            if isinstance(item, dict) and item.get('path') == str(path)]
        if json_observations and (len(json_observations) != 1
                or json_observations[0].get('sha256') != fact['sha256']
                or json_observations[0].get('bytes') != fact['size']):
            raise ValueError('native MD result JSON SHA differs from its native validation observation')
        value = _object(raw.decode())
        if value.get('success') is not True or value.get('work_dir') != str(work):
            raise ValueError('native MD result JSON does not attest successful matching output')
        script_path=_attach_executed_script(root,work,value,summary,sealed_inputs,
            read=read,budget=budget,file_facts=file_facts)
        if script_path:handled.add(script_path)
        output = _owned_path(root, value.get('output_cif'), branch=work)
        if receipt.get('expected_structure_path') != str(output):
            raise ValueError('native MD output CIF differs from its submitted expected path')
        native = value.get('thermo_history') or {}
        observed_log = observed.get('solver_log') or {}
        log = observed_log.get('source_path') or native.get('source_log')
        if native.get('source_log') is not None and native['source_log'] != log:
            raise ValueError('native MD log source declarations disagree')
        if log is None and value.get('stage_steps_source') == LEGACY_LOG_DECLARATION:
            log = str(work / 'solver.log')
        if log is not None:
            log_path = _owned_path(root, log, branch=work)
            try:
                log_raw, log_fact = read(root, log_path, budget)
            except FileNotFoundError:
                summary.setdefault('comparison_limits', []).append('The explicitly declared native MD solver log is unavailable; no temperature, integration-step or energy series is invented.')
            else:
                expected = native.get('source_sha256')
                if expected is not None and expected != log_fact['sha256']:
                    raise ValueError('native MD solver log SHA differs from its native result declaration')
                observed_sha = observed_log.get('source_sha256')
                if observed_sha is not None and (observed_sha != log_fact['sha256']
                        or observed_log.get('source_bytes') != log_fact['size']):
                    raise ValueError('native MD solver log SHA differs from its native validation observation')
                if str(log_path) in sealed_inputs and sealed_inputs[str(log_path)] != log_fact['sha256']:
                    raise ValueError('inherited native MD solver log SHA changed')
                file_facts.append(log_fact); handled.add(str(log_path))
                history = parse_md_thermo(log_raw)
                identifier = 'MDL' + summary['id'] + hashlib.sha256(log_fact['path'].encode()).hexdigest()[:12]
                summary.setdefault('text_documents', []).append({'id': identifier, 'path': log_fact['path'],
                    'sha256': log_fact['sha256'], 'format': 'text', 'preview': preview(log_raw),
                    'format_hints': text_format_hints(log_raw), 'native_thermo': md_thermo_summary(history),
                    'observation_scope': 'Current bytes of the exact adapter-declared native log; historical SHA verified only when explicitly recorded.',
                    'historical_sha_verified': expected is not None or observed_sha is not None})
                if report_tool == 'generate_scientific_report':
                    append_thermo_tables(log_raw, log_fact, summary['id'], tables, max_rows=2000)
        else:
            summary.setdefault('comparison_limits', []).append('Native MD receipt does not declare a solver-log source; no filename or native trajectory is guessed.')
        for role, cif in [('configured_input', _owned_path(root, arguments.get('cif_path'))),
                          ('native_output', output)]:
            cif_raw, cif_fact = read(root, cif, budget)
            if str(cif) in sealed_inputs and sealed_inputs[str(cif)] != cif_fact['sha256']:
                raise ValueError('inherited native MD CIF SHA changed')
            expected = value.get('input_cif_sha256' if role == 'configured_input' else 'output_cif_sha256')
            if expected is not None and expected != cif_fact['sha256']:
                raise ValueError('native MD CIF SHA differs from its native result declaration')
            historical = (expected is not None or role == 'configured_input' and _input_identity(
                root, producer, arguments, cif, cif_fact, read=read, budget=budget, file_facts=file_facts))
            geometry = _geometry(root, cif, cif_fact['sha256'])
            file_facts.append(cif_fact); handled.add(str(cif))
            scope = ('Native declared ' + role + ' CIF; exact current bytes. ' +
                ('Submitted input/native recorded SHA is unchanged.' if historical else 'Historical CIF bytes were not sealed by this older producer.'))
            if not historical and scope not in summary['comparison_limits']:
                summary['comparison_limits'].append(scope)
            summary.setdefault('text_documents', []).append({'id': 'MDC' + summary['id'] + hashlib.sha256(cif_fact['path'].encode()).hexdigest()[:12],
                'path': cif_fact['path'], 'sha256': cif_fact['sha256'], 'format': 'text',
                'preview': preview(cif_raw), 'format_hints': {'native_format': 'CIF', 'cell_geometry': geometry,
                    'source_sha256': cif_fact['sha256'], 'role': role, 'historical_sha_verified': bool(historical),
                    'scope': scope}, 'observation_scope': scope})
            if report_tool == 'generate_scientific_report':
                identity = hashlib.sha256((cif_fact['path'] + cif_fact['sha256'] + role).encode()).hexdigest()
                prior = next((item for item in tables if item.get('native_identity_sha256') == identity), None)
                if prior:
                    prior.setdefault('source_ids', [prior['source_id']])
                    if summary['id'] not in prior['source_ids']: prior['source_ids'].append(summary['id'])
                    continue
                columns = ['structure_role', 'a_A', 'b_A', 'c_A', 'alpha_deg', 'beta_deg', 'gamma_deg']
                tables.append({'id': 'T' + str(len(tables)+1), 'source_id': summary['id'],
                    'format': 'native_cif_geometry', 'path': cif_fact['path'], 'sha256': cif_fact['sha256'],
                    'native_identity_sha256': identity, 'columns': columns, 'rows': [dict(zip(columns,
                        [role, *geometry['lengths'], *geometry['angles']]))], 'row_count': 1,
                    'column_units': {**{key: 'angstrom (CIF)' for key in columns[1:4]},
                                     **{key: 'degree (CIF)' for key in columns[4:]}},
                    'native_context': {'observation_scope': scope, 'historical_sha_verified': bool(historical),
                                       'scientific_convergence': 'unknown'},
                    'units': 'CIF lengths in angstrom; CIF angles in degree; single configuration, not a time average.'})
    return handled
