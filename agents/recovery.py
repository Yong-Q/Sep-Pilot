"""Evidence-gated resubmission, independent of prompts and error wording."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
import os
import re

from .state_io import json_transaction

SUBMISSION_TOOLS = frozenset({'ml_active_learning', 'generate_structure', 'run_external_potential', 'build_guest_forcefield', 'run_isotherm_chain', 'run_henry_chain', 'ml_train', 'run_string_tst', 'run_henry', 'run_pore_analysis', 'run_pacman_charge', 'ml_predict', 'run_gcmc_batch', 'submit_job', 'run_gcmc_isotherm', 'run_xtb_optimize', 'run_md_optimize', 'run_ga_optimization'})
FAILURE_STATES = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'REJECTED', 'FATAL'}


def is_submission(tool, params):
    return (tool in SUBMISSION_TOOLS
            or tool == 'run_cdft' and params.get('action') in {'pipeline', 'submit'}
            or tool == 'run_vasp' and params.get('action') in {'submit', 'setup-batch'})


def normalize_tool_result(result):
    """Preserve native MCP transport flags instead of stringifying its model."""
    if isinstance(result,dict):return result
    if type(result)is bool:return {'ok':result,'result':result}
    if isinstance(result,str):
        try:parsed=json.loads(result)
        except (TypeError,ValueError):
            return {'error':result}if result.lstrip().startswith('Error:')else{'result':result}
        return normalize_tool_result(parsed)
    # MCP SDKs return a Pydantic CallToolResult, not a Python dict. This is an
    # envelope conversion, never a tool execution or structuredContent override.
    if type(getattr(result,'isError',None))is bool and hasattr(result,'content'):
        dump=getattr(result,'model_dump',None)
        if callable(dump):
            value=dump(mode='json',by_alias=True)
            if isinstance(value,dict):return value
        return {'isError':result.isError,'content':[
            item.model_dump(mode='json',by_alias=True)if callable(getattr(item,'model_dump',None))else item
            for item in result.content],
            'structuredContent':getattr(result,'structuredContent',None)}
    return {'result':result}


def result_object(result):
    if not isinstance(result,(dict,str))and type(getattr(result,'isError',None))is bool:
        result=normalize_tool_result(result)
    if isinstance(result, dict):
        parsed = result
    else:
        try:
            parsed = json.loads(result)
            if not isinstance(parsed, dict):
                if parsed is False:return {'ok':False,'result':False}
                return {}
        except (TypeError, ValueError):
            return {}
    if isinstance(parsed.get('content'),list)and any(key in parsed for key in ('isError','is_error','structuredContent')):
        inner=parsed.get('structuredContent')
        if not isinstance(inner,dict)and len(parsed['content'])==1:
            block=parsed['content'][0]
            if isinstance(block,dict)and block.get('type')=='text'and isinstance(block.get('text'),str):
                try:inner=json.loads(block['text'])
                except ValueError:pass
        if isinstance(inner,dict):
            # Evidence keeps the full original envelope. This projection feeds
            # typed result bindings without duplicating large source content.
            projected=dict(inner)
            for key in ('isError','is_error'):
                if key in parsed:projected[key]=parsed[key]is True or inner.get(key)is True
            if parsed.get('error'):projected['error']=parsed['error']
            parsed=projected
    # Some cDFT wrappers say submitted=True merely because their helper Python
    # exited 0. The helper's last JSON result is the actual scheduler outcome.
    if parsed.get('submitted') and isinstance(parsed.get('stdout'), str):
        tail = parsed['stdout'][-100000:]
        decoder, candidates = json.JSONDecoder(), []
        for match in list(re.finditer(r'(?m)^\s*\{', tail))[-128:]:
            start = match.end() - 1
            try:
                inner, length = decoder.raw_decode(tail[start:])
                if isinstance(inner, dict) and any(k in inner for k in ('submitted', 'job_id', 'job_ids', 'failed', 'error')):
                    candidates.append((start + length, inner))
            except ValueError:
                continue
        if candidates:
            merged={**parsed, **max(candidates, key=lambda pair: pair[0])[1]}
            for key in ('isError','is_error'):
                if parsed.get(key)is True:merged[key]=True
            return merged
    return parsed


def failure_reason(result, *, diagnostic=False):
    """Structured signals only; never scan arbitrary output for 'FAILED'."""
    if result is False and not diagnostic:return 'tool returned false'
    obj = result_object(result)
    if obj.get('isError')is True or obj.get('is_error')is True:
        return str(obj.get('error')or'MCP/tool transport reported an error')[:4000]
    if diagnostic:
        # An owned inspection can correctly observe a failed job, a missing
        # installation or an invalid physical condition. Actual query failures
        # use the native error/transport channels, not the observed status.
        if obj.get('error'):return str(obj['error'])[:4000]
        if isinstance(result,str)and result.lstrip().startswith('Error:'):return result[:4000]
        return ''
    if obj.get('skipped') or obj.get('deduplicated'):
        return ''
    state = str(obj.get('status') or obj.get('state') or '').upper()
    if obj.get('blocked'):
        # A safe duplicate/defer is not a failed execution. An explicit native
        # error plus executed=False is a real rejected tool call, however; do
        # not hide scope/schema failures from memory and fault supervision.
        waiting = {'WAITING','WAITING_JOBS','WAITING_RESOURCES','NEEDS_RESOURCES',
                   'REVIEW_BUSY','REVIEW_RETRY','AWAITING_APPROVAL','AWAITING_CONFIRMATION',
                   'NEEDS_USER','DEFERRED','PENDING'}
        if state in waiting or obj.get('review_in_progress') is True:
            return ''
        if obj.get('executed') is False and obj.get('error'):
            return str(obj['error'])[:4000]
        return ''
    exit_code = obj.get('exit_code', obj.get('returncode'))
    if (obj.get('failed') is True or obj.get('success') is False
            or obj.get('ok') is False and not diagnostic
            or state in FAILURE_STATES or obj.get('chain_status') == 'failed'
            or state in {'EXECUTION_FAILED','SANDBOX_UNAVAILABLE','PLOT_AUTHORING_EXHAUSTED',
                         'REPORT_AUTHORING_FAILED','VALIDATION_FAILED'}
            or obj.get('error')
            or isinstance(exit_code, (int, float)) and exit_code != 0):
        return str(obj.get('error') or obj.get('stderr') or obj.get('issues') or state or 'execution failed')[:4000]
    if isinstance(result, str) and result.lstrip().startswith('Error:'):
        return result[:4000]
    return ''


def tool_failure_reason(tool,result):
    """Native capability decides observation semantics; result data cannot."""
    from .parallel_workflow import READ_ONLY
    if tool not in READ_ONLY and execution_deferred(result):return ''
    aliases={'check_job_status','get_job_status','list_jobs','inspect'}
    return failure_reason(result,diagnostic=tool in READ_ONLY or tool in aliases)


def execution_deferred(result):
    """An explicit non-execution receipt is neither success nor a retry grant."""
    obj=result_object(result)
    if obj.get('isError')is True or obj.get('is_error')is True:return False
    state=str(obj.get('status')or obj.get('state')or'').upper()
    waiting={'WAITING','WAITING_JOBS','WAITING_RESOURCES','NEEDS_RESOURCES','REVIEW_BUSY',
        'REVIEW_RETRY','AWAITING_APPROVAL','AWAITING_CONFIRMATION','NEEDS_USER','DEFERRED','PENDING'}
    return bool(obj.get('skipped')or obj.get('deduplicated')or
        (obj.get('blocked')or obj.get('executed')is False)and
        (state in waiting or obj.get('review_in_progress')is True or not failure_reason(obj)))


def _input_fingerprint(params, project_root, include_resource_review=False):
    """Bind a retry permit to exact arguments and bounded input provenance."""
    files = {}
    for key in ('cif', 'cif_path', 'cif_dir', 'input_path', 'input_dir', 'data_csv', 'path', 'work_dir'):
        value = params.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = Path(project_root) / path
        candidates = [path] if path.is_file() else []
        if path.is_dir():
            extensions = {'.py', '.sh', '.input'} if key == 'work_dir' else {'.cif', '.dat', '.json', '.py', '.sh'}
            candidates = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in extensions)
            if len(candidates) > 512:
                raise ValueError('input provenance exceeds 512 files; use a scoped input directory')
        for candidate in candidates:
            stat = candidate.stat()
            if stat.st_size <= 16 * 1024 * 1024:
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            else:
                digest = f'stat:{stat.st_size}:{stat.st_mtime_ns}'
            files[str(candidate.resolve())] = digest
    # Renaming outputs/jobs is not a fix and must not grant a new permit.
    ignored = {'output', 'output_dir', 'output_csv', 'job_work_dir', 'job_name', 'work_dir'}
    if not include_resource_review:
        ignored.add('resource_review_id')
    effective_params = {k: v for k, v in params.items() if k not in ignored}
    blob = json.dumps({'params': effective_params, 'files': files}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def input_fingerprint(params, project_root):
    """Fingerprint scientific inputs and real resource constraints, not receipt identity."""
    return _input_fingerprint(params, project_root, include_resource_review=False)


class RecoveryGate:
    """A claim is persisted BEFORE dispatch; an approval is consumed exactly once.

    Counts belong to a stable task identity, not error snippets/agent segments.
    Unknown outcomes and still-running calls fail closed. Disk transactions
    serialize claims made by separate API workers.
    """
    def __init__(self, state=None, path=None):
        self.state = dict(state or {})
        self.path = Path(path) if path else None
        self.lock = threading.RLock()

    @contextmanager
    def transaction(self):
        with self.lock:
            if self.path:
                with json_transaction(self.path) as data:
                    # Import an older Session snapshot only on first creation.
                    if not data:
                        data.update(self.state)
                    yield data
                    self.state = dict(data)
            else:
                yield self.state

    def key(self, tool, params, goal_version, node='', project_root='.'):
        family = {'run_isotherm_chain': 'gcmc', 'run_gcmc_isotherm': 'gcmc',
                  'run_gcmc_batch': 'gcmc', 'run_henry_chain': 'henry',
                  'run_henry': 'henry'}.get(tool, tool)
        target = {}
        for field in ('cif', 'cif_path', 'cif_dir', 'input_path', 'input_dir', 'material', 'data_csv', 'gas', 'gases'):
            if field in params:
                value = params[field]
                if field in {'cif', 'cif_path', 'cif_dir', 'input_path', 'input_dir', 'data_csv'} and isinstance(value, str):
                    p = Path(value)
                    value = str((p if p.is_absolute() else Path(project_root) / p).resolve())
                if field == 'gases':
                    value = sorted(str(g).upper() for g in value)
                elif field == 'gas':
                    value = str(value).upper()
                canonical_field = 'structure_path' if field in {'cif', 'cif_path'} else field
                target[canonical_field] = value
        if tool == 'submit_job':
            work = Path(params.get('work_dir') or project_root)
            target = {'work_dir': str((work if work.is_absolute() else Path(project_root) / work).resolve())}
        # Resource parameters/output filenames intentionally do not change task identity.
        return hashlib.sha256(json.dumps([family, goal_version, node, target], sort_keys=True, default=str).encode()).hexdigest()[:24]

    def claim(self, key, tool, params, fingerprint, goal_version=0, project_root=None, legacy_job_id=None):
        from .installations import executor_fingerprint
        runtime = executor_fingerprint(tool, project_root) if project_root else None
        with self.transaction() as data:
            entry = data.get(key)
            if entry:
                # v3.4.70 and older included the evidence receipt identifier in
                # the retry fingerprint. Migrate only when the stored digest is
                # exactly reproducible by that legacy algorithm.
                legacy = _input_fingerprint(entry.get('params', {}), entry.get('project_root') or '.',
                                            include_resource_review=True)
                if entry.get('fingerprint') == legacy:
                    normalized = input_fingerprint(entry.get('params', {}), entry.get('project_root') or '.')
                    entry['fingerprint'] = normalized
                    if entry.get('approved_fingerprint') == legacy:
                        entry['approved_fingerprint'] = normalized
                status = entry['status']
                resume_waiting_chain = status == 'completed_unverified' and entry.get('last_result', {}).get('chain_status') == 'waiting'
                if status in {'reserved', 'submitted', 'uncertain', 'completed_unverified', 'completed'} and not resume_waiting_chain:
                    return None, f'[RETRY_BLOCK] existing attempt is {status}; inspect its jobs, do not resubmit'
                if entry['attempts'] >= 3:
                    return None, '[RETRY_BLOCK] three dispatch attempts exhausted; ask user to revise the plan'
                # Older permits used the approved scientific node arguments
                # before kernel resource allocation was attached. Restore only
                # omitted fields from the original dispatch, while requiring
                # the original verified input fingerprint to remain unchanged.
                reviewed = entry.get('approved_arguments')
                if status == 'failed' and isinstance(reviewed, dict) and entry.get('approved_fingerprint') != fingerprint:
                    fields = {'memory_mb', 'nodelist', 'partition', 'num_processes', 'cpus_per_task'}
                    merged = {**{k: v for k, v in entry.get('params', {}).items() if k in fields}, **reviewed}
                    root = entry.get('project_root') or '.'
                    if (entry.get('approved_fingerprint') == input_fingerprint(reviewed, root)
                            and all(params.get(k) == merged.get(k) for k in fields)
                            and input_fingerprint(merged, root) == fingerprint):
                        entry['approved_arguments'] = merged
                        entry['approved_fingerprint'] = fingerprint
                if not resume_waiting_chain and entry.get('approved_fingerprint') != fingerprint:
                    return None, '[RETRY_BLOCK] failure requires prepare_retry with diagnostic and verification call IDs'
                if entry.get('approved_runtime_fingerprint') and entry['approved_runtime_fingerprint'] != runtime:
                    return None, '[RETRY_BLOCK] executor changed after verification; inspect and verify again'
            attempt_id = uuid.uuid4().hex
            data[key] = {
                **(entry or {}), 'key': key, 'tool': tool, 'params': copy.deepcopy(params),
                'fingerprint': fingerprint, 'attempt_id': attempt_id,
                'attempts': (entry or {}).get('attempts', 0) + 1,
                'status': 'reserved', 'started_at': time.time(), 'job_ids': [],
                'goal_version': goal_version,
                'identity_goal_version': (entry or {}).get('identity_goal_version', (entry or {}).get('goal_version', goal_version)),
                'owner_pid': os.getpid(),
                'legacy_job_id': legacy_job_id,
                'project_root': str(project_root) if project_root else None,
                'approved_fingerprint': None,
                'runtime_fingerprint': runtime, 'approved_runtime_fingerprint': None,
            }
            return attempt_id, ''

    def outcome(self, key, attempt_id, result, uncertain=False):
        with self.transaction() as data:
            entry = data.get(key)
            if not entry or entry.get('attempt_id') != attempt_id:
                return
            obj = result_object(result)
            # A result may conservatively veto acceptance/retry. It cannot use
            # this flag to grant execution, success or scheduler ownership.
            uncertain = uncertain or obj.get('uncertain') is True or str(obj.get('status')or'').lower() in {'uncertain','submission_unknown'}
            jobs = obj.get('job_ids') or ([obj['job_id']] if obj.get('job_id') else [])
            if obj.get('context'):
                ctx = obj['context']
                jobs = jobs or ctx.get('gcmc_job_ids') or ([ctx['henry_job_id']] if ctx.get('henry_job_id') else []) or ([ctx['charge_job_id']] if ctx.get('charge_job_id') else [])
            reason = failure_reason(result)
            if obj.get('blocked') or obj.get('skipped') or obj.get('executed') is False:
                reason = str(obj.get('error') or obj.get('reason') or 'dispatch was blocked/not executed')[:4000]
            merged_jobs = sorted(set(entry.get('job_ids', [])) | set(map(str, jobs)))
            # A watcher may observe a fast terminal job before the parent tool
            # returns. A late submission receipt is not newer execution evidence.
            preserve_terminal = (entry.get('status') in {'failed', 'completed_unverified', 'completed'}
                                 and not reason and not (set(map(str, jobs)) - set(entry.get('job_ids', []))))
            entry.update({'job_ids': merged_jobs, 'last_result': copy.deepcopy(obj)})
            if not preserve_terminal:
                entry['error'] = reason
                entry['status'] = 'uncertain' if uncertain else ('failed' if reason else ('submitted' if merged_jobs or obj.get('submitted') else 'completed_unverified'))
            if reason:
                entry['failed_at'] = time.time()
                if entry.get('project_root'):
                    entry['fingerprint'] = input_fingerprint(entry['params'], entry['project_root'])

    def register_job(self, key, attempt_id, job_id, work_dir=''):
        """Bind the scheduler receipt before the parent tool returns/crashes."""
        with self.transaction() as data:
            entry = data.get(key)
            if entry and entry.get('attempt_id') == attempt_id:
                already_bound = str(job_id) in entry.get('job_ids', [])
                entry['job_ids'] = sorted(set(entry.get('job_ids', [])) | {str(job_id)})
                if already_bound and entry.get('status') in {'failed', 'completed_unverified', 'completed'}:
                    return
                entry['status'] = 'submitted'
                entry['last_result'] = {'submitted': True, 'job_id': str(job_id), 'work_dir': work_dir}

    def reconcile_watched_job(self, key, job, proof):
        with self.transaction() as data:
            entry = data.get(key)
            if not entry or entry.get('status') not in {'reserved', 'uncertain', 'submitted'}:
                raise ValueError('only unresolved dispatches can be reconciled')
            entry['job_ids'] = sorted(set(entry.get('job_ids', [])) | {str(job['job_id'])})
            entry['status'] = ('failed' if job.get('failed') else 'completed_unverified'
                               if job.get('terminal') and job.get('state') == 'COMPLETED' else 'submitted')
            entry['last_result'] = job
            entry['identity_reconciliation'] = proof
            if job.get('failed'):
                entry['failed_at'] = proof['diagnostic_time']

    def sync_jobs(self, jobs):
        by_id = {str(j['job_id']): j for j in jobs if j.get('job_id')}
        with self.transaction() as data:
            for entry in data.values():
                if entry.get('status') not in {'submitted', 'uncertain'}:
                    continue
                tracked = [by_id[jid] for jid in entry.get('job_ids', []) if jid in by_id]
                if not tracked or len(tracked) != len(entry.get('job_ids', [])):
                    continue
                if all(j.get('terminal') for j in tracked):
                    failures = [j for j in tracked if j.get('failed')]
                    if failures:
                        entry['status'] = 'failed'
                        entry['error'] = json.dumps(failures, default=str)[:4000]
                        entry['failed_at'] = time.time()
                        if entry.get('project_root'):
                            entry['fingerprint'] = input_fingerprint(entry['params'], entry['project_root'])
                    elif all(str(j.get('state', '')).upper() in {'COMPLETED', 'SUCCESS', 'DONE'} for j in tracked):
                        entry['status'] = 'completed_unverified'

    def approve_configured_preflight(self, key, attempt_id, configured_interpreter, arguments=None):
        """Kernel-only repair for the generator's explicit pre-submit rejection.

        The caller supplies the operator configuration, never a model argument.
        A changed executable path is fresh proof; no running/unknown/submitted
        attempt or generic failure may enter this path. Dispatch counts remain.
        """
        with self.transaction() as data:
            entry = data.get(key) or {}
            result = entry.get('last_result') or {}
            prefix = 'configured pormake interpreter is unavailable: '
            error = str(result.get('error', ''))
            if (entry.get('tool') != 'generate_structure' or entry.get('status') != 'failed'
                    or entry.get('attempt_id') != attempt_id or entry.get('job_ids')
                    or result.get('blocked') is not True or result.get('executed') is not False
                    or not error.startswith(prefix) or entry.get('attempts', 0) >= 3):
                raise ValueError('configuration recovery requires an unexecuted generator preflight failure')
            old = Path(error[len(prefix):]).expanduser()
            current = Path(configured_interpreter).expanduser()
            if (not current.is_absolute() or current.resolve() == old.resolve()
                    or not current.is_file() or not os.access(current, os.X_OK)):
                raise ValueError('a changed, existing executable from operator configuration is required')
            proof = {'kind': 'verified_configuration_preflight', 'attempt_id': attempt_id,
                     'previous_interpreter': str(old), 'configured_interpreter': str(current.resolve()),
                     'verified_at': time.time(), 'no_job_submitted': True}
            approved = entry['fingerprint']
            if arguments is not None:
                resource_fields = {'memory_mb', 'nodelist', 'partition', 'cpus_per_task', 'resource_review_id'}
                scientific = lambda params: {k: v for k, v in params.items() if k not in resource_fields}
                if scientific(arguments) != scientific(entry.get('params', {})):
                    raise ValueError('configuration repair cannot alter the scientific task')
                approved = input_fingerprint(arguments, entry.get('project_root') or '.')
            entry['approved_fingerprint'] = approved
            entry.setdefault('reviews', []).append(proof)
            return {'ok': True, 'proof': proof, 'single_use': True}

    def approve(self, key, fingerprint, review, runtime_verification=None, arguments=None, *, expected_attempt_id=None, control_valid=None):
        with self.transaction() as data:
            entry = data.get(key)
            if control_valid is not None and (not callable(control_valid) or control_valid() is not True):
                raise ValueError('native retry control changed before permit commit')
            if expected_attempt_id is not None and (not entry or entry.get('attempt_id') != expected_attempt_id):
                raise ValueError('failed attempt changed during retry verification')
            if not entry or entry.get('status') != 'failed':
                raise ValueError('retry requires a confirmed failed attempt, not UNKNOWN/running')
            if entry['attempts'] >= 3:
                raise ValueError('attempt budget exhausted; user must revise the plan')
            runtime = None
            if runtime_verification is not None:
                from .installations import executor_fingerprint
                proof = runtime_verification
                root = entry.get('project_root')
                if (not root or proof.get('ok') is not True or proof.get('read_only') is not True
                        or proof.get('verified_tool') != entry['tool']):
                    raise ValueError('successful native executor verification required')
                runtime = executor_fingerprint(entry['tool'], root)
                if proof.get('runtime_fingerprint') != runtime:
                    raise ValueError('executor changed since native verification')
                previous = entry.get('runtime_fingerprint')
                # Older receipts have no executor digest. Require a real source
                # change after the recorded failure, checked from disk here.
                legacy_changed = False
                if not previous:
                    from .installations import IMPLEMENTATIONS
                    for name in ['agents/registry.py'] + IMPLEMENTATIONS.get(entry['tool'], []):
                        file = Path(root) / name
                        if file.is_file() and file.stat().st_mtime > entry.get('failed_at', entry['started_at']):
                            legacy_changed = True
                if previous == runtime or not previous and not legacy_changed:
                    raise ValueError('a changed executor must be established before retry')
            if fingerprint == entry['fingerprint'] and runtime is None:
                raise ValueError('arguments and input contents are unchanged; no verified fix')
            entry['approved_fingerprint'] = fingerprint
            entry['approved_runtime_fingerprint'] = runtime
            entry['approved_attempt_id'] = entry['attempt_id']
            entry['approved_arguments'] = copy.deepcopy(arguments if arguments is not None else entry.get('params', {}))
            entry.setdefault('reviews', []).append({**review, 'time': time.time()})
            if control_valid is not None and control_valid() is not True:
                raise ValueError('native retry control changed during permit commit')
            return {'ok': True, 'key': key, 'next_attempt': entry['attempts'] + 1, 'single_use': True}

    def accept_result(self, key, evidence):
        with self.transaction() as data:
            entry = data.get(key)
            if not entry or entry.get('status') != 'completed_unverified':
                raise ValueError('calculation completion must be confirmed before accepting results')
            entry['status'] = 'completed'
            entry['result_verification'] = {**evidence, 'time': time.time()}
            return {'ok': True, 'key': key, 'status': 'completed'}

    def snapshot(self):
        with self.lock:
            if self.path and self.path.exists():
                data = json.loads(self.path.read_text())
                if not isinstance(data, dict):
                    raise ValueError('recovery ledger is corrupt')
                self.state = data
            return json.loads(json.dumps(self.state, default=str))
