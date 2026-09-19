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

SUBMISSION_TOOLS = frozenset({
    'run_gcmc_isotherm', 'run_gcmc_batch', 'run_henry', 'run_henry_chain',
    'run_isotherm_chain', 'run_pacman_charge', 'run_pore_analysis',
    'generate_structure', 'run_xtb_optimize', 'run_md_optimize',
    'run_string_tst', 'run_external_potential', 'submit_job',
    'ml_train', 'ml_predict', 'ml_feature_importance', 'ml_active_learning',
    'run_ga_optimization', 'build_guest_forcefield',
})
FAILURE_STATES = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'REJECTED', 'FATAL'}


def is_submission(tool, params):
    return (tool in SUBMISSION_TOOLS
            or tool == 'run_cdft' and params.get('action') in {'pipeline', 'submit'}
            or tool == 'run_vasp' and params.get('action') in {'submit', 'setup-batch'})


def result_object(result):
    if isinstance(result, dict):
        parsed = result
    else:
        try:
            parsed = json.loads(result)
            if not isinstance(parsed, dict):
                return {}
        except (TypeError, ValueError):
            return {}
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
            return {**parsed, **max(candidates, key=lambda pair: pair[0])[1]}
    return parsed


def failure_reason(result):
    """Structured signals only; never scan arbitrary output for 'FAILED'."""
    obj = result_object(result)
    if obj.get('blocked') or obj.get('skipped') or obj.get('deduplicated'):
        return ''
    state = str(obj.get('status') or obj.get('state') or '').upper()
    exit_code = obj.get('exit_code', obj.get('returncode'))
    if (obj.get('failed') is True or obj.get('success') is False
            or state in FAILURE_STATES or obj.get('chain_status') == 'failed'
            or obj.get('error') or obj.get('isError') is True
            or isinstance(exit_code, (int, float)) and exit_code != 0):
        return str(obj.get('error') or obj.get('stderr') or obj.get('issues') or state or 'execution failed')[:4000]
    if isinstance(result, str) and result.lstrip().startswith('Error:'):
        return result[:4000]
    return ''


def input_fingerprint(params, project_root):
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
    effective_params = {k: v for k, v in params.items()
                        if k not in {'output', 'output_dir', 'output_csv', 'job_work_dir', 'job_name', 'work_dir'}}
    blob = json.dumps({'params': effective_params, 'files': files}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


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
        with self.transaction() as data:
            entry = data.get(key)
            if entry:
                status = entry['status']
                resume_waiting_chain = status == 'completed_unverified' and entry.get('last_result', {}).get('chain_status') == 'waiting'
                if status in {'reserved', 'submitted', 'uncertain', 'completed_unverified', 'completed'} and not resume_waiting_chain:
                    return None, f'[RETRY_BLOCK] existing attempt is {status}; inspect its jobs, do not resubmit'
                if entry['attempts'] >= 3:
                    return None, '[RETRY_BLOCK] three dispatch attempts exhausted; ask user to revise the plan'
                if not resume_waiting_chain and entry.get('approved_fingerprint') != fingerprint:
                    return None, '[RETRY_BLOCK] failure requires prepare_retry with diagnostic and verification call IDs'
            attempt_id = uuid.uuid4().hex
            data[key] = {
                **(entry or {}), 'key': key, 'tool': tool, 'params': copy.deepcopy(params),
                'fingerprint': fingerprint, 'attempt_id': attempt_id,
                'attempts': (entry or {}).get('attempts', 0) + 1,
                'status': 'reserved', 'started_at': time.time(), 'job_ids': [],
                'goal_version': goal_version,
                'owner_pid': os.getpid(),
                'legacy_job_id': legacy_job_id,
                'project_root': str(project_root) if project_root else None,
                'approved_fingerprint': None,
            }
            return attempt_id, ''

    def outcome(self, key, attempt_id, result, uncertain=False):
        with self.transaction() as data:
            entry = data.get(key)
            if not entry or entry.get('attempt_id') != attempt_id:
                return
            obj = result_object(result)
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

    def approve(self, key, fingerprint, review):
        with self.transaction() as data:
            entry = data.get(key)
            if not entry or entry.get('status') != 'failed':
                raise ValueError('retry requires a confirmed failed attempt, not UNKNOWN/running')
            if entry['attempts'] >= 3:
                raise ValueError('attempt budget exhausted; user must revise the plan')
            if fingerprint == entry['fingerprint']:
                raise ValueError('arguments and input contents are unchanged; no verified fix')
            entry['approved_fingerprint'] = fingerprint
            entry.setdefault('reviews', []).append({**review, 'time': time.time()})
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
