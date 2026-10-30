"""Bounded correction of a task's own isolated code, without supervisor routing.

The caller owns the attempt directories and publishes successful outputs.  This
module never runs generated code itself or changes inputs/output declarations.
"""
import copy
import hashlib
import json
import os
from pathlib import Path


class SandboxSourceChanged(ValueError):
    def __init__(self, attempts):
        super().__init__('sandbox source changed; no corrected result was published')
        self.attempts = copy.deepcopy(attempts)


class SandboxCorrectionUnavailable(ValueError):
    def __init__(self, attempts, error_type):
        super().__init__('sandbox script failed and model correction was unavailable')
        self.attempts = copy.deepcopy(attempts)
        # Never persist arbitrary provider exception text or credentials.
        self.error_type = str(error_type)[:120]


def verify_input_bytes(root, sources):
    from .plot_sandbox import _read, _stream_text, MAX_STREAM_TEXT
    root = Path(root).absolute()
    if not isinstance(sources, list):raise ValueError('missing sandbox source identities')
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get('source_path'), str):
            raise ValueError('invalid sandbox source identity')
        path = Path(source['source_path'])
        if type(source.get('bytes')) is not int or source['bytes'] < 0:
            raise ValueError('invalid sandbox source size')
        if source['bytes'] > 8 * 1024**2:
            fact = _stream_text(root, path, MAX_STREAM_TEXT)
            digest, size = fact['sha256'], fact['bytes']
        else:
            raw = _read(root, path, 8 * 1024**2)
            digest, size = hashlib.sha256(raw).hexdigest(), len(raw)
        if digest != source.get('sha256') or size != source['bytes']:
            raise ValueError('sandbox source changed')
    return True


def run_self_correction(script, *, execute_attempt, request_correction, control_valid,
                        read_stderr, verify_inputs, max_attempts=3):
    if not isinstance(script, str) or not script or len(script.encode()) > 65536:
        raise ValueError('invalid initial sandbox script')
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError('invalid sandbox attempt budget')
    if not all(callable(x) for x in (execute_attempt, request_correction, control_valid, read_stderr, verify_inputs)):
        raise TypeError('native sandbox capabilities are required')
    attempts = []
    current = script
    fixed_inputs = None
    def source_check(inputs):
        try:
            if verify_inputs(inputs) is True:return
        except (OSError, ValueError, PermissionError, TypeError, KeyError):
            pass
        raise SandboxSourceChanged(attempts) from None
    for index in range(max_attempts):
        if control_valid() is not True:
            raise PermissionError('sandbox task authorization changed before execution')
        if fixed_inputs is not None:source_check(fixed_inputs)
        result = execute_attempt(index, current)
        if not isinstance(result, dict):
            raise TypeError('invalid sandbox execution receipt')
        entry = {'attempt': index + 1, 'script_sha256': hashlib.sha256(current.encode()).hexdigest(),
                 'status': result.get('status'), 'error_kind': result.get('error_kind'),
                 'exit_code': result.get('exit_code'), 'output_dir': result.get('output_dir'),
                 'sandbox_network_enabled': (result.get('sandbox') or {}).get('network') is True,
                 'network_connectivity_verified': False}
        attempts.append(entry)
        if control_valid() is not True:
            raise PermissionError('sandbox task authorization changed after execution')
        sources = result.get('inputs')
        if not isinstance(sources, list):
            # An unexecuted preflight/isolation error may lack a source audit;
            # do not ask a model to correct or retry it.
            if result.get('ok') is True:raise ValueError('successful sandbox receipt lacks source identities')
            return {'ok': False, 'result': result, 'attempts': attempts}
        identities = [{k: item.get(k) for k in ('source_path','input_id','format','sha256','bytes')}
                      for item in sources if isinstance(item, dict)]
        if len(identities) != len(sources):raise ValueError('invalid sandbox source identities')
        if fixed_inputs is None:fixed_inputs = copy.deepcopy(identities)
        elif identities != fixed_inputs:raise SandboxSourceChanged(attempts)
        source_check(fixed_inputs)
        if result.get('ok') is True:
            return {'ok': True, 'result': result, 'attempts': attempts}
        # Only a script that actually ran *inside* the OS sandbox and failed
        # with an exit code may be sent back to its author for correction.
        # Isolation failures, timeouts, source changes and uncertain outcomes
        # cannot safely be retried from model output.
        if (result.get('error_kind') != 'data_execution_failed'
                or (result.get('sandbox') or {}).get('os_isolated') is not True
                or type(result.get('exit_code')) is not int
                or result.get('exit_code') == 0 or result.get('timed_out') is True
                or index + 1 >= max_attempts):
            return {'ok': False, 'result': result, 'attempts': attempts}
        stderr = read_stderr(result)
        if not isinstance(stderr, str):
            raise TypeError('native sandbox stderr must be text')
        # This is explicitly untrusted data from the user's script. It cannot
        # override the approved input files, output contract, owner or control.
        try:
            response = request_correction({'script': current, 'stderr': stderr[-4000:],
                'failure': copy.deepcopy(entry), 'stderr_is_untrusted': True,
                'sandbox_inputs': [{k: item.get(k) for k in ('input_id','format','sandbox_path','sha256','bytes')}
                                   for item in sources],
                'preserve_inputs_and_output_contract': True})
        except Exception as error:
            if control_valid() is not True:
                raise PermissionError('sandbox task authorization changed during correction') from None
            # The model is data-only, cannot invoke tools, and all sandbox
            # attempts have returned; no final artifact was published.
            raise SandboxCorrectionUnavailable(attempts, type(error).__name__) from None
        if control_valid() is not True:
            raise PermissionError('sandbox task authorization changed during correction')
        if not isinstance(response, dict) or set(response) != {'script'}:
            raise ValueError('sandbox correction must return only a script')
        current = response['script']
        if not isinstance(current, str) or not current or len(current.encode()) > 65536:
            raise ValueError('invalid corrected sandbox script')
    raise AssertionError('unreachable')


def publish_corrected_result(root, output_dir, result, declared, attempts, control_valid,
                             implementation_valid, implementation):
    """Publish a successful isolated attempt at the originally approved path.

    Failed attempts and their logs remain immutable in separate directories.
    An exclusive directory creation and the existing no-replace data publisher
    prevent replacement of any user result, including on NFS.
    """
    from .plot_sandbox import _safe_parent, _read, _write, _publish_data_outputs
    root = Path(root).absolute(); out = Path(output_dir).absolute()
    if not out.is_relative_to(root) or out == root:
        raise PermissionError('sandbox output is outside owned task')
    if control_valid() is not True or implementation_valid() is not True:
        raise PermissionError('sandbox task authorization changed before publication')
    artifacts = result.get('artifacts') or []
    if (result.get('ok') is not True or len(artifacts) != len(declared)
            or {a.get('name') for a in artifacts} != {a.get('name') for a in declared}):
        raise ValueError('successful attempt does not match declared outputs')
    records = []
    verify_input_bytes(root, result.get('inputs') or [])
    for item in artifacts:
        raw = _read(root, Path(item['path']), 8 * 1024**2)
        if hashlib.sha256(raw).hexdigest() != item.get('sha256') or len(raw) != item.get('bytes'):
            raise ValueError('sandbox attempt artifact changed before publication')
        records.append((item, raw))
    parent = _safe_parent(root, out.parent)
    try:
        os.mkdir(out.name, mode=0o700, dir_fd=parent)
        fd = os.open(out.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        os.mkdir('rendered', mode=0o700, dir_fd=fd)
        rendered = os.open('rendered', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            for item, raw in records:
                _write(rendered, item['name'], raw)
            os.fsync(rendered)
        finally:
            os.close(rendered)
        if control_valid() is not True or implementation_valid() is not True:
            raise PermissionError('sandbox task authorization changed during publication')
        specs = {item['name']: item for item in declared}
        observed = [{**{k: item[k] for k in ('name', 'sha256', 'bytes')},
                     'format': specs[item['name']]['format']} for item, _ in records]
        checked, publication = _publish_data_outputs(fd, out, observed, declared, result.get('inputs') or [])
        # A cancel or source replacement racing publication must not create a
        # success receipt. Retain any already-created immutable files as
        # failure evidence; do not remove or overwrite them.
        if control_valid() is not True or implementation_valid() is not True:
            raise PermissionError('sandbox task authorization changed during publication')
        verify_input_bytes(root, result.get('inputs') or [])
        receipt = copy.deepcopy(result)
        receipt.update(output_dir=str(out), artifacts=checked, publication=publication,
                       correction_implementation=copy.deepcopy(implementation),
                       correction_attempts=copy.deepcopy(attempts),
                       correction_attempt_output_dir=result.get('output_dir'))
        raw = json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
        _write(fd, 'execution.json', raw)
        _write(fd, 'manifest.json', raw)
        os.fsync(fd)
        return receipt
    finally:
        os.close(fd)
