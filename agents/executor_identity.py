"""Read-only identity of loaded Python executors versus their disk source.

Compiling source does not execute it. Filenames, line numbers and unstable set
iteration do not change this digest. A match proves this function body only,
not all imported helper bindings, its environment or scientific correctness.
"""
import hashlib
import inspect
import json
import sys
from pathlib import Path
import types


def _code(value):
    if isinstance(value, types.CodeType):
        return ['code', value.co_code.hex(), value.co_argcount,
                getattr(value, 'co_posonlyargcount', 0), value.co_kwonlyargcount,
                value.co_flags, list(value.co_names), list(value.co_varnames),
                list(value.co_freevars), list(value.co_cellvars),
                [_code(item) for item in value.co_consts]]
    if isinstance(value, tuple):
        return ['tuple', [_code(item) for item in value]]
    if isinstance(value, frozenset):
        return ['frozenset', sorted((_code(item) for item in value), key=repr)]
    if isinstance(value, bytes):
        return ['bytes', value.hex()]
    if value is Ellipsis:
        return ['ellipsis']
    if value is None or type(value) in (str, int, float, complex, bool):
        return [type(value).__name__, repr(value)]
    raise TypeError('unsupported code constant')


def code_digest(code):
    return hashlib.sha256(json.dumps(_code(code), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def executor_identity(function):
    """Compare one named Python executor against compiled source without exec.

    Unknown/ambiguous/native functions return verified=False, never a match.
    Callers decide policy; this helper does not dispatch, reload or grant retry.
    """
    record = {'schema': 'loaded-executor-body-v1', 'verified': False,
              'scope': 'unwrapped executor body only; decorators and imported dependencies excluded'}
    try:
        function = inspect.unwrap(function)
        loaded = function.__code__
        name = getattr(function, '__qualname__', function.__name__)
        if not inspect.isfunction(function) or '<' in name:
            raise ValueError('a named Python executor is required')
        source = Path(inspect.getsourcefile(function)).resolve(strict=True)
        if source.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('executor source exceeds bounded inspection')
        raw = source.read_bytes()
        compiled = compile(raw, str(source), 'exec', dont_inherit=True, optimize=sys.flags.optimize)
        path = name.split('.')
        current = compiled
        for part in path:
            found = [item for item in current.co_consts
                     if isinstance(item, types.CodeType) and item.co_name == part]
            if len(found) != 1:
                raise ValueError('executor definition is absent or ambiguous')
            current = found[0]
        disk, actual = code_digest(current), code_digest(loaded)
        record.update(verified=True, source_path=str(source), qualified_name=name,
                      source_sha256=hashlib.sha256(raw).hexdigest(),
                      loaded_code_sha256=actual, source_code_sha256=disk,
                      matches=actual == disk)
    except (AttributeError, OSError, TypeError, ValueError, SyntaxError):
        record['reason'] = 'executor identity unavailable or ambiguous'
    return record
