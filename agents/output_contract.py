"""Machine-checkable artifacts shared by approval, locking and validation."""
from pathlib import Path
import re


OUTPUT_SCHEMA = {'oneOf': [
    {'type': 'string', 'minLength': 1},
    {'type': 'object', 'additionalProperties': False, 'properties': {
        'kind': {'type': 'string', 'enum': ['file', 'directory']},
        'path': {'type': 'string', 'minLength': 1},
        'pattern': {'type': 'string', 'minLength': 1},
        'min_count': {'type': 'integer', 'minimum': 1},
    }, 'required': ['kind', 'path']},
]}


def normalize_output(value):
    from jsonschema import Draft7Validator
    issues = list(Draft7Validator(OUTPUT_SCHEMA).iter_errors(value))
    if issues:
        raise ValueError('expected_outputs require file paths or typed file/directory contracts')
    contract = {'kind': 'file', 'path': value} if isinstance(value, str) else dict(value)
    path = contract['path']
    if path != path.strip() or any(c in path for c in ('\n', '\r', '\x00')):
        raise ValueError('artifact path is not a valid literal path')
    if isinstance(value, str) and '/' not in path and not re.search(r'\.[A-Za-z0-9_*?]+$', path):
        raise ValueError('expected_outputs is prose or an ambiguous name; use a file path or typed directory contract')
    if contract['kind'] == 'file' and set(contract) & {'pattern', 'min_count'}:
        raise ValueError('pattern/min_count belong to directory contracts')
    pattern = contract.get('pattern', '*')
    if '/' in pattern or '\\' in pattern or pattern in {'.', '..'}:
        raise ValueError('directory pattern must be a bounded filename glob, not a recursive path')
    return contract


def output_path(value, base):
    path = Path(normalize_output(value)['path'])
    return (path if path.is_absolute() else Path(base) / path).resolve()


def artifact_fact(value, base):
    contract = normalize_output(value)
    path = output_path(value, base)
    # Legacy *.cif strings mean files in one directory, never an actual
    # filename containing '*'. New plans should use directory contracts.
    wildcard = contract['kind'] == 'file' and any(c in path.name for c in '*?[')
    if contract['kind'] == 'directory' or wildcard:
        folder = path.parent if wildcard else path
        pattern = path.name if wildcard else contract.get('pattern', '*')
        if not folder.is_dir():
            return None
        files = []
        for index, item in enumerate(folder.glob(pattern)):
            if index >= 10000:
                return None  # bounded validation, never silently validate a subset
            if not item.is_file() or not item.resolve().is_relative_to(folder.resolve()):
                return None
            stat = item.stat()
            # JSON checkpoints use arrays. Tuples never compared equal after a
            # reload, incorrectly invalidating every completed directory.
            files.append([str(item), stat.st_size, stat.st_mtime_ns])
        if len(files) < contract.get('min_count', 1) or any(size == 0 for _, size, _ in files):
            return None
        return {'size': sum(size for _, size, _ in files),
                'mtime_ns': min(mtime for _, _, mtime in files), 'files': sorted(files)}
    if path.is_file():
        stat = path.stat()
        return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return None
