"""Source-bound RASPA local composition; no parameter fitting or global edits."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
import stat
from pathlib import Path
import re
import uuid

MAX_ASSET_BYTES = 8 * 1024 * 1024
MANIFEST = '.raspa-assets.json'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def composition_sha256(manifest):
    return hashlib.sha256(_canonical(manifest)).hexdigest()


def _label(value):
    if not isinstance(value, str) or value in {'.', '..'} or not re.fullmatch(r'[A-Za-z0-9_.+\-]+', value):
        raise ValueError('RASPA asset selector must be one literal native name')
    return value


def _read(path):
    # Keep the compute-node verifier stdlib-only: importing agents would load
    # the backend and its state for every pressure point.
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('asset requires an absolute non-traversing path')
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent); parent = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_ASSET_BYTES:
            raise ValueError('asset must be a bounded nonempty regular file')
        chunks, size = [], 0
        while True:
            block = os.read(fd, min(65536, MAX_ASSET_BYTES + 1 - size))
            if not block: break
            size += len(block)
            if size > MAX_ASSET_BYTES: raise ValueError('asset read bound exceeded')
            chunks.append(block)
        after = os.fstat(fd);linked = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns, item.st_mode)
        if size != before.st_size or identity(before) != identity(after) or identity(after) != identity(linked):
            raise ValueError('asset changed while observing')
        raw = b''.join(chunks)
        return raw, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size': size}
    finally:
        if fd is not None: os.close(fd)
        os.close(parent)


def _rows(raw, *, mixing=False):
    lines = raw.decode('utf-8').splitlines(keepends=True)
    data = [(i, line.split('#', 1)[0].strip()) for i, line in enumerate(lines)
            if line.split('#', 1)[0].strip()]
    count_position = next((j for j, (_, text) in enumerate(data) if text.isdigit()), None)
    if count_position is None or (not mixing and count_position != 0):
        raise ValueError('unrecognized native parameter count')
    count_index, count_text = data[count_position]
    count = int(count_text)
    if not 0 < count <= 8192 or len(data) < count_position + 1 + count:
        raise ValueError('native parameter count is incomplete or exceeds bound')
    records = data[count_position + 1:count_position + 1 + count]
    result = {}
    for index, text in records:
        tokens = text.split(); name = tokens[0]
        if name in result or len(tokens) < (2 if mixing else 13):
            raise ValueError('duplicate or malformed native parameter row')
        result[name] = {'tokens': tokens, 'raw': lines[index].rstrip('\r\n')}
    tail = data[count_position + 1 + count:]
    if mixing:
        if len(tail) != 1:
            raise ValueError('native mixing rule must be explicit and unique')
        rule = tail[0][1]
    else:
        if tail:
            raise ValueError('extra native pseudoatom rows beyond declared count')
        rule = None
    return lines, count_index, result, rule


def _lj(row):
    tokens = row['tokens']
    values = tuple(float(value.replace('D', 'E').replace('d', 'e')) for value in tokens[2:])
    if tokens[1].lower() != 'lennard-jones' or len(values) != 2 or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('guest requires an explicit finite native Lennard-Jones row')
    return tokens[1].lower(), values


def _finite_guest(row):
    mass, charge = (float(row['tokens'][index]) for index in (5, 6))
    if not math.isfinite(mass) or mass < 0 or not math.isfinite(charge):
        raise ValueError('guest pseudoatom requires finite source mass and charge')


def _guest_atoms(raw):
    active, atoms = False, []
    for line in raw.decode('utf-8').splitlines():
        text = line.strip()
        if text.startswith('#'):
            active = 'atomic positions' in text.lower()
            continue
        if active and text:
            row = text.split()
            if len(row) < 2 or not row[0].isdigit():
                raise ValueError('unrecognized molecule atomic positions')
            atoms.append(row[1])
            if len(atoms) > 4096:
                raise ValueError('molecule site count exceeds bound')
    if not atoms:
        raise ValueError('molecule has no explicit native atom types')
    return sorted(set(atoms))


@dataclass(frozen=True)
class RaspaComposition:
    files: dict
    manifest: dict


def compose_raspa_assets(share, *, forcefield, molecule_name, molecule_def, guest_reference=None):
    """Append missing named guest pseudoatoms only from a reviewed source.

    Guest LJ coefficients and the mixing rule must already agree exactly in
    the selected framework model and its reviewed guest reference. Framework
    rows, all LJ bytes and the selected molecule definition stay unchanged.
    """
    root = Path(share).resolve(strict=True)
    forcefield, molecule_name, molecule_def = map(_label, (forcefield, molecule_name, molecule_def))
    sources, files = {}, {}
    for name in ('pseudo_atoms.def', 'force_field_mixing_rules.def', 'force_field.def'):
        files[name], sources['framework/' + name] = _read(root / 'forcefield' / forcefield / name)
    molecule = molecule_name + '.def'
    files[molecule], sources['guest/' + molecule] = _read(root / 'molecules' / molecule_def / molecule)
    guest_types = _guest_atoms(files[molecule])
    lines, count_index, pseudo, _ = _rows(files['pseudo_atoms.def'])
    _, _, mixing, rule = _rows(files['force_field_mixing_rules.def'], mixing=True)
    if any(atom not in mixing for atom in guest_types):
        raise ValueError('selected force field lacks guest LJ parameters; do not infer them')
    for atom in guest_types:
        if atom in pseudo: _finite_guest(pseudo[atom])
    missing = sorted(set(guest_types) - set(pseudo))
    if missing:
        reference = _label(guest_reference)
        donor_raw, sources['guest_reference/pseudo_atoms.def'] = _read(root / 'forcefield' / reference / 'pseudo_atoms.def')
        donor_mixing, sources['guest_reference/force_field_mixing_rules.def'] = _read(root / 'forcefield' / reference / 'force_field_mixing_rules.def')
        _, _, donor, _ = _rows(donor_raw)
        _, _, donor_lj, donor_rule = _rows(donor_mixing, mixing=True)
        if rule.casefold() != donor_rule.casefold():
            raise ValueError('guest reference mixing rule conflicts with the selected model')
        for atom in guest_types:
            if atom not in donor or atom not in donor_lj or _lj(mixing[atom]) != _lj(donor_lj[atom]):
                raise ValueError('guest source is incomplete or its LJ parameters conflict with the selected model')
            _finite_guest(donor[atom])
            # Existing same-name charges must not be silently replaced either.
            if atom in pseudo and pseudo[atom]['tokens'] != donor[atom]['tokens']:
                raise ValueError('existing guest pseudoatom conflicts with reviewed source')
        newline = '\r\n' if lines[count_index].endswith('\r\n') else '\n'
        lines[count_index] = str(len(pseudo) + len(missing)) + newline
        base = ''.join(lines)
        if not base.endswith(('\n', '\r')):
            base += newline
        files['pseudo_atoms.def'] = (base + ''.join(donor[atom]['raw'] + newline for atom in missing)).encode()
    manifest = {'schema_version': 1, 'source_root': str(root), 'forcefield': forcefield,
        'molecule_name': molecule_name, 'molecule_def': molecule_def, 'guest_reference': guest_reference,
        'guest_atom_types': guest_types, 'added_pseudo_types': missing, 'mixing_rule': rule,
        'sources': sources, 'file_hashes': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
    return RaspaComposition(files, manifest)


def _output_directory(path):
    absolute = Path(os.path.abspath(path))
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in absolute.parts[1:]:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            try: child = os.open(name, flags, dir_fd=fd)
            except FileNotFoundError:
                try: os.mkdir(name, dir_fd=fd)
                except FileExistsError: pass
                child = os.open(name, flags, dir_fd=fd)
            os.close(fd); fd = child
        return fd
    except BaseException:
        os.close(fd); raise


def _publish(directory, filename, raw):
    """Exclusive publication, retaining every existing different result."""
    fd = _output_directory(directory)
    temporary = None
    try:
        try:
            existing, _ = _read(Path(directory) / filename)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != raw: raise ValueError('existing local RASPA evidence differs; use a fresh job directory')
            return
        temporary = '.raspa-stage-' + uuid.uuid4().hex
        handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        with os.fdopen(handle, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try: os.link(temporary, filename, src_dir_fd=fd, dst_dir_fd=fd)
        except FileExistsError:
            existing, _ = _read(Path(directory) / filename)
            if existing != raw: raise ValueError('concurrent local RASPA evidence differs')
        os.fsync(fd)
    finally:
        if temporary is not None:
            try: os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError: pass
        os.close(fd)


def stage_raspa_assets(directory, composition):
    directory = Path(os.path.abspath(directory))
    if directory.resolve().is_relative_to(Path(composition.manifest['source_root'])):
        raise ValueError('job-local staging must never edit the parameter source tree')
    payloads = {**composition.files, MANIFEST: _canonical(composition.manifest)}
    # Preflight every existing target before adding any new file.
    for name, raw in payloads.items():
        path = directory / name
        if path.exists() or path.is_symlink():
            current, _ = _read(path)
            if current != raw: raise ValueError('existing local RASPA evidence differs; use a fresh job directory')
    for name, raw in payloads.items():
        _publish(directory, name, raw)
    return [str(directory / name) for name in payloads]


def verify_staged_raspa_assets(directory, expected_sha256):
    raw, _ = _read(Path(directory) / MANIFEST)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise ValueError('duplicate RASPA manifest field')
            value[key] = item
        return value
    manifest = json.loads(raw, object_pairs_hook=unique)
    if manifest.get('schema_version') != 1 or composition_sha256(manifest) != expected_sha256:
        raise ValueError('local RASPA manifest does not match the approved composition')
    hashes = manifest.get('file_hashes') or {}
    if set(hashes) != {'pseudo_atoms.def', 'force_field_mixing_rules.def', 'force_field.def', manifest['molecule_name'] + '.def'}:
        raise ValueError('incomplete local RASPA model')
    for name, expected in hashes.items():
        _label(name)
        _, fact = _read(Path(directory) / name)
        if fact['sha256'] != expected: raise ValueError('staged RASPA parameter file changed')
    return manifest


def discover_raspa_forcefields(share, query, required_atom_types=None, max_results=12):
    if type(max_results) is not int or not 1 <= max_results <= 32:
        raise ValueError('RASPA discovery result count must be in [1, 32]')
    query = query.strip()
    root = Path(share).resolve(strict=True);folder = root / 'forcefield'
    matches = []
    for index, path in enumerate(sorted(folder.iterdir())):
        if index >= 256: raise ValueError('RASPA force-field directory exceeds discovery bound')
        if path.is_dir() and not path.is_symlink() and query.casefold() in path.name.casefold():matches.append(path)
    candidates = []
    for path in matches[:max_results]:
        pseudo_raw, pseudo_fact = _read(path / 'pseudo_atoms.def')
        mixing_raw, mixing_fact = _read(path / 'force_field_mixing_rules.def')
        _, _, pseudo, _ = _rows(pseudo_raw);_, _, mixing, rule = _rows(mixing_raw, mixing=True)
        required = set(required_atom_types or [])
        candidates.append({'name': path.name, 'family': path.name, 'target_engine': 'raspa', 'format': 'raspa_native',
            'integrity_verified': True, 'asset_path': str(path), 'sources': {'pseudo_atoms': pseudo_fact, 'mixing_rules': mixing_fact},
            'mixing_rule': rule, 'pseudo_atom_count': len(pseudo), 'missing_named_types': sorted(required - set(pseudo)),
            'missing_lj_types': sorted(required - set(mixing)), 'ready_to_submit': False,
            'coverage_status': 'native_named_types_only_not_structure_assignment'})
    return {'ok': True, 'read_only': True, 'target_engine': 'raspa', 'query': query,
        'total_matches': len(matches), 'candidates': candidates,
        'selection_status': 'candidate_found' if candidates else 'not_found',
        'note': 'Framework and guest sources remain separate; native preflight verifies the selected local composition. No Towhee conversion or force-field substitution is implied.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('action', choices=['verify'])
    parser.add_argument('--directory', required=True);parser.add_argument('--manifest-sha', required=True)
    arguments = parser.parse_args()
    verify_staged_raspa_assets(arguments.directory, arguments.manifest_sha)
