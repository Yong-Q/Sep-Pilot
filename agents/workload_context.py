"""Trusted, bounded observations for native resource requests; no execution or writes."""
import copy
import hashlib
import math
import os
from pathlib import Path
import re
import stat

from .workload_budget import (MAX_FILE_BYTES, _owned_fd, collect_workload_facts)
from .cdft_grid_guard import UnsupportedNativeGrid

MAX_EXECUTABLE_BYTES = 256 * 1024 * 1024


def _read_regular(path, *, limit, executable=False):
    """Walk every path component without following links; bind bytes to stable stat."""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute non-traversing path required')
    fd = _owned_fd(Path('/'), path)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit
                or executable and not before.st_mode & 0o111):
            raise ValueError('bounded regular file required')
        chunks, size = [], 0
        while True:
            block = os.read(fd, min(1024 * 1024, limit + 1 - size))
            if not block:
                break
            chunks.append(block); size += len(block)
            if size > limit:
                raise ValueError('file bound exceeded')
        after = os.fstat(fd)
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(before, key) != getattr(after, key) for key in fields) or size != before.st_size:
            raise ValueError('file changed during observation')
        data = b''.join(chunks)
        return data, hashlib.sha256(data).hexdigest()
    finally:
        os.close(fd)


def _native_bounds(data, expected_mesh=None):
    """Parse the dft.cpp input stream through framework rows, not guessed labels.

    The native grid is min(Nmax, int(cell_length / dx) + 1), with pdx/y/z=1.
    Rotational allocation uses Nutation * Precession * Rotation. Site bounds
    cover both type tables and molecule-site tables for every gas.
    """
    tokens = iter(data.decode('ascii').split())
    def take():
        return next(tokens)
    def labels(expected):
        if [take() for _ in expected.split()] != expected.split():
            raise ValueError('unrecognized native header')
    def integer(maximum, minimum=1):
        raw = take()
        if not re.fullmatch(r'[+]?\d+', raw):
            raise ValueError('invalid native integer')
        value = int(raw)
        if not minimum <= value <= maximum:
            raise ValueError('native count outside observation bound')
        return value
    def numbers(count):
        values = [float(take()) for _ in range(count)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('nonfinite native numeric field')
        return values
    take(); numbers(2)  # Native tip, massmof, Henry.
    labels('Nmaxx Nmaxy Nmaxz'); caps = [integer(1000000) for _ in range(3)]
    labels('Lxa Lyb Lzc dx angle_a angle_b angle_c'); cell = numbers(7)
    if any(value <= 0 for value in cell[:4]):
        raise ValueError('positive native lengths and spacing required')
    expected_mesh = expected_mesh or {}
    if ('ngrid' in expected_mesh and any(cap != expected_mesh['ngrid'] for cap in caps)
            or 'dl' in expected_mesh and cell[3] != float(f"{expected_mesh['dl']:.6f}")):
        raise ValueError('declared mesh differs from actual prepared native inputs')
    if any(not math.isfinite(length / cell[3]) or length / cell[3] > 2**31 - 2 for length in cell[:3]):
        raise ValueError('native grid count conversion exceeds safe signed integer bound')
    from .cdft_grid_guard import native_grid_axes
    grid = math.prod(native_grid_axes(data, expected_mesh))
    if grid > 2**31 - 1:
        raise ValueError('native grid exceeds safe integer bound')
    labels('Nutation Precession Rotation'); rotations = math.prod(integer(1000000) for _ in range(3))
    if rotations > 2**31 - 1:
        raise ValueError('native rotations exceed safe integer bound')
    labels('Temperature(K) Kapa Delta_Max Torr'); numbers(4)
    labels('Kind_of_gas'); gases = integer(64)
    labels('Name Epsilon(K) Sigma(A) Bulk_Density(molec/A^3)')
    for _ in range(gases):
        integer(1000000, 0); numbers(3)
    labels('Special_case'); special = integer(4096, 0)
    labels('Id1 Id2 Epsilon(K) Sigma(A)')
    for _ in range(special):
        integer(gases - 1, 0); integer(gases - 1, 0); numbers(2)
    labels('Read_Vext EOS Cutoff deos N3_cutoff Diele_con Cutoff_inside cut_coul'); numbers(8)
    labels('Mass(g/mol) Ds_Knudsen(m^2/s) MFV_parameter'); numbers(3)
    site_types = []
    for _ in range(gases):
        labels('Kind_of_solvent_atoms'); count = integer(4096); site_types.append(count)
        labels('ID diameter(A) Epsilon(K) charge')
        for _ in range(count):
            integer(count - 1, 0); numbers(3)
    sites = []
    for gas in range(gases):
        if not re.fullmatch(r'Number_of_solvent_atoms\d*', take()):
            raise ValueError('unrecognized native site header')
        count = integer(4096); sites.append(count)
        labels('ID x y z')
        for _ in range(count):
            integer(site_types[gas] - 1, 0); numbers(3)
    labels('Number_of_solute_atoms'); atoms = integer(1000000)
    labels('ID x y z diameter(A) Epsilon(K) charge')
    for _ in range(atoms):
        integer(2**31 - 1, 0); numbers(6)
    if next(tokens, None) is not None:
        raise ValueError('unrecognized trailing native input')
    return {'max_grid_points': grid, 'max_gases': gases,
            'max_sites_per_gas': max(site_types + sites),
            'max_framework_atoms': atoms, 'max_rotations': rotations}


def enrich_resource_request(request, *, owned_root, cdft_executable=None):
    """Replace supplied node workload facts with observations of exact owned inputs.

    Only prepared cDFT submit receives native_bounds and executable_sha256.
    Unsupported tools and unprepared pipelines retain the collector's explicit
    status. Missing/invalid binary or malformed native input never becomes ready.
    This function grants no resource, scheduler, submission or estimate authority.
    """
    if not isinstance(request, dict) or not isinstance(request.get('nodes', []), list):
        raise ValueError('resource request nodes must be a list')
    result = copy.deepcopy(request)
    root = Path(owned_root)
    for node in result.get('nodes', []):
        if not isinstance(node, dict):
            raise ValueError('resource request node must be an object')
        node.pop('executable_sha256', None)
        args = node.get('arguments') if isinstance(node.get('arguments'), dict) else {}
        tool = node.get('tool')
        try:
            # Validate the scope's full ancestor chain before collector IO.
            root_fd = _owned_fd(Path('/'), root, directory=True)
            os.close(root_fd)
            facts = collect_workload_facts(tool, args, owned_root=root)
            node['workload_facts'] = facts
            if tool != 'run_cdft' or facts.get('status') != 'ready':
                continue
            if cdft_executable is None:
                raise ValueError('configured executable required')
            _, executable_sha = _read_regular(cdft_executable, limit=MAX_EXECUTABLE_BYTES, executable=True)
            bounds = []
            directory = Path(args['input_dir'])
            for item in facts['inputs']:
                path = directory / item['path']
                path.relative_to(root)
                data, sha = _read_regular(path, limit=MAX_FILE_BYTES)
                if sha != item['sha256'] or len(data) != item['bytes']:
                    raise ValueError('input no longer matches collected evidence')
                bounds.append(_native_bounds(data, {key: args[key] for key in ('ngrid', 'dl') if key in args}))
            facts['native_bounds'] = {key: max(bound[key] for bound in bounds) for key in bounds[0]}
            facts['native_bounds_schema_version'] = 1
            node['executable_sha256'] = executable_sha
        except UnsupportedNativeGrid as error:
            node['workload_facts'] = {'tool': tool, 'status': 'invalid_workload',
                'reason': 'unsupported_native_grid', 'grid_axes': list(error.grid_axes),
                'error': str(error)}
        except (OSError, ValueError, TypeError, KeyError, UnicodeError, StopIteration, OverflowError):
            node['workload_facts'] = {'tool': tool, 'status': 'invalid_workload',
                                      'reason': 'native_input_or_executable_verification_failed'}
    return result
