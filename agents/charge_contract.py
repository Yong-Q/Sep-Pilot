"""Read-only charge validation; never force a scientific model to neutrality."""
import hashlib
import io
import math
import json
import os
import stat
import tempfile
from pathlib import Path


def _unique_report_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('duplicate report field')
        result[key] = value
    return result


def _report_json(report):
    return json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _identical_charge_report_sha256(destination, report):
    """Strict JSON equality and byte provenance, with no existing-file writes."""
    try:
        fd = os.open(destination, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 64*1024*1024:
                raise ValueError('existing charge report is not a bounded regular file')
            raw = stream.read(64*1024*1024 + 1)
        if len(raw) > 64*1024*1024:
            raise ValueError('existing charge report exceeds read bound')
        def invalid_constant(_value): raise ValueError('non-finite report value')
        previous = json.loads(raw, object_pairs_hook=_unique_report_object, parse_constant=invalid_constant)
        if not isinstance(previous, dict) or _report_json(previous) != _report_json(report):
            raise ValueError('charge report differs')
        return hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, UnicodeError):
        raise ValueError('Charge-validation report already exists with different or invalid data; use a fresh output_path.') from None


def _publish_charge_report(destination, report):
    """Publish once; a concurrent identical writer may be reused, never replaced."""
    if destination.exists() or destination.is_symlink():
        return True, _identical_charge_report_sha256(destination, report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = _report_json(report).encode('utf-8')
    temporary = None
    try:
        fd, name = tempfile.mkstemp(prefix='.charge-report-', suffix='.json', dir=destination.parent)
        temporary = Path(name)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return True, _identical_charge_report_sha256(destination, report)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
        return False, hashlib.sha256(raw).hexdigest()
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def validate_framework_charges(cif_path, expected_net_charge, charge_source, tolerance=1e-6, output_path=None):
    path = Path(cif_path).resolve(strict=True)
    if not charge_source.strip():
        raise ValueError('An approved charge source must be declared.')
    target, tolerance = float(expected_net_charge), float(tolerance)
    if not math.isfinite(target) or not 0 <= tolerance <= 1e-4:
        raise ValueError('Finite expected cell charge and tolerance in [0, 1e-4] required.')
    if path.is_dir():
        files = sorted(path.glob('*_pacmof.cif'))
        if not files:
            raise ValueError('Charge-validation directory contains no *_pacmof.cif files.')
        if not output_path:
            raise ValueError('Batch charge validation requires an explicit output_path.')
        results = [validate_framework_charges(str(cif), target, charge_source, tolerance) for cif in files]
        ok = all(item.get('ok') is True for item in results)
        report = {'ok': ok, 'read_only_inputs': True, 'input_modified': False,
                  'charge_source': charge_source, 'expected_net_charge': target,
                  'tolerance_e': tolerance, 'n_cifs': len(files), 'n_valid': sum(bool(r.get('ok')) for r in results),
                  'results': results, 'status': 'validated' if ok else 'charge_mismatch'}
        if not ok:
            report['error'] = 'At least one framework charge disagrees with the declared model.'
        requested = Path(output_path).expanduser()
        if requested.is_symlink():
            raise ValueError('Charge-validation report cannot be a symlink; use a fresh output_path.')
        destination = requested.resolve()
        reused, report_sha256 = _publish_charge_report(destination, report)
        report.update(output_path=str(destination), output_files=[str(destination)],
                      reused_existing_report=reused, report_sha256=report_sha256)
        return report
    try:
        item = _validate_single_framework_charges(path, target, charge_source, tolerance)
    except Exception as error:
        if not output_path:
            raise
        raw = path.read_bytes() if path.is_file() and path.stat().st_size <= 16_000_000 else None
        item = {'ok': False, 'read_only': True, 'input_modified': False, 'cif_path': str(path),
                'sha256': hashlib.sha256(raw).hexdigest() if raw is not None else None,
                'charge_source': charge_source, 'expected_net_charge': target, 'tolerance_e': tolerance,
                'cell_net_charge': None, 'residual_e': None, 'status': 'charge_validation_failed',
                'simulation_ready': False, 'error': str(error)}
    if not output_path:
        return item
    report = {**item, 'read_only_inputs': True, 'n_cifs': 1,
              'n_valid': int(item.get('ok') is True), 'results': [item],
              'charge_unit': 'e', 'finite_charges': True if item.get('cell_net_charge') is not None else None,
              'validation_method': 'symmetry_expanded_occupancy_weighted_site_sum',
              'charge_source_provenance': {'declared_by': 'caller', 'independently_verified': False}}
    requested = Path(output_path).expanduser()
    if requested.is_symlink():
        raise ValueError('Charge-validation report cannot be a symlink; use a fresh output_path.')
    destination = requested.resolve()
    reused, report_sha256 = _publish_charge_report(destination, report)
    report.update(output_path=str(destination), output_files=[str(destination)],
                  reused_existing_report=reused, report_sha256=report_sha256)
    return report


def _identity_p1_atoms(block, site_count):
    """Read explicit P1 sites without ASE's quadratic symmetry deduplication.

    Preserve its periodic fractional-coordinate duplicate tolerance. All other
    space groups retain the existing ASE expansion and occupancy semantics.
    """
    import numpy as np
    from ase.spacegroup.spacegroup import spacegroup_from_data
    from ase.io.cif import CIFBlock
    declarations = []
    for tag in ('_space_group.it_number', '_space_group_it_number', '_symmetry_int_tables_number'):
        value = block.get(tag)
        if value is not None:
            declarations.append(int(value))
            if float(value) != int(value):
                raise ValueError('Space-group number must be integral.')
    for tag in ('_space_group_name_h-m_alt', '_symmetry_space_group_name_h-m'):
        value = block.get(tag)
        if value is not None:
            declarations.append(int(CIFBlock('declared_symmetry', {tag: value}).get_spacegroup(True)))
    if len(set(declarations)) > 1:
        raise ValueError('Conflicting CIF symmetry declarations.')
    group = block.get_spacegroup(subtrans_included=True)
    if int(group) != 1:
        return None
    if declarations and any(number != 1 for number in declarations):
        raise ValueError('Declared symmetry conflicts with identity P1.')
    for tag in ('_space_group_symop_operation_xyz', '_space_group_symop.operation_xyz',
                '_symmetry_equiv_pos_as_xyz'):
        declared = block.get(tag)
        if declared is not None:
            operations_text = [declared] if isinstance(declared, str) else declared
            if not isinstance(operations_text, (list, tuple)) or len(operations_text) != 1:
                raise ValueError('P1 must declare exactly one identity symmetry operation.')
            declared_group = spacegroup_from_data(no=1, setting=1, sitesym=operations_text,
                                                 subtrans=[(0.0, 0.0, 0.0)])
            declared_ops = declared_group.get_symop()
            if (len(declared_ops) != 1 or not np.array_equal(declared_ops[0][0], np.eye(3))
                    or not np.array_equal(np.mod(declared_ops[0][1], 1.0), np.zeros(3))):
                raise ValueError('Conflicting nonidentity operation in P1 CIF.')
    operations = group.get_symop()
    if (len(operations) != 1 or not np.array_equal(operations[0][0], np.eye(3))
            or not np.array_equal(np.mod(operations[0][1], 1.0), np.zeros(3))):
        raise ValueError('P1 requires exactly the identity symmetry operation.')
    for tag in ('_atom_site_symmetry_multiplicity', '_atom_site_site_symmetry_multiplicity'):
        values = block.get(tag)
        if values is not None:
            if (not isinstance(values, (list, tuple)) or len(values) != site_count
                    or any(not math.isfinite(float(value)) or float(value) != 1 for value in values)):
                raise ValueError('Explicit P1 site multiplicities must all equal one.')
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    atoms = block.get_unsymmetrized_structure()
    if len(atoms) != site_count or atoms.cell.rank != 3:
        raise ValueError('P1 coordinates and charge rows require one full periodic cell.')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError('P1 coordinates and cell must be finite.')
    coordinates = np.mod(atoms.get_scaled_positions(), 1.0)
    if not np.isfinite(coordinates).all():
        raise ValueError('P1 fractional coordinates must be finite.')
    tree = cKDTree(coordinates, boxsize=1.0)
    # Query one point at a time: dense invalid inputs cannot allocate an N²
    # candidate-pairs array. Verify ASE's strict component-wise '<' boundary.
    for index, position in enumerate(coordinates):
        for other in tree.query_ball_point(position, r=1e-3, p=np.inf):
            if other <= index:
                continue
            delta = np.abs(position - coordinates[other])
            if np.all((delta < 1e-3) | (np.abs(delta - 1.0) < 1e-3)):
                raise ValueError('Duplicate/unmapped P1 charge rows; no silent site dropping.')
    atoms.new_array('spacegroup_kinds', np.arange(site_count, dtype=int))
    return atoms


def _validate_single_framework_charges(path, target, charge_source, tolerance):
    """Preserve the original single-CIF charge mathematics and return semantics."""
    from ase.io.cif import parse_cif
    if path.stat().st_size > 16_000_000:
        raise ValueError('CIF exceeds bounded reader size')
    raw = path.read_bytes()
    blocks = [b for b in parse_cif(io.BytesIO(raw)) if b.has_structure()]
    if len(blocks) != 1:
        raise ValueError('Exactly one structural CIF block required.')
    block = blocks[0]
    charges = block.get('_atom_site_charge')
    if charges is None:
        raise ValueError('Missing per-site charges; zero filling or forcefield substitution forbidden.')
    charges = [float(q) for q in charges]
    if not charges or not all(math.isfinite(q) for q in charges):
        raise ValueError('Every site needs a finite charge.')
    symmetry_tags = ('_space_group_it_number', '_symmetry_int_tables_number', '_space_group_name_h-m_alt',
                     '_symmetry_space_group_name_h-m', '_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz')
    if not any(block.get(tag) is not None for tag in symmetry_tags):
        raise ValueError('Declare cell symmetry; do not assume P1.')
    atoms = _identity_p1_atoms(block, len(charges))
    if atoms is None:
        atoms = block.get_atoms(store_tags=True, fractional_occupancies=True)
    if atoms.cell.rank != 3:
        raise ValueError('A full periodic cell is required.')
    if any(len(species) > 1 for species in atoms.info.get('occupancy', {}).values()):
        raise ValueError('Mixed-species disorder needs an approved resolved model.')
    kinds = atoms.arrays.get('spacegroup_kinds')
    if kinds is None:
        if len(atoms) != len(charges):
            raise ValueError('Cannot map expanded sites to charges.')
        kinds = list(range(len(charges)))
    occupancy = [float(o) for o in block.get('_atom_site_occupancy', [1.0] * len(charges))]
    if len(occupancy) != len(charges) or any(not math.isfinite(o) or not 0 <= o <= 1 for o in occupancy):
        raise ValueError('Complete occupancies in [0, 1] required.')
    multiplicities = [0] * len(charges)
    for kind in kinds:
        if not 0 <= int(kind) < len(charges):
            raise ValueError('Invalid expanded-site mapping.')
        multiplicities[int(kind)] += 1
    if any(m == 0 and o > 0 for m, o in zip(multiplicities, occupancy)):
        raise ValueError('Duplicate/unmapped charge rows; no silent site dropping.')
    total = math.fsum(q * o * m for q, o, m in zip(charges, occupancy, multiplicities))
    residual = total - target
    if not math.isfinite(total) or not math.isfinite(residual):
        raise ValueError('Expanded cell charge and residual must be finite.')
    ok = abs(residual) <= tolerance
    return {'ok': ok, 'read_only': True, 'input_modified': False, 'cif_path': str(path),
            'sha256': hashlib.sha256(raw).hexdigest(), 'charge_source': charge_source,
            'expected_net_charge': target, 'cell_net_charge': total, 'residual_e': residual,
            'tolerance_e': tolerance, 'asymmetric_site_count': len(charges), 'expanded_site_count': len(atoms),
            'multiplicities': multiplicities, 'occupancies': occupancy,
            'status': 'validated' if ok else 'charge_mismatch', 'simulation_ready': False,
            'counterion_or_charge_compensation_review_required': target != 0,
            **({} if ok else {'error': 'Cell charge disagrees with the declared model; never shift residual onto an atom.',
                             'next_action': 'Review provenance, symmetry/occupancy and counterions with main chat; model changes require user approval.'})}
