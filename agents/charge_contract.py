"""Read-only charge validation; never force a scientific model to neutrality."""
import hashlib
import io
import math
from pathlib import Path


def validate_framework_charges(cif_path, expected_net_charge, charge_source, tolerance=1e-6, output_path=None):
    from ase.io.cif import parse_cif
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
        from .state_io import write_checkpoint
        destination = Path(output_path).resolve()
        if destination.exists():
            raise ValueError('Charge-validation report already exists; use a fresh output_path.')
        write_checkpoint(destination, report)
        report.update(output_path=str(destination), output_files=[str(destination)])
        return report
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
