#!/usr/bin/env python3
"""Run PACMOF as an attrition-aware batch and persist item-level outcomes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def _write_manifest(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_batch(cif_dir, output_dir, charge_one, completion_policy='best_effort'):
    source = Path(cif_dir).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    inputs = sorted(source.glob('*.cif'))
    successes, failures = [], []
    for cif in inputs:
        try:
            charge_one(cif, destination)
            output = destination / f'{cif.stem}_pacmof.cif'
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError('PACMOF returned without a nonempty output CIF')
            successes.append({'file': cif.name, 'output': output.name})
        except Exception as error:
            failures.append({'file': cif.name, 'error': str(error)[:500]})
    minimum = 1
    accepted = (len(successes) == len(inputs) and bool(inputs)) if completion_policy == 'strict' \
        else len(successes) >= minimum
    result = {
        'schema_version': 1,
        'method': 'pacmof',
        'completion_policy': completion_policy,
        'artifact_presence_floor': minimum,
        'attempted_count': len(inputs),
        'successful_count': len(successes),
        'failed_count': len(failures),
        'accepted': accepted,
        'successes': successes,
        'failures': failures,
    }
    _write_manifest(destination / 'charge_manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--completion-policy', choices=('best_effort', 'strict'), default='best_effort')
    args = parser.parse_args()

    import sys
    sys.path.insert(0, '/home/user/pacmof')
    from pacmof.pacmof import get_charges_single_serial

    def charge_one(cif, output_dir):
        get_charges_single_serial(
            str(cif), create_cif=True, path_to_output_dir=str(output_dir), add_string='_pacmof')

    result = run_batch(
        args.cif_dir, args.output_dir, charge_one,
        completion_policy=args.completion_policy,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
