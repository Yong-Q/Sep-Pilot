"""Execute the String generator's manifest; solver failures remain failures."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _native_info(log):
    matches = re.findall(r'\binfo:\s*([01])\s+(\d+)\s+(\d+)', log)
    if not matches:
        return {'converged': None, 'iterations': None, 'expanded_atoms': None}
    flag, iterations, atoms = matches[-1]
    return {'converged': flag == '1', 'iterations': int(iterations), 'expanded_atoms': int(atoms)}


def _output_protocol():
    """Expected adapter ABI, not an attestation of arbitrary installed code."""
    return {'id': 'tutrast_string_seven_columns_v1', 'column_count': 7,
        'columns': ['fractional_a', 'fractional_b', 'fractional_c', 'orientation_alpha',
                    'orientation_beta', 'orientation_gamma', 'potential'],
        'energy_unit': None,
        'provenance': {'declared_by_adapter': True, 'source_verified': False, 'binary_source_binding': False}}


def _publish_receipt(receipt, path):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.string-results-', suffix='.json', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # link() atomically creates a new name and refuses an existing file,
        # including a concurrent result or dangling symlink. Never replace().
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_manifest(manifest, executable):
    manifest = Path(manifest).resolve()
    executable = Path(executable).resolve()
    receipt_path = manifest.parent / 'string_results.json'
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError('String receipt already exists; use a fresh output directory')
    executable_sha256 = _sha256(executable)
    manifest_bytes = manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    reader = csv.DictReader(io.StringIO(manifest_bytes.decode('utf-8'), newline=''))
    if not {'structure', 'direction', 'input_dat'} <= set(reader.fieldnames or []):
        raise ValueError('String manifest lacks structure/direction/input_dat columns')
    rows = list(reader)
    if not rows:
        raise ValueError('String manifest contains no runnable inputs')
    sources = [Path(row['input_dat']).resolve() for row in rows]
    if len(set(sources)) != len(sources):
        raise ValueError('String manifest repeats an input')
    if len({source.parent for source in sources}) != len(sources):
        raise ValueError('String manifest repeats a native output directory')
    for source in sources:
        if not source.is_file() or not source.is_relative_to(manifest.parent):
            raise ValueError('String input is missing or outside its manifest workspace')
        for existing in (source.parent / 'string_path.dat', source.parent / 'run.log'):
            if existing.exists() or existing.is_symlink():
                raise FileExistsError('String result/log already exists; use a fresh output directory')
    results = []
    for row, source in zip(rows, sources):
        output = source.parent / 'string_path.dat'
        run_log = source.parent / 'run.log'
        source_sha256 = _sha256(source)
        declared_points = re.search(r'(?m)^#_of_points[^\n]*\n\s*(\d+)\b', source.read_text())
        expected_points = int(declared_points[1]) if declared_points else None
        print(f"Running {row['structure']} direction {row['direction']}: {source}", flush=True)
        started = time.monotonic()
        with run_log.open('x') as log:
            subprocess.run([str(executable), str(source), str(output)],
                           cwd=source.parent, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        wall_elapsed = time.monotonic() - started
        point_count = 0
        with output.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 7 or not all(math.isfinite(float(value)) for value in values):
                    raise ValueError('String solver did not produce a finite seven-column path')
                point_count += 1
        if point_count < 2:
            raise ValueError('String solver did not produce a finite seven-column path')
        if expected_points is not None and point_count != expected_points:
            raise ValueError('String path point count does not match its native input')
        if _sha256(source) != source_sha256:
            raise ValueError('String input changed during native execution')
        results.append({'structure': row['structure'], 'direction': row['direction'],
                        'input_path': str(source), 'input_sha256': source_sha256,
                        'output_path': str(output), 'output_sha256': _sha256(output),
                        'point_count': point_count, 'expected_point_count': expected_points,
                        'run_log_path': str(run_log), 'run_log_sha256': _sha256(run_log),
                        'adapter_native_wall_elapsed_seconds': wall_elapsed,
                        'native_info': _native_info(run_log.read_text(errors='replace'))})
        print(f'Verified native String path: {output}', flush=True)
    if _sha256(manifest) != manifest_sha256:
        raise ValueError('String manifest changed during native execution')
    if _sha256(executable) != executable_sha256:
        raise ValueError('String executable changed during native execution')
    receipt = {'schema_version': 2, 'success': True, 'manifest_path': str(manifest),
               'manifest_sha256': manifest_sha256, 'executable_path': str(executable),
               'executable_sha256': executable_sha256, 'output_protocol': _output_protocol(),
               'results': results,
               'validation_scope': 'Finite native paths and engine-reported flags only; short budgets do not establish scientific convergence or transport coefficients'}
    _publish_receipt(receipt, receipt_path)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--executable', required=True)
    args = parser.parse_args()
    run_manifest(args.manifest, args.executable)


if __name__ == '__main__':
    main()
