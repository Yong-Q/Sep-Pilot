"""Real CPU-only integration; synthetic business data is explicitly labelled."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import socket
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', choices=['A', 'B'], required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.duration <= 50: raise ValueError('probe duration must be 1..50 seconds')
    if not os.environ.get('SLURM_JOB_ID') and not os.environ.get('PBS_JOBID'):
        raise RuntimeError('probe must run in a scheduler job, never on the login node')
    started = time.time()
    monotonic = time.monotonic()
    count, intervals = 0, 2048
    while count == 0 or time.monotonic() - monotonic < args.duration:
        total = 4.0 + 2.0
        for i in range(1, intervals):
            x = i / intervals
            total += (4 if i % 2 else 2) * 4 / (1 + x*x)
        answer = total / (3 * intervals)
        count += 1
    error = abs(answer - math.pi)
    if error > 1e-10: raise RuntimeError(f'numerical integration failed known-answer check: {error}')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'kind': 'real_numeric_calculation', 'label': args.label, 'job_id': os.environ.get('SLURM_JOB_ID') or os.environ['PBS_JOBID'],
        'hostname': socket.gethostname(), 'glibc': platform.libc_ver(), 'pid': os.getpid(), 'started_at': started,
        'finished_at': time.time(), 'duration_seconds': time.monotonic()-monotonic, 'cpu_seconds': time.process_time(),
        'iterations': count, 'integral': answer, 'expected': math.pi, 'absolute_error': error, 'validation_passed': True,
        'business_results_are_mock': True}
    temporary = output / 'compute_receipt.json.tmp'
    temporary.write_text(json.dumps(receipt, ensure_ascii=False))
    os.replace(temporary, output / 'compute_receipt.json')
    with (output / 'business_mock.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['data_kind', 'label', 'gas', 'mock_henry_value'])
        writer.writerow(['MOCK_NOT_SCIENTIFIC_RESULT', args.label, 'Kr' if args.label == 'A' else 'Xe', .25 if args.label == 'A' else .75])
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
