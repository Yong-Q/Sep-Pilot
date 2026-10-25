"""Execute the String generator's manifest; solver failures remain failures."""
import argparse
import csv
import math
from pathlib import Path
import subprocess


def run_manifest(manifest, executable):
    manifest = Path(manifest).resolve()
    executable = Path(executable).resolve()
    with manifest.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'structure', 'direction', 'input_dat'} <= set(reader.fieldnames or []):
            raise ValueError('String manifest lacks structure/direction/input_dat columns')
        rows = list(reader)
    if not rows:
        raise ValueError('String manifest contains no runnable inputs')
    sources = [Path(row['input_dat']).resolve() for row in rows]
    if len(set(sources)) != len(sources):
        raise ValueError('String manifest repeats an input')
    for source in sources:
        if not source.is_file() or not source.is_relative_to(manifest.parent):
            raise ValueError('String input is missing or outside its manifest workspace')
    for row, source in zip(rows, sources):
        output = source.parent / 'string_path.dat'
        if output.exists():
            raise FileExistsError('String result already exists; use a fresh output directory')
        print(f"Running {row['structure']} direction {row['direction']}: {source}", flush=True)
        with (source.parent / 'run.log').open('x') as log:
            subprocess.run([str(executable), str(source), str(output)],
                           cwd=source.parent, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        values = [line.split() for line in output.read_text().splitlines() if line.strip()]
        if len(values) < 2 or any(len(value) != 7 or not all(
                math.isfinite(float(x)) for x in value) for value in values):
            raise ValueError('String solver did not produce a finite seven-column path')
        print(f'Verified native String path: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--executable', required=True)
    args = parser.parse_args()
    run_manifest(args.manifest, args.executable)


if __name__ == '__main__':
    main()
