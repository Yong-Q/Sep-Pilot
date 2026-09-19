"""
Parse cDFT output files and collect results into a CSV.

For each MOF in {work_dir}/input_dir/:
  - output.dat  → gas adsorption densities (mol/L) and percentages
  - hr.dat       → Henry constants (mol/L/atm)

Output: {work_dir}/results.csv  (or custom path)

output.dat format (after "Molecule" header line):
  0 0.146259 (53.9191%)  65552.8
  1 0.124997 (46.0809%)  126.581
   total density: 0.000163378

hr.dat format:
  298 unit:mol/(L*atm)
  0 0.333726
  298 unit:mol/(L*atm)
  1 0.246679
"""

import os
import re
import csv
import argparse
from pathlib import Path


def parse_output_dat(path):
    """
    Returns dict: {gas_idx(int): {'density': float, 'percentage': float}}
    Also returns total_density (float or None).
    """
    result = {}
    total  = None
    in_mol = False
    mol_re = re.compile(r'^\s*(\d+)\s+([\d.eE+\-]+)\s+\(([\d.]+)%\)')
    tot_re = re.compile(r'total density[:\s]+([\d.eE+\-]+)', re.IGNORECASE)

    try:
        with open(path) as f:
            for line in f:
                if 'Molecule' in line and 'Average_density' in line:
                    in_mol = True
                    continue
                if in_mol:
                    m = mol_re.match(line)
                    if m:
                        idx = int(m.group(1))
                        result[idx] = {
                            'density':    float(m.group(2)),
                            'percentage': float(m.group(3)),
                        }
                        continue
                    t = tot_re.search(line)
                    if t:
                        total = float(t.group(1))
                    if line.strip() == '':
                        in_mol = False
    except (OSError, ValueError):
        pass
    return result, total


def parse_hr_dat(path):
    """
    Returns dict: {gas_idx(int): henry_constant(float)}
    """
    result = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or 'unit:' in line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    try:
                        result[int(parts[0])] = float(parts[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    return result


def collect_results(work_dir, gases, output_csv=None):
    """
    Scan {work_dir}/input_dir/ for MOF subdirs, parse output.dat + hr.dat,
    write results.csv.

    Parameters
    ----------
    work_dir : str
        Job work directory (contains input/ and input_dir/).
    gases : list of str
        Gas names in order.  Length determines how many columns to expect.
    output_csv : str, optional
        Output CSV path.  Defaults to {work_dir}/results.csv.

    Returns
    -------
    list of dict  (one per MOF)
    """
    input_dir_path = os.path.join(work_dir, 'input_dir')
    if output_csv is None:
        output_csv = os.path.join(work_dir, 'results.csv')

    n_gases = len(gases)

    # Build CSV header
    header = ['MOF']
    for i, g in enumerate(gases):
        header.append(f'{g}_density_mol_L')
    for i, g in enumerate(gases):
        header.append(f'{g}_percentage')
    header.append('total_density_molec_A3')
    for i, g in enumerate(gases):
        header.append(f'{g}_henry_mol_L_atm')

    rows = []

    if not os.path.isdir(input_dir_path):
        return rows

    for entry in sorted(Path(input_dir_path).iterdir()):
        if not entry.is_dir():
            continue
        mof_name = entry.name
        out_path  = entry / 'output.dat'
        hr_path   = entry / 'hr.dat'

        adsorption, total = parse_output_dat(str(out_path))
        henry             = parse_hr_dat(str(hr_path))

        row = {'MOF': mof_name}

        for i in range(n_gases):
            row[header[1 + i]] = adsorption.get(i, {}).get('density', '')

        for i in range(n_gases):
            row[header[1 + n_gases + i]] = adsorption.get(i, {}).get('percentage', '')

        row['total_density_molec_A3'] = total if total is not None else ''

        for i in range(n_gases):
            row[header[1 + 2 * n_gases + 1 + i]] = henry.get(i, '')

        rows.append(row)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect cDFT results into CSV')
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--gases',    nargs='+', required=True)
    parser.add_argument('--output',   default=None)
    args = parser.parse_args()

    rows = collect_results(
        work_dir=args.work_dir,
        gases=args.gases,
        output_csv=args.output,
    )
    out = args.output or os.path.join(args.work_dir, 'results.csv')
    print(f"Collected {len(rows)} MOFs → {out}")
    if rows:
        success = sum(1 for r in rows if any(r[k] != '' for k in r if 'density' in k))
        print(f"  Successful (has density): {success}/{len(rows)}")
