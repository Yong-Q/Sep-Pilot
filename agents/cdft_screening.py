"""Evidence-linked single-component density ratios from canonical cDFT CSVs."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


def analyze_screening(params, session_root):
    root = Path(session_root).resolve()
    output = Path(params['output_csv']).resolve()
    selected_dir = Path(params['output_dir']).resolve()
    if not output.is_relative_to(root) or not selected_dir.is_relative_to(root):
        raise ValueError('analysis outputs must stay inside this session')
    if output.exists() or selected_dir.exists():
        raise ValueError('analysis output already exists; use fresh output paths')
    if output.is_relative_to(selected_dir):
        raise ValueError('ranking CSV must be separate from selected structure directory')
    gases = params['gases']
    if len(gases) != 2 or len(set(gases)) != 2:
        raise ValueError('exactly two different gas names are required')
    n = params['top_n']
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('top_n must be a positive integer')
    primary = Path(params['data_csv']).resolve()
    secondary = Path(params.get('input_path') or primary).resolve()

    def read(source, gas):
        if not source.is_file() or source.stat().st_size > 16_000_000:
            raise ValueError('input CSV missing or exceeds 16MB analysis bound')
        column = gas + '_density_mol_L'
        rows = {}
        with source.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or 'MOF' not in reader.fieldnames or column not in reader.fieldnames:
                raise ValueError('canonical MOF and density_mol_L columns required; units cannot be guessed')
            for row in reader:
                name = row.get('MOF', '')
                if not name or Path(name).name != name or name in {'.', '..'} or name in rows or len(rows) >= 10000:
                    raise ValueError('missing, unsafe or duplicate material identity, or too many rows')
                value = float(row[column])
                if not math.isfinite(value) or value < 0:
                    raise ValueError('missing, nonfinite or negative density')
                rows[name] = value
        if not rows:
            raise ValueError('empty cDFT evidence')
        return rows

    numerator, denominator = gases
    a, b = read(primary, numerator), read(secondary, denominator)
    if set(a) != set(b) or n > len(a):
        raise ValueError('gas material sets must match and cover requested top_n')
    if any(value <= 0 for value in b.values()):
        raise ValueError('uptake ratios require positive denominator densities')
    rows = [{'MOF': name, numerator + '_density_mol_L': a[name],
             denominator + '_density_mol_L': b[name], 'uptake_ratio': a[name] / b[name]}
            for name in a]
    if any(not math.isfinite(row['uptake_ratio']) for row in rows):
        raise ValueError('nonfinite uptake ratio')
    rows.sort(key=lambda row: (-row['uptake_ratio'], row['MOF']))
    source_dir = Path(params['cif_dir']).resolve()
    files = []
    for row in rows[:n]:
        file = source_dir / (row['MOF'] + '.cif')
        if not file.is_file() or file.is_symlink() or not file.stat().st_size:
            raise ValueError('ranked material has no exact nonempty source CIF: ' + row['MOF'])
        files.append(file)
    definition = f'density uptake ratio {numerator}/{denominator} from supplied columns; not IAST or mixture selectivity'
    provenance = [{'path': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
                  for file in dict.fromkeys([primary, secondary, *files])]
    manifest = {'ratio_definition': definition, 'density_unit': 'mol/L',
                'conditions_verified': False,
                'component_mode_verified': False,
                'condition_note': 'CSV alone does not establish single-component versus mixture mode, temperature or pressure. Verify separate single-component source jobs and matching conditions before interpreting as a single-component uptake ratio.',
                'gases': gases, 'top_n': n, 'selected_materials': [row['MOF'] for row in rows[:n]],
                'source_evidence': provenance}
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.cdft-selection-', dir=selected_dir.parent))
    temporary_csv = None
    try:
        for file in files:
            shutil.copy2(file, staging / file.name)
        (staging / 'selection_manifest.json').write_text(json.dumps(manifest, indent=2))
        fd, temporary_csv = tempfile.mkstemp(prefix='.cdft-ranking-', dir=output.parent)
        with os.fdopen(fd, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        # Exclusive publication prevents a rerun from overwriting evidence.
        os.link(temporary_csv, output)
        os.rename(staging, selected_dir)
    finally:
        if temporary_csv and Path(temporary_csv).exists():
            Path(temporary_csv).unlink()
        if staging.exists():
            shutil.rmtree(staging)
    return {'ok': True, **manifest, 'rows': rows, 'row_count': len(rows),
            'output_csv': str(output), 'output_dir': str(selected_dir),
            'output_files': [str(output), str(selected_dir / 'selection_manifest.json')]}
