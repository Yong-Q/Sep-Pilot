"""Explicit operator import of immutable parameter assets, never conversion."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.forcefield_catalog import parse_towhee_metadata
from agents.state_io import write_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reindex', action='store_true', help='explicitly regenerate derived metadata, never overwrite original parameter bytes')
    args = parser.parse_args()
    policy = json.loads((ROOT / 'env/forcefield_sources.json').read_text())['collections']['towhee']
    source = Path(policy['source_root'])
    destination = ROOT / policy['project_directory']
    files = sorted((source / policy['source_directory']).glob('towhee_ff_*'))
    if not files or len(files) > 512: raise ValueError('unexpected/unbounded forcefield collection')
    plan = [(p, destination / p.name) for p in files] + [(source / 'license.gpl', destination.parent / 'UPSTREAM_LICENSE.gpl')]
    for src, dst in plan:
        if src.is_symlink() or not src.is_file(): raise ValueError('unexpected source asset')
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise ValueError('existing project asset differs; refusing overwrite: ' + str(dst))
    entries = [parse_towhee_metadata(path) for path in files]
    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists(): shutil.copy2(src, dst)
        if hashlib.sha256(src.read_bytes()).digest() != hashlib.sha256(dst.read_bytes()).digest():
            raise ValueError('parameter copy checksum mismatch')
    catalog = {'schema_version': 1, 'imported_at': time.time(), 'source': policy,
        'native_format': 'towhee_ff', 'scientific_assignment_status': 'not_assigned', 'entries': entries,
        'licensing_note': 'Upstream distribution license retained verbatim; individual scientific provenance and redistribution conditions must be reviewed for external distribution.'}
    target = ROOT / policy['catalog']
    if target.exists():
        old = json.loads(target.read_text())
        if old['entries'] != entries:
            if not args.reindex: raise ValueError('catalog changed; explicit update review required')
            write_checkpoint(target, catalog)
    else: write_checkpoint(target, catalog)
    print(json.dumps({'copied_parameter_files': len(files), 'catalog': str(target),
        'opls': [{k:e[k] for k in ('name','nonbonded_type_count','mixing_rule')} for e in entries if e.get('family')=='OPLS']}, ensure_ascii=False))


if __name__ == '__main__': main()
