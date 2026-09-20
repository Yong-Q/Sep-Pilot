#!/usr/bin/env python3
"""Publish selected pormake CIFs at the dataset root for downstream tools."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


SIZE_CLASSES = ('large', 'small')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stage_selected_cifs(output_dir, selection='all', expected_count=None):
    root = Path(output_dir).resolve()
    if selection not in {'all', *SIZE_CLASSES}:
        raise ValueError(f'unsupported size selection: {selection}')
    classes = SIZE_CLASSES if selection == 'all' else (selection,)
    sources = [(size_class, path) for size_class in classes
               for path in sorted((root / size_class).glob('*.cif'))]
    if expected_count is not None and len(sources) != int(expected_count):
        raise ValueError(f'expected {int(expected_count)} selected CIFs, found {len(sources)}')

    by_name = {}
    records = []
    for size_class, source in sources:
        digest = _sha256(source)
        previous = by_name.get(source.name)
        if previous and previous != digest:
            raise ValueError(f'conflicting selected CIF filename: {source.name}')
        by_name[source.name] = digest
        records.append((size_class, source, digest))

    published = []
    for size_class, source, digest in records:
        target = root / source.name
        if target.exists():
            if not target.is_file() or _sha256(target) != digest:
                raise FileExistsError(f'refusing to overwrite different dataset file: {target}')
            mode = 'hardlink' if os.path.samefile(source, target) else 'copy'
        else:
            try:
                os.link(source, target)
                mode = 'hardlink'
            except OSError:
                shutil.copy2(source, target)
                mode = 'copy'
        published.append({
            'name': source.name,
            'source': str(source.relative_to(root)),
            'size_class': size_class,
            'bytes': source.stat().st_size,
            'sha256': digest,
            'publish_mode': mode,
        })

    manifest = {
        'schema_version': 1,
        'selection': selection,
        'count': len(published),
        'files': published,
    }
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=root, delete=False) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
        temporary = Path(handle.name)
    os.replace(temporary, root / 'structure_manifest.json')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--selection', choices=('all', *SIZE_CLASSES), default='all')
    parser.add_argument('--expected-count', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_selected_cifs(
        args.output_dir, selection=args.selection, expected_count=args.expected_count,
    ), ensure_ascii=False))


if __name__ == '__main__':
    main()
