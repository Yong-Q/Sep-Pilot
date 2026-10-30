"""Preserve mutable submission logs before a verified retry uses the same workdir."""
import hashlib
import json
from pathlib import Path
import shutil


def preserve_attempt_logs(entry, session_root):
    root = Path(session_root).resolve()
    attempt = str(entry.get('attempt_id', ''))
    if not attempt or any(c not in '0123456789abcdef' for c in attempt):
        raise ValueError('failed attempt must have a native attempt identifier')
    sources = set()
    receipt = entry.get('last_result') or {}
    params = entry.get('params') or {}
    for value in (receipt.get('work_dir'), receipt.get('output_dir'), params.get('work_dir'), params.get('output_dir')):
        if not value:
            continue
        folder = Path(value).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            continue
        for file in folder.iterdir():
            if file.name == 'submit.sh' or file.suffix in {'.log', '.out', '.err'}:
                if file.is_file() and not file.is_symlink():
                    sources.add(file)
    destination = root / 'attempt_evidence' / attempt
    records = []
    for source in sorted(sources):
        relative = source.relative_to(root)
        target = destination / relative
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists():
            if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError('existing failed-attempt evidence differs; refusing to overwrite')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        records.append({'source': str(source), 'copy': str(target), 'sha256': digest})
    if records:
        manifest = destination / 'manifest.json'
        content = json.dumps({'attempt_id': attempt, 'files': records}, indent=2)
        if manifest.exists() and manifest.read_text() != content:
            raise ValueError('existing attempt manifest differs')
        if not manifest.exists():
            manifest.write_text(content)
    return {'attempt_id': attempt, 'files': records, 'evidence_dir': str(destination)}
