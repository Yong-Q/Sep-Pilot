"""Operator refresh; resource monitoring itself remains read-only."""
from pathlib import Path
import sys
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.node_inventory import node_inventory, candidates, record_probe_receipt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-receipt', action='append', default=[])
    args = parser.parse_args()
    snapshot = node_inventory(ROOT, refresh=True, persist=True)
    probes = [record_probe_receipt(ROOT, path) for path in args.probe_receipt]
    print(json.dumps({'path': str(ROOT / 'env/node_inventory.json'), 'verified': snapshot['verified'],
        'node_count': len(snapshot['nodes']), 'candidates': [n['name'] for n in candidates(snapshot)], 'published_probes': probes}, ensure_ascii=False))
    raise SystemExit(0 if snapshot['verified'] else 1)
