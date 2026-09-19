#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT.parent
for path in (str(PROJECT_ROOT), str(LEGACY_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tools import zeopp  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Zeo++ pore analysis for one CIF or a CIF directory.")
    parser.add_argument("--cif-path", default=None)
    parser.add_argument("--cif-dir", default=None)
    parser.add_argument("--probe-radius", type=float, default=1.525)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if bool(args.cif_path) == bool(args.cif_dir):
        raise SystemExit("Provide exactly one of --cif-path or --cif-dir")

    if args.cif_path:
        result = zeopp.run_single(
            cif_path=args.cif_path,
            probe_radius=args.probe_radius,
            n_samples=args.n_samples,
        )
        payload = asdict(result)
    else:
        results = zeopp.run_batch(
            cif_dir=args.cif_dir,
            output_csv=args.output_csv,
            probe_radius=args.probe_radius,
            n_samples=args.n_samples,
            n_threads=args.n_threads,
        )
        payload = {
            "results": [asdict(item) for item in results],
            "output_csv": args.output_csv,
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
