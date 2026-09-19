#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT.parent
for path in (str(PROJECT_ROOT), str(LEGACY_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tools import pormake_generate  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PORMAKE structure generation through the BiMemAgent wrapper.")
    parser.add_argument("--material-type", required=True, choices=["MOF", "COF", "HOF"])
    parser.add_argument("--n-structures", type=int, default=1)
    parser.add_argument("--max-atoms", type=int, default=1500)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutoff", type=float, default=45.0)
    parser.add_argument("--max-cell", type=float, default=None)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--bb-dir", default=None)
    args = parser.parse_args()

    payload = SimpleNamespace(
        type=args.material_type,
        n=args.n_structures,
        max_atoms=args.max_atoms,
        output_dir=args.output_dir,
        cutoff=args.cutoff,
        max_cell=args.max_cell,
        small=args.small,
        bb_dir=args.bb_dir,
    )
    pormake_generate.cmd_generate(payload)

    output_dir = Path(args.output_dir)
    result = {
        "material_type": args.material_type,
        "output_dir": str(output_dir),
        "candidates_file": str(output_dir / "candidates.txt"),
        "small_dir": str(output_dir / "small"),
        "large_dir": str(output_dir / "large"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
