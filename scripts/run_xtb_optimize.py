#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ase.io import read, write


SUPPORTED_INPUT_SUFFIXES = {".xyz", ".mol", ".sdf", ".pdb"}


def _detect_xtbopt(work_dir: Path, namespace: str) -> Path | None:
    candidates = [
        work_dir / f"{namespace}.xtbopt.xyz",
        work_dir / "xtbopt.xyz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(work_dir.glob("*xtbopt.xyz"))
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run xTB geometry optimization from a structure file.")
    parser.add_argument("--input", required=True, help="Input structure path")
    parser.add_argument("--output-dir", required=True, help="Working/output directory")
    parser.add_argument("--xtb-executable", required=True, help="Path to xtb executable")
    parser.add_argument("--gfn", type=int, default=2, help="GFN-xTB level")
    parser.add_argument("--opt-level", default="normal", help="Optimization level")
    parser.add_argument("--charge", type=int, default=0, help="System charge")
    parser.add_argument("--uhf", type=int, default=0, help="Number of unpaired electrons")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel processes")
    parser.add_argument("--namespace", default=None, help="xTB namespace")
    parser.add_argument("--output-format", choices=["xyz", "cif"], default="xyz", help="Final exported structure format")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    xtb_executable = Path(args.xtb_executable).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    if not xtb_executable.exists():
        raise SystemExit(f"xTB executable not found: {xtb_executable}")

    output_dir.mkdir(parents=True, exist_ok=True)
    namespace = args.namespace or input_path.stem
    input_xyz = output_dir / "input.xyz"

    if input_path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES:
        shutil.copy2(input_path, input_xyz)
    else:
        atoms = read(str(input_path))
        write(str(input_xyz), atoms)

    command = [
        str(xtb_executable),
        str(input_xyz),
        "--opt",
        str(args.opt_level),
        "--gfn",
        str(args.gfn),
        "--chrg",
        str(args.charge),
        "--uhf",
        str(args.uhf),
        "--parallel",
        str(args.parallel),
        "--namespace",
        namespace,
    ]
    proc = subprocess.run(
        command,
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    xtbopt = _detect_xtbopt(output_dir, namespace)
    final_output = None
    if proc.returncode == 0 and xtbopt is not None:
        if args.output_format == "cif":
            atoms = read(str(xtbopt))
            final_output = output_dir / f"{namespace}_xtbopt.cif"
            write(str(final_output), atoms)
        else:
            final_output = output_dir / f"{namespace}_xtbopt.xyz"
            shutil.copy2(xtbopt, final_output)

    payload = {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "input_xyz": str(input_xyz),
        "xtbopt_xyz": str(xtbopt) if xtbopt else None,
        "final_output": str(final_output) if final_output else None,
        "output_format": args.output_format,
        "namespace": namespace,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
