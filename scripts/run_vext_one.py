from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run_one(input_dat: Path, output_dir: Path, vext_executable: Path) -> dict:
    if not input_dat.exists():
        raise FileNotFoundError(f"Input .dat not found: {input_dat}")
    if not vext_executable.exists():
        raise FileNotFoundError(f"Vext executable not found: {vext_executable}")
    if not vext_executable.is_file():
        raise ValueError(f"Vext executable is not a file: {vext_executable}")

    output_dir.mkdir(parents=True, exist_ok=True)
    local_input = output_dir / "input.dat"
    shutil.copy2(input_dat, local_input)

    proc = subprocess.run(
        [str(vext_executable)],
        cwd=output_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout_path = output_dir / "vext.stdout"
    stderr_path = output_dir / "vext.stderr"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    vext_path = output_dir / "Vext.dat"
    status = "ok" if proc.returncode == 0 and vext_path.exists() else "failed"
    error = ""
    if proc.returncode != 0:
        error = f"Executable exited with code {proc.returncode}"
    elif not vext_path.exists():
        error = "Executable finished without creating Vext.dat"

    payload = {
        "status": status,
        "returncode": proc.returncode,
        "input_dat": str(input_dat.resolve()),
        "copied_input": str(local_input.resolve()),
        "vext_path": str(vext_path.resolve()) if vext_path.exists() else "",
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "error": error,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one external-potential executable against an input.dat file.")
    parser.add_argument("--input-dat", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vext-executable", required=True)
    args = parser.parse_args()

    payload = run_one(
        input_dat=Path(args.input_dat).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        vext_executable=Path(args.vext_executable).resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
