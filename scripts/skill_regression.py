from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_WORKFLOW = PROJECT_ROOT / "scripts" / "run_workflow.py"
CONFIG_PATH = PROJECT_ROOT / "config" / "runtime.json"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _case_specs() -> list[dict]:
    cfg = _load_config()
    paths = cfg["paths"]
    cif_dir = _resolve_path(paths["cif_dir"])
    pacman_dir = cif_dir / "pacman_04201447"
    guest_fixture_root = Path("/home/user/RASPA/RASPA_tools/high_throughput_adsorption_test/test/MOF_AM1-BCC")
    tmp_dir = _resolve_path(paths["tmp_dir"]) / "skill_regression"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    staged_charged_dir = tmp_dir / "charged_inputs"
    staged_charged_dir.mkdir(parents=True, exist_ok=True)
    for name in ("MOF-5_pacman.cif", "Ni-MOF-74_pacman.cif"):
        src = pacman_dir / name
        if src.exists():
            shutil.copy2(src, staged_charged_dir / name)
    normalize_cif = tmp_dir / "guest_forcefield_normalize.cif"
    normalize_cif.write_text(
        "\n".join(
            [
                "data_demo",
                "loop_",
                "_atom_site_type_symbol",
                "_atom_site_charge",
                "C 0.100000",
                "O -0.250000",
                "H 0.140000",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return [
        {
            "label": "harness metadata",
            "execute": False,
            "spec": {"skill": "harness"},
        },
        {
            "label": "overview metadata",
            "execute": False,
            "spec": {"skill": "overview"},
        },
        {
            "label": "cif lookup",
            "execute": False,
            "spec": {"skill": "cif", "params": {"keyword": "MOF-5"}},
        },
        {
            "label": "paper download",
            "execute": False,
            "spec": {
                "skill": "paper",
                "params": {
                    "identifier": "2202.09886",
                    "save_dir": str(tmp_dir / "papers"),
                },
            },
        },
        {
            "label": "literature rag",
            "execute": False,
            "spec": {
                "skill": "literature-rag",
                "params": {
                    "query": "What methods are used for CH4 H2 separation screening in porous materials?",
                    "query_type": "methods",
                    "n_results": 3,
                },
            },
        },
        {
            "label": "adsorption quick",
            "execute": False,
            "spec": {
                "skill": "adsorption",
                "action": "isotherm",
                "params": {
                    "cif": str(pacman_dir / "MOF-5_pacman.cif"),
                    "gas": "CH4",
                    "T": 298,
                    "p_min": 0.1,
                    "p_max": 0.2,
                    "n_points": 2,
                    "n_cycles": 100,
                },
            },
        },
        {
            "label": "jobs check",
            "execute": False,
            "spec": {"skill": "jobs", "action": "check", "params": {"job_id": "12345.node01"}},
        },
        {
            "label": "results inspect",
            "execute": False,
            "spec": {
                "skill": "results",
                "action": "inspect",
                "params": {
                    "path": str(_resolve_path(paths["gcmc_output_dir"]) / "raspa" / "gcmc_CO_298K_1e+05Pa" / "results.csv"),
                },
            },
        },
        {
            "label": "analysis existing result",
            "execute": False,
            "spec": {
                "skill": "analysis",
                "params": {
                    "csv_path": str(_resolve_path(paths["gcmc_output_dir"]) / "raspa" / "gcmc_CO_298K_1e+05Pa" / "results.csv"),
                    "run_zeopp": False,
                    "run_interaction": False,
                },
            },
        },
        {
            "label": "charge execute",
            "execute": True,
            "spec": {
                "skill": "charge",
                "params": {
                    "cif_dir": str(pacman_dir),
                    "output_dir": str(tmp_dir / "pacman_out"),
                },
            },
        },
        {
            "label": "cdft inputs",
            "execute": False,
            "spec": {
                "skill": "cdft",
                "action": "inputs",
                "params": {
                    "initial_cif_dir": str(staged_charged_dir),
                    "gases": ["CO2"],
                    "temperature": 298,
                    "bulk_densities": [1.25e-5],
                    "already_charged": True,
                    "pacman_output_dir": str(tmp_dir / "cdft_pacman"),
                    "cdft_output_dir": str(tmp_dir / "cdft_inputs"),
                },
            },
        },
        {
            "label": "pore analysis",
            "execute": False,
            "spec": {
                "skill": "pore-analysis",
                "params": {
                    "cif_dir": str(staged_charged_dir),
                    "probe_radius": 1.525,
                    "n_samples": 5000,
                },
            },
        },
        {
            "label": "material props",
            "execute": False,
            "spec": {
                "skill": "material-props",
                "params": {
                    "cif_dir": str(staged_charged_dir),
                    "output_csv": str(tmp_dir / "mof_features.csv"),
                    "n_threads": 2,
                },
            },
        },
        {
            "label": "interaction energy",
            "execute": False,
            "spec": {
                "skill": "interaction",
                "params": {
                    "cif_path": str(pacman_dir / "MOF-5_pacman.cif"),
                    "gas": "CO2",
                    "method": "raspa_energy",
                    "raspa_dir": "/home/user/RASPA2/simulations/",
                },
            },
        },
        {
            "label": "md optimize quick",
            "execute": False,
            "spec": {
                "skill": "md-optimize",
                "params": {
                    "mode": "single",
                    "cif_path": str(pacman_dir / "MOF-5_pacman.cif"),
                    "output_dir": str(tmp_dir / "md_opt"),
                    "force_field": "UFF",
                    "n_iter": 1,
                    "maxiter": 20,
                    "maxeval": 100,
                    "skip_convert": True,
                },
            },
        },
        {
            "label": "string tst assets",
            "execute": True,
            "spec": {
                "skill": "string-tst",
                "params": {
                    "cif_dir": str(staged_charged_dir),
                    "gases": ["C2H4", "C2H6"],
                    "temperature": 298,
                    "output_dir": str(tmp_dir / "string_tst"),
                },
            },
        },
        {
            "label": "guest forcefield build",
            "execute": True,
            "spec": {
                "skill": "guest-forcefield",
                "action": "build",
                "params": {
                    "input_path": str(guest_fixture_root / "1-octanol.mol"),
                    "name": "octanol",
                    "from_existing_ligpargen": str(guest_fixture_root / "ligpargen_extract" / "octanol" / "tmp"),
                    "output_dir": str(tmp_dir / "guest_forcefield_build"),
                    "critical_temperature": 652.5,
                    "critical_pressure": 2860000.0,
                    "acentric_factor": 0.587,
                },
            },
        },
        {
            "label": "guest forcefield case",
            "execute": True,
            "spec": {
                "skill": "guest-forcefield",
                "action": "case",
                "params": {
                    "output_dir": str(tmp_dir / "guest_forcefield_case"),
                    "framework_name": "MOF-5_pacman",
                    "unit_cells": [2, 2, 2],
                    "components": ["octanol:0.5:flexible", "o_phenylenediamine:0.5:rigid"],
                    "framework_cif": str(staged_charged_dir / "MOF-5_pacman.cif"),
                    "prepare_script": "./prepare_raspa_case.py",
                },
            },
        },
        {
            "label": "guest forcefield normalize cif",
            "execute": True,
            "spec": {
                "skill": "guest-forcefield",
                "action": "normalize-cif",
                "params": {
                    "cif_path": str(normalize_cif),
                    "decimals": 6,
                },
            },
        },
        {
            "label": "structure generation plan",
            "execute": False,
            "spec": {
                "skill": "structure-gen",
                "params": {
                    "material_type": "cof",
                    "n_structures": 2,
                    "max_atoms": 200,
                    "output_dir": str(tmp_dir / "structure_gen"),
                },
            },
        },
        {
            "label": "external potential assets",
            "execute": True,
            "spec": {
                "skill": "external-potential",
                "params": {
                    "cif_dir": str(staged_charged_dir),
                    "gas": "CO2",
                    "temperature": 298,
                    "output_dir": str(tmp_dir / "external_potential"),
                    "max_parallel": 2,
                },
            },
        },
        {
            "label": "vasp setup",
            "execute": False,
            "spec": {
                "skill": "vasp-dft",
                "action": "setup",
                "params": {
                    "cif_path": str(pacman_dir / "MOF-5_pacman.cif"),
                    "calc_type": "static",
                    "work_dir": str(tmp_dir / "vasp_setup"),
                },
            },
        },
    ]


def _run_case(case: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(case["spec"], handle, ensure_ascii=False, indent=2)
        spec_path = Path(handle.name)

    command = [
        "python3",
        str(RUN_WORKFLOW),
        "--project",
        "skill-regression",
        "--label",
        case["label"].replace(" ", "-"),
        "--no-log",
        "--spec",
        str(spec_path),
    ]
    if case["execute"]:
        command.append("--execute")

    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    payload = {
        "label": case["label"],
        "execute": case["execute"],
        "returncode": proc.returncode,
    }
    parsed = None
    if proc.stdout.strip():
        parsed = _extract_json(proc.stdout)
        if parsed is not None:
            payload["stdout_json"] = parsed
        else:
            payload["stdout"] = proc.stdout
    if proc.stderr.strip():
        payload["stderr"] = proc.stderr
    payload["status"] = _classify_status(proc.returncode, parsed)
    spec_path.unlink(missing_ok=True)
    return payload


def _extract_json(stdout: str) -> dict | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.rfind("\n{")
    if start >= 0:
        try:
            return json.loads(text[start + 1 :])
        except Exception:
            return None
    return None


def _classify_status(returncode: int, parsed: dict | None) -> str:
    if returncode != 0:
        return "failed"
    if parsed is None:
        return "passed"

    task_status = parsed.get("status")
    if task_status == "failed":
        return "failed"

    payload = parsed.get("payload")
    if isinstance(payload, dict):
        if payload.get("success") is False:
            return "partial"
        if payload.get("converged") is False and payload.get("error"):
            return "partial"
        if payload.get("returncode", 0) not in (0, None):
            return "failed"
    return "passed"


def main() -> int:
    results = [_run_case(case) for case in _case_specs()]
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] == "passed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
