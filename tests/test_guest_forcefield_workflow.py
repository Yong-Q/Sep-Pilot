import json
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_WORKFLOW = PROJECT_ROOT / "scripts" / "run_workflow.py"
FIXTURE_ROOT = Path("/home/user/RASPA/RASPA_tools/high_throughput_adsorption_test/test/MOF_AM1-BCC")


class GuestForcefieldWorkflowTest(unittest.TestCase):
    def run_spec(self, spec: dict, *, execute: bool = True) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(spec, handle, ensure_ascii=False, indent=2)
            spec_path = Path(handle.name)
        try:
            command = ["python3", str(RUN_WORKFLOW), "--no-log", "--spec", str(spec_path)]
            if execute:
                command.append("--execute")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            return json.loads(result.stdout)
        finally:
            spec_path.unlink(missing_ok=True)

    def test_alias_build_executes_through_formal_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "octanol"
            spec = {
                "skill": "molecule-forcefield-pipeline",
                "action": "build",
                "params": {
                    "input_path": str(FIXTURE_ROOT / "1-octanol.mol"),
                    "name": "octanol",
                    "from_existing_ligpargen": str(FIXTURE_ROOT / "ligpargen_extract" / "octanol" / "tmp"),
                    "allow_charge_rounding_correction": True,
                    "output_dir": str(output_dir),
                    "critical_temperature": 652.5,
                    "critical_pressure": 2860000.0,
                    "acentric_factor": 0.587,
                },
            }
            payload = self.run_spec(spec)
            self.assertEqual(payload["skill"], "guest-forcefield")
            self.assertEqual(payload["status"], "completed")
            self.assertTrue((output_dir / "package" / "guest.json").exists())

    def test_case_action_writes_mix_and_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "case"
            framework_cif = Path(tmpdir) / "2530361.cif"
            framework_cif.write_text(
                "data_demo\nloop_\n_atom_site_type_symbol\n_atom_site_charge\nC 0.1\nC -0.1\n",
                encoding="utf-8",
            )
            spec = {
                "skill": "guest-forcefield",
                "action": "case",
                "params": {
                    "output_dir": str(output_dir),
                    "framework_name": "2530361_pacman",
                    "unit_cells": [2, 1, 1],
                    "components": ["octanol:0.5:flexible", "o_phenylenediamine:0.5:rigid"],
                    "cycles": 1000,
                    "init_cycles": 500,
                    "framework_cif": str(framework_cif),
                    "expected_net_charge": 0.0,
                    "prepare_script": "./prepare_raspa_case.py",
                },
            }
            payload = self.run_spec(spec)
            self.assertEqual(payload["status"], "completed")
            self.assertTrue((output_dir / "mix.input").exists())
            self.assertTrue((output_dir / "raspa.pbs").exists())
            self.assertTrue((output_dir / "run_charge_and_raspa.sh").exists())

    def test_charge_validation_requires_declared_model_and_preserves_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "demo.cif"
            cif_path.write_text(
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
            spec = {
                "skill": "guest-forcefield",
                "action": "normalize-cif",
                "params": {
                    "cif_path": str(cif_path),
                    "decimals": 6,
                },
            }
            original = cif_path.read_bytes()
            payload = self.run_spec(spec)
            self.assertNotEqual(payload["status"], "completed")
            self.assertEqual(cif_path.read_bytes(), original)
            values = [
                float(line.split()[-1])
                for line in cif_path.read_text(encoding="utf-8").splitlines()
                if line and line[0].isalpha() and len(line.split()) == 2
            ]
            self.assertAlmostEqual(sum(values), -0.01, places=9)


if __name__ == "__main__":
    unittest.main()
