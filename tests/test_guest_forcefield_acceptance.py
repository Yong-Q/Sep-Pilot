import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_WORKFLOW = PROJECT_ROOT / "scripts" / "run_workflow.py"
FIXTURE_ROOT = Path("/home/user/RASPA/RASPA_tools/high_throughput_adsorption_test/test/MOF_AM1-BCC")
RASPA_SIM = Path("/home/user/RASPA2/simulations/bin/simulate")


class GuestForcefieldAcceptanceTest(unittest.TestCase):
    def run_skill(self, spec: dict, *, execute: bool = True) -> dict:
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

    def exported_guest_charge_sum(self, guest_def_path: Path, pseudo_atoms_path: Path) -> float:
        guest_def_lines = guest_def_path.read_text(encoding="utf-8").splitlines()
        atom_count = int(guest_def_lines[5])
        atom_start = guest_def_lines.index("# atomic positions") + 1
        pseudo_names = [guest_def_lines[atom_start + idx].split()[1] for idx in range(atom_count)]
        charges: dict[str, float] = {}
        for row in pseudo_atoms_path.read_text(encoding="utf-8").splitlines():
            parts = row.split()
            if len(parts) >= 7:
                try:
                    charges[parts[0]] = float(parts[6])
                except ValueError:
                    continue
        return sum(charges[name] for name in pseudo_names)

    def test_flexible_guest_records_neutral_charge_and_runs_200_cycle_gcmc(self) -> None:
        self.assertTrue(RASPA_SIM.exists(), msg=f"Missing RASPA binary: {RASPA_SIM}")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            build_dir = tmpdir_path / "guest_build"
            case_dir = tmpdir_path / "gcmc_case"

            build_spec = {
                "skill": "guest-forcefield",
                "action": "build",
                "params": {
                    "input_path": str(FIXTURE_ROOT / "1-octanol.mol"),
                    "name": "octanol",
                    "from_existing_ligpargen": str(FIXTURE_ROOT / "ligpargen_extract" / "octanol" / "tmp"),
                    "allow_charge_rounding_correction": True,
                    "output_dir": str(build_dir),
                    "critical_temperature": 652.5,
                    "critical_pressure": 2860000.0,
                    "acentric_factor": 0.587,
                },
            }
            build_result = self.run_skill(build_spec)
            self.assertEqual(build_result["skill"], "guest-forcefield")
            manifest_path = build_dir / "package" / "guest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["mode_selected"], "flexible")
            self.assertGreater(manifest["rotatable_heavy_bonds"], 0)
            self.assertEqual(manifest["rotatable_heavy_bonds"], 6)
            self.assertIn("component_scaling_recommendation", manifest)
            self.assertEqual(
                manifest["component_scaling_recommendation"]["Intra14VDWScalingValue"],
                0.5,
            )
            self.assertEqual(
                manifest["component_scaling_recommendation"]["Intra14ChargeChargeScalingValue"],
                0.5,
            )

            guest_def_path = Path(manifest["exports"]["guest_def"])
            pseudo_atoms_path = Path(manifest["exports"]["pseudo_atoms_def"])
            self.assertAlmostEqual(
                self.exported_guest_charge_sum(guest_def_path, pseudo_atoms_path),
                0.0,
                places=9,
            )

            case_spec = {
                "skill": "guest-forcefield",
                "action": "case",
                "params": {
                    "output_dir": str(case_dir),
                    "framework_name": "2530361_pacman",
                    "unit_cells": [2, 1, 1],
                    "components": ["octanol:1.0:flexible"],
                    "cycles": 200,
                    "init_cycles": 100,
                    "print_every": 50,
                    "temperature": 298.0,
                    "pressure": 100000.0,
                    "input_name": "mix.input",
                },
            }
            self.run_skill(case_spec)

            shutil.copy2(guest_def_path, case_dir / guest_def_path.name)
            shutil.copy2(pseudo_atoms_path, case_dir / pseudo_atoms_path.name)
            shutil.copy2(Path(manifest["exports"]["force_field_mixing_rules_def"]), case_dir / "force_field_mixing_rules.def")
            shutil.copy2(Path(manifest["exports"]["force_field_def"]), case_dir / "force_field.def")
            shutil.copy2(FIXTURE_ROOT / "2530361_pacman.cif", case_dir / "2530361_pacman.cif")

            mix_input = (case_dir / "mix.input").read_text(encoding="utf-8")
            self.assertIn("NumberOfCycles                200", mix_input)
            self.assertIn("NumberOfInitializationCycles  100", mix_input)
            self.assertIn("Component 0 MoleculeName              octanol", mix_input)
            self.assertIn("Intra14VDWScalingValue    0.5", mix_input)
            self.assertIn("Intra14ChargeChargeScalingValue 0.5", mix_input)

            run = subprocess.run(
                [str(RASPA_SIM), "-i", "mix.input"],
                cwd=case_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, msg=run.stderr or run.stdout)
            self.assertNotIn("ERROR", run.stderr.upper())

            output_dir = case_dir / "Output" / "System_0"
            output_files = sorted(output_dir.glob("output_*.data"))
            self.assertTrue(output_files, msg=f"No RASPA output files under {output_dir}")
            output_text = output_files[0].read_text(encoding="utf-8")
            self.assertIn("Enthalpy of adsorption", output_text)


if __name__ == "__main__":
    unittest.main()
