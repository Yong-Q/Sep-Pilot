from pathlib import Path

from agents.result_validation import ValidationLimits, build_validation_dossier


def generic_node(root: Path) -> dict:
    return {
        "workflow_id": "workflow-1",
        "plan_version": 3,
        "step_id": "calculate",
        "job_ids": ["123"],
        "contract": {
            "step_id": "calculate",
            "tool": "unknown_calculator",
            "arguments": {"output_dir": str(root)},
            "expected_outputs": [str(root)],
        },
        "result": {"work_dir": str(root), "job_id": "123"},
        "path_manifest": {"result_dir": str(root), "expected_outputs": [str(root)]},
    }


def test_dossier_recurses_once_and_samples_deterministically(tmp_path):
    root = tmp_path / "results"
    for index in range(8):
        path = root / f"case-{index}" / "result.dat"
        path.parent.mkdir(parents=True)
        path.write_text(f"loading={index + 1}.0\ncompleted=true\n")

    first = build_validation_dossier(generic_node(root), "attempt-1", tmp_path / "evidence")
    second = build_validation_dossier(generic_node(root), "attempt-1", tmp_path / "evidence")

    assert first["inventory"]["file_count"] == 8
    assert first["samples"] == second["samples"]
    assert first["evidence_call_id"] == second["evidence_call_id"]
    assert Path(first["evidence_path"]).exists()
    assert len(first["samples"]) == 3
    assert first["contract"] == generic_node(root)["contract"]

    changed = generic_node(root)
    changed["contract"]["arguments"]["scientific_condition"] = "different"
    third = build_validation_dossier(changed, "attempt-1", tmp_path / "evidence")
    assert third["evidence_call_id"] != first["evidence_call_id"]


def test_symlink_escape_cannot_pass(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    outside = tmp_path / "outside.dat"
    outside.write_text("completed=true\n")
    (root / "escape.dat").symlink_to(outside)

    dossier = build_validation_dossier(generic_node(root), "escape", tmp_path / "evidence")

    assert dossier["verdict"] == "invalid"
    assert dossier["inventory"]["escaped_paths"]


def test_truncated_scan_cannot_pass(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    (root / "result.dat").write_text("completed=true\n")

    dossier = build_validation_dossier(
        generic_node(root),
        "limited",
        tmp_path / "evidence",
        limits=ValidationLimits(max_entries=0),
    )

    assert dossier["verdict"] != "pass"
    assert dossier["inventory"]["truncated"] is True


def method_node(root: Path, tool: str, *, arguments=None) -> dict:
    node = generic_node(root)
    node["contract"]["tool"] = tool
    node["contract"]["arguments"] = {"output_dir": str(root), **(arguments or {})}
    return node


def test_profiles_recognize_native_content_without_global_result_filename(tmp_path):
    cases = []

    structure = tmp_path / "structure"; structure.mkdir()
    (structure / "software-specific.output").write_text(
        "data_framework\n_cell_length_a 10.0\nloop_\n_atom_site_label\n_atom_site_fract_x\nC1 0.0\n")
    cases.append((method_node(structure, "generate_structure"), "valid_cif"))

    charge = tmp_path / "charge"; charge.mkdir()
    (charge / "charges.native").write_text(
        "data_charged\nloop_\n_atom_site_label\n_atom_site_charge\nC1 -0.125\nH1 0.125\n")
    cases.append((method_node(charge, "run_pacman_charge"), "finite_charge_column"))

    cdft = tmp_path / "cdft"; cdft.mkdir()
    (cdft / "summary.from-program").write_text(
        "MOF,total_density_molec_A3,CO2_henry_mol_L_atm\na,0.005,12.5\nb,0.006,13.5\n")
    cases.append((method_node(cdft, "run_cdft", arguments={"action": "pipeline"}),
                  "finite_adsorption_or_vext"))

    gcmc = tmp_path / "gcmc"; gcmc.mkdir()
    (gcmc / "native-log.anything").write_text(
        "Simulation finished\nAverage loading absolute [mol/kg framework] 2.75 +/- 0.03\n")
    cases.append((method_node(gcmc, "run_gcmc_isotherm"), "finite_loading_and_completion"))

    for index, (node, required_fact) in enumerate(cases):
        dossier = build_validation_dossier(node, f"profile-{index}", tmp_path / "evidence")
        assert dossier["verdict"] == "pass"
        assert any(sample["facts"].get(required_fact) for sample in dossier["samples"])
        if node["contract"]["tool"] == "run_cdft":
            assert len(dossier["successful_artifacts"]) == 1


def test_nonfinite_and_low_success_batch_require_recovery(tmp_path):
    root = tmp_path / "batch"; root.mkdir()
    values = ["1.0", "nan", "failed", "inf"]
    for index, value in enumerate(values):
        text = (f"Simulation finished\nAverage loading absolute [mol/kg framework] {value}\n"
                if value != "failed" else "Simulation failed: convergence error\n")
        (root / f"program-output-{index}").write_text(text)
    node = method_node(root, "run_gcmc_isotherm", arguments={"min_success_ratio": 0.60})

    dossier = build_validation_dossier(node, "bad-batch", tmp_path / "evidence")

    assert dossier["counts"] == {"attempted": 4, "succeeded": 1, "failed": 3}
    assert dossier["verdict"] == "recover"
    assert len(dossier["successful_artifacts"]) == 1
    assert len(dossier["failed_items"]) == 3


def test_scheduler_text_is_reported_without_keyword_verdict(tmp_path):
    root = tmp_path / "logged"; root.mkdir()
    (root / "native-result").write_text("completed=true\nvalue=1.25\n")
    (root / "scheduler.stderr").write_text(
        "warning: harmless setup message\nERROR solver matrix is singular\nTraceback (most recent call last):\n")
    node = method_node(root, "unknown_calculator")

    dossier = build_validation_dossier(node, "logged", tmp_path / "evidence")

    # Text markers are context for the scientific Agent, never a standalone verdict.
    assert dossier["verdict"] == "pass"
    excerpts = {item["path"]: item["text"] for item in dossier["diagnostic_excerpts"]}
    assert "scheduler.stderr" in excerpts
    assert "ERROR" in excerpts["scheduler.stderr"]


def test_md_and_electronic_dft_profiles_require_native_completion_and_finite_energy(tmp_path):
    md = tmp_path / "md"; md.mkdir()
    (md / "lammps-screen.custom").write_text(
        "Step Temp TotEng\n0 298 -1234.5\nLoop time of 4.2 on 8 procs for 1000 steps\n")
    dft = tmp_path / "dft"; dft.mkdir()
    (dft / "electronic-stdout.custom").write_text(
        "reached required accuracy - stopping structural energy minimisation\n"
        "free energy TOTEN = -432.125 eV\nGeneral timing and accounting informations for this job\n")

    md_dossier = build_validation_dossier(method_node(md, "run_md_optimize"), "md", tmp_path / "evidence")
    dft_dossier = build_validation_dossier(method_node(dft, "run_vasp"), "dft", tmp_path / "evidence")

    assert md_dossier["verdict"] == "pass"
    assert md_dossier["method_facts"]["finite_energy_and_completion"] is True
    assert dft_dossier["verdict"] == "pass"
    assert dft_dossier["method_facts"]["finite_total_energy_and_completion"] is True


def test_dft_nonfinite_energy_requires_recovery(tmp_path):
    root = tmp_path / "bad-dft"; root.mkdir()
    (root / "program-output").write_text(
        "reached required accuracy\nfree energy TOTEN = NaN eV\nGeneral timing and accounting informations for this job\n")

    dossier = build_validation_dossier(method_node(root, "run_vasp"), "bad-dft", tmp_path / "evidence")

    assert dossier["verdict"] == "recover"
    assert dossier["counts"]["failed"] == 1
