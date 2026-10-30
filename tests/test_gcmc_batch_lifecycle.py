from pathlib import Path

from agents import registry, slurm
from agents.watch_context import clear_context, set_context


def _charged_cif(path: Path):
    path.write_text("data_x\n_cell_length_a 30\n_cell_length_b 30\n_cell_length_c 30\n"
                    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
                    "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
                    "_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_charge\nC1 C 0 0 0 0.0\n")


def test_batch_executor_passes_scoped_branch_and_registers_jobwatch(tmp_path, monkeypatch):
    root = tmp_path / "runs" / "u" / "c"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _charged_cif(inputs / "a.cif")
    seen = {}

    monkeypatch.setattr("agents.workspace.session_root", lambda *a, **k: root)
    monkeypatch.setattr(registry, "get_config", lambda: type("C", (), {"project_root": tmp_path})())
    monkeypatch.setattr(slurm, "submit_gcmc_batch", lambda **kwargs: seen.setdefault(
        "submission", {"submitted": True, "job_id": "42", "work_dir": kwargs["output_dir"], "kwargs": kwargs}))
    monkeypatch.setattr(registry, "_register_jobwatch", lambda result, tool="": seen.update(watch=(result, tool)))
    set_context(username="u", conv_id="c", agent_name="adsorption")
    try:
        output = root / "gcmc" / "uff"
        result = registry._exec_gcmc_batch({
            "cif_dir": str(inputs), "gases": ["CO2"], "temperature": 298,
            "pressure": 1.0, "output_dir": str(output),
        })
    finally:
        clear_context()
    assert result["job_id"] == "42"
    assert seen["submission"]["kwargs"]["output_dir"] == str(output.resolve())
    assert seen["watch"][1] == "run_gcmc_batch"


def test_batch_executor_rejects_output_outside_current_session(tmp_path, monkeypatch):
    root = tmp_path / "runs" / "u" / "c"
    monkeypatch.setattr("agents.workspace.session_root", lambda *a, **k: root)
    monkeypatch.setattr(registry, "get_config", lambda: type("C", (), {"project_root": tmp_path})())
    result = registry._exec_gcmc_batch({
        "cif_dir": str(tmp_path), "gases": ["CO2"], "temperature": 298,
        "pressure": 1.0, "output_dir": str(tmp_path / "other"),
    })
    assert result["error_code"] == "OUTPUT_OUTSIDE_SESSION"
    assert not result["submitted"]


def test_batch_submitter_never_overwrites_existing_scientific_results(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _charged_cif(inputs / "a.cif")
    output = tmp_path / "branch"
    data = output / "CO2" / "298K" / "a" / "Output" / "System_0" / "result.data"
    data.parent.mkdir(parents=True)
    data.write_text("scientific evidence")
    called = []
    monkeypatch.setattr(slurm, "submit_and_return", lambda **kwargs: called.append(kwargs))
    result = slurm.submit_gcmc_batch(str(inputs), output_dir=str(output))
    assert result["error_code"] == "EXISTING_RESULTS_REQUIRE_REUSE"
    assert result["work_dir"] == str(output.resolve())
    assert called == []


def test_one_batch_job_runs_independent_material_gas_branches_with_bounded_parallelism(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _charged_cif(inputs / "a.cif")
    _charged_cif(inputs / "b.cif")
    captured = {}
    submitted=[];staged=[]
    import shutil
    def stage_exact_cif(directory, forcefield, molecule, definition, cif, **kwargs):
        destination=Path(directory);destination.mkdir(parents=True,exist_ok=True)
        source=Path(cif);raw=source.read_bytes();target=destination/source.name
        shutil.copy2(source,target)
        assert target.read_bytes()==raw
        staged.append((target,raw))
    # This test exercises native geometry and batch concurrency, not a site FF install.
    monkeypatch.setattr(slurm, "stage_raspa_local_files", stage_exact_cif)
    monkeypatch.setattr(slurm, "_raspa_composition", lambda *a, **k: {})
    monkeypatch.setattr(slurm, "stage_raspa_local_shell", lambda **k: "# controlled fixture; no scientific executable")
    def submit_once(**kwargs):
        submitted.append(kwargs);captured.update(kwargs)
        return {"submitted": True, "job_id": "42", "work_dir": kwargs["work_dir"]}
    monkeypatch.setattr(slurm, "submit_and_return", submit_once)

    result = slurm.submit_gcmc_batch(str(inputs), gases=["CO2", "N2"],
                                     output_dir=str(tmp_path / "output"), cpus_per_task=3)

    script = captured["command"]
    assert result["job_id"] == "42"
    assert script.count("launch run_one ") == 4
    assert 'MAX_PARALLEL=${SLURM_CPUS_PER_TASK:-1}' in script
    assert 'wait "${PIDS[0]}"' in script
    assert 'if (( FAILURES > 0 ))' in script
    assert captured["cpus_per_task"] == 3
    assert len(submitted)==1
    assert len(staged)==4 and len({str(path)for path,raw in staged})==4
    assert all(path.read_bytes()==raw for path,raw in staged)
