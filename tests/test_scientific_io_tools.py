import csv
import hashlib
import inspect
import importlib.util
import math
import socket
from pathlib import Path
from types import SimpleNamespace
import pytest

from agents import registry
from agents.defns import ADSORPTION, ANALYST, COMMUNICATOR
from agents.registry import (_exec_analyze_diffusion_msd, _exec_build_mof_database,
                             _exec_download_scientific_file, _exec_features, _exec_ml_active_learning,
                             _exec_ml_predict, _exec_ml_train, get_registry)
from agents.watch_context import clear_context, set_context


def _scope(tmp_path, monkeypatch, username="u", conv_id="c"):
    config = SimpleNamespace(project_root=tmp_path)
    monkeypatch.setattr("agents.registry.get_config", lambda: config)
    monkeypatch.setattr("agents.workspace.get_config", lambda: config)
    set_context(username, conv_id, "analyst")
    return tmp_path / "runs" / username / conv_id


def test_download_scientific_file_is_scoped_hashed_and_atomic(tmp_path, monkeypatch):
    root = _scope(tmp_path, monkeypatch)
    payload = b"data_demo\n_cell_length_a 10.0\n"

    from tests.native_protocol_cases import public_https_transport
    calls=public_https_transport(monkeypatch,payload)
    target = root / "downloads" / "demo.cif"
    result = _exec_download_scientific_file({
        "url": "https://example.org/demo.cif",
        "output_path": str(target),
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
    })
    clear_context()

    assert result["status"] == "success"
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert target.read_bytes() == payload
    assert not target.with_name("demo.cif.part").exists()
    assert calls[0][0:3]==('pool','93.184.216.34',443)
    assert calls[1][0:3]==('request','GET','/demo.cif')
    assert result['transport_hops'][0]['connect_target_public_ip']=='93.184.216.34'
    assert result['transport_hops'][0]['origin_tls_verified']is True


def test_download_rejects_private_network_before_request(tmp_path, monkeypatch):
    root = _scope(tmp_path, monkeypatch)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))])
    called = []
    monkeypatch.setattr("requests.get", lambda *a, **k: called.append(True))
    result = _exec_download_scientific_file({
        "url": "https://internal.invalid/file.cif",
        "output_path": str(root / "downloads" / "file.cif"),
    })
    clear_context()
    assert "forbidden network address" in result["error"]
    assert not called


def test_ml_train_fails_closed_without_real_labelled_csv(tmp_path, monkeypatch):
    from agents.ml_tool_adapter import _legacy_dataset
    live_source = inspect.getsource(get_registry().get("ml_train").execute) + inspect.getsource(_legacy_dataset)
    assert "labels are not generated" in live_source
    assert "np.random.uniform" not in live_source
    root = _scope(tmp_path, monkeypatch)
    submitted=[]
    def forbidden_submit(self,*args,**kwargs):
        submitted.append((args,kwargs))
        pytest.fail('missing native label inputs must not submit a training job')
    monkeypatch.setattr('agents.ml_runtime.MLRuntime.submit',forbidden_submit)
    try:
        result = _exec_ml_train({
            "data_csv": str(root / "missing.csv"),
            "target": "uptake",
            "output_dir": str(root / "ml" / "model"),
        })
        assert result['error_kind']=='native_ml_contract' and result['executed'] is False
        assert result['labels_generated'] is False and result['scientific_acceptance'] is False
        assert result['error'] and submitted==[]
        assert list(root.rglob('labels.csv'))==[]
        assert not (root / "ml" / "model" / "model.joblib").exists()
        source=root/'missing-target.csv'
        source.write_text('feature\n'+''.join(str(index)+'\n' for index in range(8)))
        before=source.read_bytes()
        result = _exec_ml_train({"data_csv":str(source),"target":"uptake",
            "output_dir":str(root/"ml"/"model")})
        assert result['error_kind']=='native_ml_contract' and result['executed'] is False
        assert result['labels_generated'] is False and result['scientific_acceptance'] is False
        assert "labels are not generated" in result["error"]
        assert submitted==[] and list(root.rglob('labels.csv'))==[]
        assert source.read_bytes()==before
        assert not (root / "ml" / "model" / "model.joblib").exists()
    finally:
        clear_context()


@pytest.mark.skipif(importlib.util.find_spec('tools.mof_features') is None,
                    reason='Optional scientific CIF feature extractor is not installed')
def test_batch_feature_extraction_is_structured_atomic_and_session_scoped(tmp_path, monkeypatch):
    root = _scope(tmp_path, monkeypatch)
    source = tmp_path / "cifs"
    source.mkdir()
    for index in range(2):
        (source / f"m{index}.cif").write_text(f"""data_m{index}
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0 0 0
O1 O 0.1 0.1 0.1
""")
    output = root / "ml" / "features.csv"
    result = _exec_features({"cif_dir": str(source), "output_csv": str(output), "n_threads": 2})
    clear_context()
    assert result["status"] == "success" and result["n_features"] == 2
    assert output.is_file() and not output.with_name("features.csv.part").exists()
    assert all(row["volume_A3"] == 1320 for row in result["features"])
    assert get_registry().validate_params("extract_features", {"cif_path": "a", "cif_dir": "b"})


def test_legacy_ga_with_estimated_features_is_not_available_to_analyst():
    assert "run_ga_optimization" not in {getattr(tool, "__name__", "") for tool in ANALYST.functions}
    result = get_registry().get("run_ga_optimization").execute({
        "target_selectivity": 20, "target_uptake": 2,
    })
    assert result["blocked"] and result["executed"] is False
    assert "estimated/random" in result["error"]


def test_framework_md_cannot_be_misrepresented_as_guest_diffusion():
    registry = get_registry()
    schema = registry.get("run_md_optimize").description
    assert "does NOT insert guest molecules" in schema
    issues = registry.validate_params("run_md_optimize", {
        "cif_path": "framework.cif", "mode": "md", "temperature": 298, "gas": "CH4",
    })
    assert any("unknown" in issue and "gas" in issue for issue in issues)


@pytest.mark.skipif(importlib.util.find_spec('tools.mof_features') is None,
                    reason='Optional scientific CIF feature extractor is not installed')
def test_ml_real_training_prediction_and_rf_uncertainty_are_reproducible(tmp_path, monkeypatch):
    root = _scope(tmp_path, monkeypatch)
    data = tmp_path / "real_labels.csv"
    with data.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["volume_A3", "metallic_pct", "total_unsaturation", "en_ratio", "uptake"])
        for index in range(12):
            writer.writerow([1000 + index * 50, .1 + index * .01, 20 + index, .2 + index * .005,
                             1.0 + index * .15])
    model_dir = root / "ml" / "rf_model"
    trained = _exec_ml_train({
        "data_csv": str(data), "target": "uptake", "model_type": "RF",
        "features": ["volume_A3", "metallic_pct", "total_unsaturation", "en_ratio"],
        "output_dir": str(model_dir),
    })
    assert trained["status"] == "success"
    assert trained["synthetic_data_used"] is False
    assert trained["n_samples"] == 12 and trained["random_seed"] == 42
    assert all((model_dir / name).is_file() for name in
               ("model.joblib", "model_metadata.json", "test_predictions.csv"))

    cif_dir = root / "candidates"
    cif_dir.mkdir(parents=True)
    cif = cif_dir / "candidate.cif"
    cif.write_text("""data_candidate
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0 0 0
O1 O 0.1 0.1 0.1
Zn1 Zn 0.2 0.2 0.2
""")
    predicted = _exec_ml_predict({
        "model_dir": str(model_dir), "cif_dir": str(cif_dir),
        "output_csv": str(root / "ml" / "candidate_predictions.csv"),
    })
    assert predicted["status"] == "success" and predicted["n_predicted"] == 1
    assert Path(predicted["output_csv"]).is_file()
    selected = _exec_ml_active_learning({
        "model_dir": str(model_dir), "cif_dir": str(cif_dir),
        "n_select": 1, "strategy": "uncertainty",
    })
    clear_context()
    assert selected["status"] == "success"
    assert selected["selected"][0]["cif"] == "candidate.cif"
    assert selected["selected"][0]["uncertainty"] >= 0
    assert Path(selected["output_path"]).is_file()


def test_analyze_diffusion_msd_csv_units_fit_and_session_output(tmp_path, monkeypatch):
    root = _scope(tmp_path, monkeypatch)
    trajectory = root / "md" / "trajectory.csv"
    trajectory.parent.mkdir(parents=True)
    with trajectory.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_ps", "particle_id", "x_A", "y_A", "z_A"])
        for t in range(12):
            # r^2=t A^2 gives a known 3D Einstein slope of 1 A^2/ps.
            writer.writerow([t, 1, math.sqrt(t), 0, 0])
    output = root / "analysis" / "msd.csv"
    result = _exec_analyze_diffusion_msd({
        "trajectory_path": str(trajectory), "format": "csv", "timestep_ps": 1,
        "fit_start_fraction": 0.1, "fit_end_fraction": 0.9,
        "output_csv": str(output),
    })
    clear_context()

    assert result["status"] == "success"
    assert result["fit_r2"] > 0.999999
    assert abs(result["diffusion_coefficient_m2_s"] - (1e-8 / 6)) < 1e-16
    assert result["n_frames"] == 12 and result["n_particles"] == 1
    assert output.is_file()


def test_new_tools_are_registered_and_exposed_to_scientific_agents():
    names = set(get_registry().all_names())
    assert {"download_scientific_file", "analyze_diffusion_msd"} <= names
    for agent in (ADSORPTION, COMMUNICATOR):
        exposed = {fn.__name__ for fn in agent.functions}
        assert {"download_scientific_file", "analyze_diffusion_msd"} <= exposed


def test_build_mof_database_reads_real_cifs_and_reports_duplicates(tmp_path, monkeypatch):
    root = _scope(tmp_path, monkeypatch)
    source = tmp_path / "data" / "cifs"
    source.mkdir(parents=True)
    cif = """data_demo
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_charge
C1 C 0 0 0 -0.2
Zn1 Zn 0.5 0.5 0.5 0.2
"""
    (source / "one.cif").write_text(cif)
    (source / "duplicate.cif").write_text(cif)
    output = root / "database"
    result = _exec_build_mof_database({"cif_dir": str(source), "output_dir": str(output)})
    clear_context()

    assert result["status"] == "success" and result["n_cifs"] == 2
    assert len(result["duplicate_groups"]) == 1
    database = __import__("json").loads((output / "mof_database.json").read_text())
    record = database["records"][0]
    assert record["atom_count"] == 2
    assert record["composition"] == {"C": 1, "Zn": 1}
    assert abs(record["cell_volume_A3"] - 1320.0) < 1e-9
    assert abs(record["net_charge_e"]) < 1e-12
    assert (output / "mof_database.csv").is_file()
