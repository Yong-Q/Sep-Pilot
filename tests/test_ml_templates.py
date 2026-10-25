"""Scientific contracts for model-family and active-learning templates."""

import pytest

from agents.ml_templates import build_ml_template, plan_active_learning


def _manifest():
    return {
        "split": {"train": ["a", "b", "c"], "valid": ["v"], "test": ["t"]},
        "normalization": {"source": "train_labels_only"},
        "source_csv_sha256": "a" * 64,
        "label_source": "measured adsorption uptake",
    }


def test_unimof_template_is_staged_and_test_is_held_out():
    plan = build_ml_template(_manifest(), scientific_target="adsorption_uptake",
                             architecture="joint_physics")
    assert plan["model_family"] == "unimof_3d"
    assert plan["model_blocks"] == [
        "3d_structure_encoder", "gas_embedding", "environment_embedding",
        "prediction_head", "physics_residual_adapter",
    ]
    assert plan["source_sha256"] == "a" * 64
    assert [stage["id"] for stage in plan["stages"]] == [
        "verify_data", "establish_baseline", "joint_training_gate",
        "train", "select_by_validation", "evaluate_held_out_test",
    ]
    assert plan["stages"][-1]["depends_on"] == ["select_by_validation"]
    assert plan["selection_split"] == "valid"
    assert plan["test_used_for_selection"] is False
    assert plan["execution"] == "template_only"


def test_gas_holdout_manifest_is_supported_without_disclosing_source_path():
    manifest = {
        "split_records": {"train": 50, "valid": 10, "test": 10},
        "split_by": "gas_species",
        "train_gases": ["Ar", "CH4"], "valid_gas": "Kr", "test_gas": "O2",
        "normalization": {"source": "train_labels_only"},
        "source_record_audit_sha256": "b" * 64,
        "label_source": "simulated adsorption uptake",
        "source": "/private/input.csv",
    }
    plan = build_ml_template(manifest, scientific_target="adsorption_uptake")
    assert plan["split_kind"] == "gas_species"
    assert plan["split_sizes"] == {"train": 50, "valid": 10, "test": 10}
    assert "/private" not in str(plan)


@pytest.mark.parametrize("mutation", [
    lambda m: m["split"]["test"].append("a"),
    lambda m: m["normalization"].update(source="all_labels"),
    lambda m: m.pop("source_csv_sha256"),
])
def test_training_template_rejects_leakage_or_missing_provenance(mutation):
    manifest = _manifest()
    mutation(manifest)
    with pytest.raises(ValueError):
        build_ml_template(manifest, scientific_target="adsorption_uptake")


def test_adsorption_labels_cannot_train_a_membrane_permeance_claim():
    with pytest.raises(ValueError, match="permeance"):
        build_ml_template(_manifest(), scientific_target="membrane_permeance")


def test_undocumented_labels_cannot_support_membrane_permeance():
    manifest = _manifest()
    manifest.pop("label_source")
    with pytest.raises(ValueError, match="label_source"):
        build_ml_template(manifest, scientific_target="membrane_permeance")


def test_training_template_requires_label_provenance_for_adsorption_too():
    manifest = _manifest()
    manifest.pop("label_source")
    with pytest.raises(ValueError, match="label_source"):
        build_ml_template(manifest, scientific_target="adsorption_uptake")


def test_active_learning_only_uses_revealed_train_labels_and_unlabelled_train_candidates():
    plan = plan_active_learning(_manifest(), revealed_labels={"a": [1.2, 0.8]},
                                candidate_ids=["b", "c"], acquisition="committee_proxy")
    assert plan["training_ids"] == ["a"]
    assert plan["candidate_ids"] == ["b", "c"]
    assert plan["uncertainty_interpretation"] == "heuristic_proxy_not_calibrated"
    assert "revealed_labels" not in plan
    assert plan["selection_split"] == "train"
    assert plan["requires_new_labels_from"] == "real_calculation_or_measurement"


@pytest.mark.parametrize("labels,candidates", [
    ({"v": [1.0]}, ["b"]),
    ({"a": [1.0]}, ["t"]),
    ({"a": [1.0]}, ["a"]),
    ({"a": [float("nan")]}, ["b"]),
])
def test_active_learning_rejects_validation_test_known_or_invalid_labels(labels, candidates):
    with pytest.raises(ValueError):
        plan_active_learning(_manifest(), revealed_labels=labels,
                             candidate_ids=candidates)


def test_active_learning_requires_explicit_split_ids():
    manifest = _manifest()
    manifest.pop("split")
    manifest["split_records"] = {"train": 3, "valid": 1, "test": 1}
    with pytest.raises(ValueError, match="sample IDs"):
        plan_active_learning(manifest, revealed_labels={"a": [1.0]}, candidate_ids=["b"])
