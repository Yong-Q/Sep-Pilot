"""Evidence-bound ML planning templates; this module does not run training.

The Uni-MOF architecture and active-learning safeguards are adapted as a small
planning contract. Model checkpoints and numerical improvements are never
inferred from a template.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_SPLITS = ("train", "valid", "test")


def native_dataset_planner_mismatch(manifest, planner):
    """Explain the executable/native versus legacy-template capability boundary.

    Native dataset wrappers are not legacy planning manifests. Never fabricate
    normalization/provenance, infer a trained model, select labels or dispatch a
    replacement tool merely to make the older planner accept such a wrapper.
    """
    native = manifest.get('schema') if isinstance(manifest, Mapping) else None
    if (not isinstance(native, Mapping) or not isinstance(native.get('split'), Mapping)
            or not isinstance(native.get('features'), list) or not native.get('target')
            or not manifest.get('feature_csv') or not manifest.get('label_csv')
            or isinstance(manifest.get('split'), Mapping)):
        return None
    target = {'ml_active_learning_plan': 'ml_active_learning', 'ml_template_plan': 'ml_train'}.get(planner)
    if target is None:
        return None
    return {'error': 'Native ML dataset manifest supplied to a legacy template-only planner. '
                     'Use the executable native capability for training or candidate acquisition; '
                     'a template is not a trained model, selected sample, or output file.',
            'error_kind': 'ml_capability_mismatch', 'manifest_format': 'native_ml_dataset',
            'requested_planner': planner, 'available_tool': target,
            'next_action': {'tool': 'get_tool_schema', 'arguments': {'tool_name': target}},
            'requires_user': False, 'executed': False, 'scientific_execution': False,
            'retry_granted': False,
            'workflow_requirements': ['Preserve the original held-out split and label provenance.',
                'Acquisition depends on an owned verified model produced by native ml_train.',
                'Preserve the user-requested selection count explicitly; do not guess it.',
                'Use only real native result paths and a versioned authorized workflow repair.']}


def _provenance(manifest: Mapping[str, Any]) -> str:
    digest = manifest.get("source_csv_sha256") or manifest.get("source_record_audit_sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("a source data SHA-256 provenance hash is required")
    return digest.lower()


def _split_contract(manifest: Mapping[str, Any]) -> tuple[str, dict[str, int]]:
    split = manifest.get("split")
    if isinstance(split, Mapping):
        ids: dict[str, list[str]] = {}
        for name in _SPLITS:
            values = split.get(name)
            if (not isinstance(values, list) or not values
                    or any(not isinstance(value, str) or not value for value in values)
                    or len(values) != len(set(values))):
                raise ValueError(f"{name} split needs unique nonempty sample IDs")
            ids[name] = values
        if any(set(ids[left]) & set(ids[right]) for left, right in
               (("train", "valid"), ("train", "test"), ("valid", "test"))):
            raise ValueError("train, validation and test sample IDs must be disjoint")
        return "sample_ids", {name: len(ids[name]) for name in _SPLITS}

    records = manifest.get("split_records")
    if isinstance(records, Mapping) and manifest.get("split_by") == "gas_species":
        sizes = {name: records.get(name) for name in _SPLITS}
        if any(type(value) is not int or value <= 0 for value in sizes.values()):
            raise ValueError("gas holdout requires positive split record counts")
        train = manifest.get("train_gases")
        valid, test = manifest.get("valid_gas"), manifest.get("test_gas")
        if (not isinstance(train, list) or not train
                or any(not isinstance(gas, str) or not gas for gas in train)
                or not isinstance(valid, str) or not valid
                or not isinstance(test, str) or not test
                or len(set(train + [valid, test])) != len(train) + 2):
            raise ValueError("train, validation and test gas identities must be disjoint")
        return "gas_species", sizes
    raise ValueError("explicit sample IDs or a documented gas-species holdout is required")


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[str, dict[str, int], str]:
    if not isinstance(manifest, Mapping):
        raise ValueError("dataset manifest must be an object")
    kind, sizes = _split_contract(manifest)
    if (not isinstance(manifest.get("normalization"), Mapping)
            or manifest["normalization"].get("source") != "train_labels_only"):
        raise ValueError("normalization must be fitted on training labels only")
    if not isinstance(manifest.get("label_source"), str) or not manifest["label_source"].strip():
        raise ValueError("label_source must document the measured or simulated target provenance")
    return kind, sizes, _provenance(manifest)


def build_ml_template(manifest: Mapping[str, Any], *, scientific_target: str,
                      architecture: str = "baseline") -> dict[str, Any]:
    """Build a reviewable Uni-MOF training recipe from a documented data split.

    This is a model-family template, not an executable training job. A real
    dataset adapter, checkpoint, compute resources and measured labels are
    required before scheduling it.
    """
    kind, sizes, source_hash = _validate_manifest(manifest)
    if not isinstance(scientific_target, str) or not scientific_target.strip():
        raise ValueError("an explicit scientific target is required")
    if architecture not in {"baseline", "joint_physics"}:
        raise ValueError("architecture must be baseline or joint_physics")
    label_source = str(manifest.get("label_source") or "")
    if "permeance" in scientific_target.lower() and "permeance" not in label_source.lower():
        raise ValueError("membrane permeance requires documented permeance labels")

    blocks = ["3d_structure_encoder", "gas_embedding", "environment_embedding",
              "prediction_head"]
    stages = [
        {"id": "verify_data", "depends_on": [], "checks": [
            "source_hash", "disjoint_fixed_splits", "train_only_normalization",
            "label_provenance"]},
        {"id": "establish_baseline", "depends_on": ["verify_data"], "checks": [
            "baseline_checkpoint_and_validation_metrics"]},
    ]
    train_dep = "establish_baseline"
    if architecture == "joint_physics":
        blocks.append("physics_residual_adapter")
        stages.append({"id": "joint_training_gate", "depends_on": ["establish_baseline"],
                       "checks": ["source_and_adapter_finite_nonzero_gradients",
                                  "source_and_adapter_weights_changed",
                                  "baseline_state_restored_before_training"]})
        train_dep = "joint_training_gate"
    stages.extend([
        {"id": "train", "depends_on": [train_dep], "checks": [
            "reproducible_seed", "checkpoint_provenance", "training_metrics"]},
        {"id": "select_by_validation", "depends_on": ["train"], "checks": [
            "fixed_validation_split", "baseline_comparison", "test_not_seen"]},
        {"id": "evaluate_held_out_test", "depends_on": ["select_by_validation"],
         "checks": ["single_final_test_evaluation", "report_uncertainty_and_limits"]},
    ])
    return {
        "execution": "template_only", "model_family": "unimof_3d",
        "architecture": architecture, "scientific_target": scientific_target.strip(),
        "split_kind": kind, "split_sizes": sizes, "source_sha256": source_hash,
        "model_blocks": blocks, "stages": stages,
        "selection_split": "valid", "test_used_for_selection": False,
        "scientific_limit": "adsorption predictions alone do not establish membrane permeance or selectivity",
    }


def plan_active_learning(manifest: Mapping[str, Any], *,
                         revealed_labels: Mapping[str, Sequence[float]],
                         candidate_ids: Sequence[str],
                         acquisition: str = "coverage") -> dict[str, Any]:
    """Declare a label-blind acquisition round without selecting by hidden labels."""
    kind, _sizes, source_hash = _validate_manifest(manifest)
    if kind != "sample_ids":
        raise ValueError("active learning needs explicit train sample IDs")
    if acquisition not in {"coverage", "committee_proxy"}:
        raise ValueError("acquisition must be coverage or committee_proxy")
    if not isinstance(revealed_labels, Mapping) or not revealed_labels:
        raise ValueError("at least one revealed training label is required")
    if (not isinstance(candidate_ids, (list, tuple)) or not candidate_ids
            or any(not isinstance(item, str) or not item for item in candidate_ids)
            or len(candidate_ids) != len(set(candidate_ids))):
        raise ValueError("candidate IDs must be unique and nonempty")
    train = set(manifest["split"]["train"])
    known = set(revealed_labels)
    candidates = set(candidate_ids)
    if not known <= train or not candidates <= train or known & candidates:
        raise ValueError("revealed labels and unlabelled candidates must be disjoint TRAIN IDs")
    for label in revealed_labels.values():
        if (not isinstance(label, (list, tuple)) or not label
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(value) for value in label)):
            raise ValueError("revealed labels must contain finite measured values")
    return {
        "execution": "template_only", "source_sha256": source_hash,
        "selection_split": "train", "training_ids": sorted(known),
        "candidate_ids": list(candidate_ids), "acquisition": acquisition,
        "requires_new_labels_from": "real_calculation_or_measurement",
        "uncertainty_interpretation": (
            "heuristic_proxy_not_calibrated" if acquisition == "committee_proxy" else "none"),
        "guards": ["fit_only_revealed_train_labels", "never_use_validation_or_test_labels_for_acquisition",
                   "freeze_validation_and_test_splits", "append_new_labels_with_provenance"],
    }
