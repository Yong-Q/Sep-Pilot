import json
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'stage_pormake_outputs.py'
SPEC = importlib.util.spec_from_file_location('stage_pormake_outputs_under_test', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
stage_selected_cifs = MODULE.stage_selected_cifs


def _cif(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_stage_all_publishes_both_size_classes_at_dataset_root(tmp_path):
    _cif(tmp_path / 'large' / 'large.cif', 'large')
    _cif(tmp_path / 'small' / 'small.cif', 'small')

    result = stage_selected_cifs(tmp_path, selection='all', expected_count=2)

    assert result['count'] == 2
    assert (tmp_path / 'large.cif').read_text() == 'large'
    assert (tmp_path / 'small.cif').read_text() == 'small'
    manifest = json.loads((tmp_path / 'structure_manifest.json').read_text())
    assert manifest['selection'] == 'all'
    assert {item['size_class'] for item in manifest['files']} == {'large', 'small'}


def test_stage_large_only_rejects_wrong_expected_count_and_never_publishes_small(tmp_path):
    _cif(tmp_path / 'large' / 'large.cif', 'large')
    _cif(tmp_path / 'small' / 'small.cif', 'small')

    with pytest.raises(ValueError, match='expected 2 selected CIFs, found 1'):
        stage_selected_cifs(tmp_path, selection='large', expected_count=2)

    assert not (tmp_path / 'small.cif').exists()
    assert not (tmp_path / 'structure_manifest.json').exists()


def test_stage_is_idempotent_for_the_same_published_dataset(tmp_path):
    _cif(tmp_path / 'large' / 'a.cif', 'same')
    first = stage_selected_cifs(tmp_path, selection='large', expected_count=1)
    second = stage_selected_cifs(tmp_path, selection='large', expected_count=1)

    assert first['files'] == second['files']
    assert (tmp_path / 'a.cif').read_text() == 'same'
