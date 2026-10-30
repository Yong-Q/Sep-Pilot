"""MD/String relative artifacts follow only an owned absolute native receipt."""
from pathlib import Path

import pytest

from agents.parallel_workflow import artifact_base


@pytest.mark.parametrize('tool', ['run_md_optimize', 'run_string_tst'])
@pytest.mark.parametrize('field', ['output_dir', 'work_dir'])
@pytest.mark.parametrize('child', ['', 'attempt-1'])
def test_owned_native_artifact_root_can_equal_or_descend_from_approved_root(tmp_path, tool, field, child):
    approved = tmp_path / 'approved'
    actual = approved / child
    node = {'tool': tool, 'arguments': {'output_dir': str(approved)}}
    assert artifact_base(node, tmp_path, tmp_path / 'session', {field: str(actual)}) == actual.resolve()


@pytest.mark.parametrize('tool', ['run_md_optimize', 'run_string_tst'])
@pytest.mark.parametrize('escape', ['outside', 'traversal', 'symlink', 'relative', 'conflicting_fields'])
def test_native_receipt_cannot_rebase_outside_approved_root(tmp_path, tool, escape):
    approved, outside = tmp_path / 'approved', tmp_path / 'outside'
    approved.mkdir(); outside.mkdir()
    node = {'tool': tool, 'arguments': {'output_dir': str(approved)}}
    if escape == 'outside':receipt = {'output_dir': str(outside)}
    elif escape == 'traversal':receipt = {'output_dir': str(approved / '..' / 'outside')}
    elif escape == 'symlink':
        (approved / 'link').symlink_to(outside, target_is_directory=True)
        receipt = {'output_dir': str(approved / 'link')}
    elif escape == 'relative':receipt = {'output_dir': 'approved/attempt-1'}
    else:receipt = {'output_dir': str(approved), 'work_dir': str(outside)}
    with pytest.raises(ValueError):artifact_base(node, tmp_path, tmp_path / 'session', receipt)


@pytest.mark.parametrize('tool', ['run_md_optimize', 'run_string_tst'])
def test_native_root_without_an_approved_output_root_is_not_authority(tmp_path, tool):
    with pytest.raises(ValueError):
        artifact_base({'tool': tool, 'arguments': {}}, tmp_path, tmp_path / 'session',
                      {'output_dir': str(tmp_path / 'unapproved')})


def test_project_relative_approved_root_remains_project_anchored(tmp_path):
    approved = tmp_path / 'runs/approved'
    node = {'tool': 'run_string_tst', 'arguments': {'output_dir': 'runs/approved'}}
    assert artifact_base(node, tmp_path, tmp_path / 'session',
                         {'work_dir': str(approved / 'attempt-1')}) == approved / 'attempt-1'
    assert artifact_base(node, tmp_path, tmp_path / 'session') == approved


def test_other_tools_do_not_gain_the_md_string_rebase_exception(tmp_path):
    approved, actual = tmp_path / 'approved', tmp_path / 'approved/attempt-1'
    node = {'tool': 'run_gcmc_batch', 'arguments': {'output_dir': str(approved)}}
    assert artifact_base(node, tmp_path, tmp_path / 'session', {'output_dir': str(actual)}) == approved
