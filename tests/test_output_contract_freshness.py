"""Static input discovery is read-only; scientific writes still need fresh facts."""
from agents.output_contract import artifact_fact, artifact_is_valid
import pytest


def test_readonly_existing_cif_needs_no_rewrite_or_timestamp_change(tmp_path):
    source = tmp_path / 'existing.cif'
    source.write_text('data_existing\n')
    before = artifact_fact(str(source), tmp_path)
    started_at = before['mtime_ns'] / 1e9 + 10
    after = artifact_fact(str(source), tmp_path)
    assert before == after
    assert artifact_is_valid(after, before, started_at=started_at, read_only=True)
    assert source.stat().st_mtime_ns == before['mtime_ns']
    assert not artifact_is_valid(after, before, started_at=started_at)


def test_readonly_nonempty_directory_can_reuse_existing_static_inputs(tmp_path):
    (tmp_path / 'a.cif').write_text('existing A')
    (tmp_path / 'b.cif').write_text('existing B')
    contract = {'kind': 'directory', 'path': str(tmp_path), 'pattern': '*.cif', 'min_count': 2}
    fact = artifact_fact(contract, tmp_path)
    assert artifact_is_valid(fact, fact, started_at=fact['mtime_ns'] / 1e9 + 10, read_only=True)
    assert not artifact_is_valid(fact, fact, started_at=fact['mtime_ns'] / 1e9 + 10)


def test_scientific_writer_requires_both_changed_and_fresh_artifact():
    baseline = {'size': 2, 'mtime_ns': 1_000_000_000}
    fresh = {'size': 3, 'mtime_ns': 3_000_000_000}
    changed_but_old = {'size': 3, 'mtime_ns': 1_000_000_000}
    assert artifact_is_valid(fresh, baseline, started_at=2)
    assert not artifact_is_valid(changed_but_old, baseline, started_at=2)
    assert not artifact_is_valid(fresh, fresh, started_at=2)


@pytest.mark.parametrize('fact', [None, {}, {'size': 0, 'mtime_ns': 1},
    {'size': -1, 'mtime_ns': 1}, {'size': True, 'mtime_ns': 1},
    {'size': float('nan'), 'mtime_ns': 1}, {'size': 2}, {'size': 2, 'mtime_ns': -1}])
def test_readonly_and_verified_reuse_cannot_accept_missing_or_invalid_artifacts(fact):
    assert not artifact_is_valid(fact, started_at=2, read_only=True)
    assert not artifact_is_valid(fact, started_at=2, verified_reuse=True)


def test_verified_reuse_preserves_runtime_owned_proof_exception():
    old = {'size': 2, 'mtime_ns': 1_000_000_000}
    assert artifact_is_valid(old, old, started_at=2, verified_reuse=True)
    assert not artifact_is_valid(old, old, started_at=2)


@pytest.mark.parametrize('started_at', [0, -1, float('nan'), float('inf'), None])
def test_scientific_writer_cannot_treat_missing_or_invalid_start_time_as_fresh(started_at):
    assert not artifact_is_valid({'size': 2, 'mtime_ns': 3_000_000_000}, started_at=started_at)
