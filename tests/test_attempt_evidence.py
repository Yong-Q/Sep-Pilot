import pytest
from agents.attempt_evidence import preserve_attempt_logs


def test_retry_preserves_original_submission_and_log_without_overwriting(tmp_path):
    work = tmp_path / 'calculation'
    work.mkdir()
    (work / 'generation.log').write_text('original failure')
    (work / 'submit.sh').write_text('original script')
    entry = {'attempt_id': 'a' * 32, 'params': {'output_dir': str(work)}}
    receipt = preserve_attempt_logs(entry, tmp_path)
    assert len(receipt['files']) == 2
    assert preserve_attempt_logs(entry, tmp_path) == receipt
    (work / 'generation.log').write_text('later output')
    assert (tmp_path / 'attempt_evidence' / ('a' * 32) / 'calculation/generation.log').read_text() == 'original failure'
    with pytest.raises(ValueError, match='refusing to overwrite'):
        preserve_attempt_logs(entry, tmp_path)


def test_archive_never_reads_outside_session_or_follows_symlink(tmp_path):
    private = tmp_path / 'private';private.mkdir()
    own = tmp_path / 'own';own.mkdir()
    (private / 'private.log').write_text('private')
    (own / 'link.log').symlink_to(private / 'private.log')
    assert not preserve_attempt_logs({'attempt_id': 'b' * 32, 'params': {'work_dir': str(private)}}, own)['files']
    assert not preserve_attempt_logs({'attempt_id': 'b' * 32, 'params': {'work_dir': str(own)}}, own)['files']
