import json

from agents import registry
from agents.recovery import failure_reason


def test_search_without_matches_is_successful_empty_result(tmp_path, monkeypatch):
    source = tmp_path / 'native.log'; source.write_text('real native values\n')
    monkeypatch.setattr(registry, '_resolve_fs_path', lambda value: str(source))
    result = registry._exec_grep_search({'pattern': 'NOT_PRESENT', 'path': str(source)})
    assert result['total_matches'] == 0 and result['results'] == []
    assert result['process_exit_code'] == 1
    assert failure_reason(json.dumps(result)) == ''


def test_invalid_regex_remains_a_real_search_error(tmp_path, monkeypatch):
    source = tmp_path / 'native.log'; source.write_text('native\n')
    monkeypatch.setattr(registry, '_resolve_fs_path', lambda value: str(source))
    result = registry._exec_grep_search({'pattern': '[', 'path': str(source)})
    assert result['exit_code'] > 1
    assert failure_reason(json.dumps(result))
