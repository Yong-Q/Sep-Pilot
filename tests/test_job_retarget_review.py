"""Independent fixed-budget resource feedback checks, never scheduler actions."""
import copy
from subprocess import TimeoutExpired
from types import SimpleNamespace

from agents.job_control import retarget_compatibility
from tests.test_job_retarget_feedback import assessment


def test_alternative_with_shorter_partition_walltime_is_not_mixed_with_valid_targets():
    facts = assessment()
    short = copy.deepcopy(facts['node_inventory']['nodes']['good'])
    short.update(name='short', partitions=['short_queue'])
    facts['node_inventory']['nodes']['short'] = short
    facts['partition_limits']['short_queue'] = {'verified': True, 'state': 'UP', 'max_time': '2-00:00:00'}
    before = copy.deepcopy(facts)
    result = retarget_compatibility(facts, {'tool': 'run_cdft'}, 'planned', 'compute', 16384)
    assert result['target_compatible'] is False
    assert [(n['nodelist'], n['partition'], n['memory_mb'], n['walltime']) for n in result['compatible_targets']] == [
        ('good', 'compute', 16384, '3-00:00:00'),
    ]
    assert 'partition_walltime_too_short' in result['target_rejections']['short_queue/short']
    assert facts == before


def test_gpu_capable_target_does_not_create_a_gpu_reservation_for_an_underrequested_job():
    facts = assessment()
    facts['node_inventory']['nodes']['good']['gpus'] = {'verified': True, 'available': 8, 'total': 8}
    result = retarget_compatibility(facts, {'tool': 'run_string_tst'}, 'good', 'compute', 16384)
    assert result['target_compatible'] is False
    assert result['compatible_targets'] == []
    assert 'existing_gpu_request_below_native_or_operator_requirement' in result['reason_codes']


def test_assessment_with_unavailable_partition_probe_never_suggests_a_target(tmp_path, monkeypatch):
    from agents.resource_review import assess_job_resources
    facts = assessment()
    calls = []
    monkeypatch.setattr('agents.job_watch.get_watch', lambda: SimpleNamespace(get=lambda _: {
        'tool': 'run_cdft', 'username': 'a', 'conv_id': 'c', 'work_dir': str(tmp_path),
    }))
    monkeypatch.setattr('agents.watch_context.get_context', lambda: {'username': 'a', 'conv_id': 'c'})
    monkeypatch.setattr('agents.config.get_config', lambda: SimpleNamespace(project_root=tmp_path))
    monkeypatch.setattr('agents.node_inventory.node_inventory', lambda *args, **kwargs: facts['node_inventory'])
    monkeypatch.setattr('agents.slurm.check_job_status', lambda *args, **kwargs: {'state': 'PENDING'})
    def run(argv, **kwargs):
        calls.append(argv)
        if argv == ['scontrol', 'show', 'partition', '-o']:
            raise TimeoutExpired(argv, 8)
        assert argv == ['scontrol', 'show', 'job', '123', '-o']
        return SimpleNamespace(returncode=0,
            stdout=' '.join(f'{k}={v}' for k, v in facts['effective_constraints'].items()), stderr='')
    monkeypatch.setattr('agents.resource_review.subprocess.run', run)
    result = assess_job_resources('123')
    assert result['read_only'] is True
    assert result['compatible_targets'] == [] and result['partition_limits'] == {}
    assert calls == [['scontrol', 'show', 'job', '123', '-o'], ['scontrol', 'show', 'partition', '-o']]
