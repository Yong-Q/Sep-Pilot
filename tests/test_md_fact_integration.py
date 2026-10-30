"""Native MD facts reach production inspection; synthetic log fixtures only."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.result_validation import build_validation_dossier


def native_log(force, pressure, stop):
    return f'''Setting up cg style minimization ...
Unit style : real
Step Temp TotEng Press Volume
1 0 677.3576 {pressure} 16755.846
Loop time of 1 on 1 procs for 1 steps
100% CPU use with 1 MPI tasks x no OpenMP threads
Minimization stats:
Stopping criterion = {stop}
Energy initial, next-to-last, final = 678 677.357595995 677.357595995
Force two-norm initial, final = {force} {force}
Force max component initial, final = 14.3773 14.3773
Final line search alpha, max atom move = 3e-11 4e-10
Iterations, force evaluations = 1 43
MPI task timing breakdown:
'''


@pytest.fixture
def md_run(tmp_path):
    root = tmp_path / 'owned-md'; root.mkdir()
    (root / 'md_result.json').write_text(json.dumps({'success': True, 'energy': 677.357595995, 'mode': 'single'}))
    text = native_log('.282444', '-22.888999', 'energy tolerance') + native_log('14.3778', '-19.611668', 'linesearch alpha is zero')
    (root / 'solver.log').write_text(text)
    return root


def node(root, mode='single'):
    return {'workflow_id': 'owned-workflow', 'step_id': 'md', 'plan_version': 1,
        'contract': {'tool': 'run_md_optimize', 'arguments': {'output_dir': str(root), 'mode': mode},
                     'expected_outputs': [str(root / 'md_result.json')]},
        'result': {'work_dir': str(root)}, 'path_manifest': {'result_dir': str(root)}}


def test_dossier_contains_last_stage_facts_without_calling_parse_success_convergence(md_run, tmp_path):
    dossier = build_validation_dossier(node(md_run), 'attempt', tmp_path / 'evidence')
    assert dossier['schema_version'] >= 3
    facts = dossier['method_facts']['md_native']
    assert facts['mode'] == 'single'
    assert facts['adapter_reported_success'] is True
    final = facts['solver_log']['final_minimization']
    assert final['force_two_norm_final'] == 14.3778 and final['native_tolerance_stop'] is False
    assert facts['solver_log']['final_minimization_thermo']['values']['Press'] == -19.611668
    assert facts['line_refs']['final_minimization']['sha256'] == hashlib.sha256((md_run / 'solver.log').read_bytes()).hexdigest()
    assert dossier['requires_agent_review'] is True
    assert 'converged' not in facts


def test_md_mode_labels_minimization_scope_and_keeps_original_result(md_run, tmp_path):
    original = (md_run / 'md_result.json').read_bytes()
    facts = build_validation_dossier(node(md_run, 'md'), 'attempt', tmp_path / 'evidence')['method_facts']['md_native']
    assert facts['mode'] == 'md'
    assert 'minimization' in facts['thermo_scope'] and 'NPT' in facts['thermo_scope']
    assert (md_run / 'md_result.json').read_bytes() == original


def test_schema_upgrade_writes_new_evidence_without_overwriting_old(md_run, tmp_path, monkeypatch):
    import agents.result_validation as module
    with monkeypatch.context() as patch:
        patch.setattr(module, 'VALIDATION_SCHEMA_VERSION', 2)
        old = build_validation_dossier(node(md_run), 'attempt', tmp_path / 'evidence')
    original = Path(old['evidence_path']).read_bytes()
    new = build_validation_dossier(node(md_run), 'attempt', tmp_path / 'evidence')
    assert old['evidence_call_id'] != new['evidence_call_id']
    assert Path(old['evidence_path']).read_bytes() == original


def test_legacy_mode_comes_from_exact_submission_parameters_not_assumptions(md_run):
    from bimem_agent.result_inspector import inspect_md_run
    (md_run / 'md_result.json').write_text('{"success": true, "energy": 677.3576}')
    (md_run / 'submit.sh').write_text("#!/bin/bash\npython /owned/tools/md_runtime.py --parameters '{\"mode\":\"single\"}'\n")
    facts = inspect_md_run(md_run)
    assert facts['mode'] == 'single' and facts['mode_source'] == 'submit.sh parameters'
    assert facts['line_refs']['mode']['line'] == 2


def test_registered_run_inspection_adds_facts_without_following_foreign_spec(md_run, tmp_path, monkeypatch):
    from bimem_agent import result_inspector
    private = tmp_path / 'private-spec.json'; private.write_text('SYNTHETIC_OUTSIDE_SECRET')
    spec = md_run / 'spec.json'; spec.symlink_to(private)
    record = {'run_dir': str(md_run), 'result_path': str(md_run / 'md_result.json'), 'spec_path': str(spec)}
    monkeypatch.setattr(result_inspector, 'load_run', lambda *a, **k: record)
    result = result_inspector.inspect_run('owned-run')
    assert result['md_native']['solver_log']['final_minimization']['force_two_norm_final'] == 14.3778
    assert 'spec' not in result and result['inspection_errors']
    assert 'SYNTHETIC_OUTSIDE_SECRET' not in json.dumps(result)


def test_md_inspector_only_reads_exact_local_files_and_rejects_symlink_escape(md_run, tmp_path, monkeypatch):
    from bimem_agent.result_inspector import inspect_md_run
    outside = tmp_path / 'outside.log'; outside.write_text('SYNTHETIC_OUTSIDE_SECRET')
    (md_run / 'solver.log').unlink(); (md_run / 'solver.log').symlink_to(outside)
    seen = []
    original = Path.open
    def guarded(path, *a, **kw):
        if path.resolve() == outside:
            seen.append(path); pytest.fail('outside log must not be opened')
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', guarded)
    facts = inspect_md_run(md_run)
    assert not facts['ok'] and not seen
    assert 'SYNTHETIC_OUTSIDE_SECRET' not in json.dumps(facts)


@pytest.mark.parametrize('blocker', [None, 'foreign_owner', 'different_workdir'])
def test_numeric_inspect_run_exposes_facts_only_for_owned_matching_md_receipt(md_run, monkeypatch, blocker):
    from agents import registry, job_watch
    from agents.watch_context import set_context
    record = {'job_id': '109107', 'tool': 'run_md_optimize', 'username': 'alice',
              'conv_id': 'md', 'work_dir': str(md_run)}
    if blocker == 'foreign_owner': record['username'] = 'bob'
    monkeypatch.setattr(job_watch, 'get_watch', lambda: SimpleNamespace(get=lambda _: record))
    monkeypatch.setattr(registry.slurm, 'check_job_status', lambda *a, **kw: {'state': 'COMPLETED'})
    set_context('alice', 'md', 'lead-orchestrator', '')
    params = {'run_id': '109107', 'work_dir': str(md_run.parent / 'different') if blocker == 'different_workdir' else str(md_run)}
    result = registry._exec_inspect_run(params)
    if blocker:
        assert result.get('blocked') and 'md_native' not in result
    else:
        assert result['md_native']['solver_log']['final_minimization']['force_two_norm_final'] == 14.3778


def test_owned_terminal_md_inspection_survives_scheduler_history_purge(md_run, monkeypatch):
    from agents import registry, job_watch
    from agents.watch_context import set_context
    record = {'job_id': '109107', 'tool': 'run_md_optimize', 'username': 'alice', 'conv_id': 'md',
              'work_dir': str(md_run), 'terminal': True, 'failed': False, 'state': 'COMPLETED'}
    monkeypatch.setattr(job_watch, 'get_watch', lambda: SimpleNamespace(get=lambda _: record))
    monkeypatch.setattr(registry.slurm, 'check_job_status', lambda *a, **k: pytest.fail('owned terminal receipt needs no scheduler query'))
    set_context('alice', 'md', 'lead-orchestrator', '')
    result = registry._exec_inspect_run({'run_id': '109107'})
    assert result['state'] == 'COMPLETED' and result['source'] == 'owned_job_watch_receipt'
    assert result['md_native']['solver_log']['final_minimization']['native_tolerance_stop'] is False
