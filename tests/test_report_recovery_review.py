"""Independent recovery protocol boundaries; synthetic owned files only."""
from agents.report_recovery import verify_returned_readonly
from tests.test_report_recovery import build_returned_readonly_case


def test_running_returned_readonly_proves_existing_result_without_reexecution(tmp_path):
    case = build_returned_readonly_case(tmp_path)
    case['state']['nodes']['qc']['status'] = 'running'
    proof = verify_returned_readonly(case['state'], 'qc', case['root'])
    assert proof['kind'] == 'current_returned_evidence'
    assert proof['no_resubmission'] is True
    assert proof['no_output_write'] is True
    assert proof['execution_call_id'] == 'readonly_call'
    assert proof['current_token'] == 'returned_token'
    assert case['state']['nodes']['qc']['status'] == 'running'
