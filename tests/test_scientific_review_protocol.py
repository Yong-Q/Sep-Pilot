"""Supplementary protocol checks; real SDK scenarios remain release acceptance."""
from types import SimpleNamespace
import hashlib

import pytest
from anthropic.types import ToolUseBlock

from agents.config import AgentConfig
from agents.scientific_review import canonical_source_inputs, digest, review_action
from agents.session import Session


def response(name, arguments, call_id):
    return SimpleNamespace(content=[ToolUseBlock(type='tool_use', id=call_id, name=name, input=arguments)])


def test_repeated_and_reordered_source_reads_keep_the_same_cache_key():
    a = {'path': '/owned/a', 'sha256': 'hash-a'}
    b = {'path': '/owned/b', 'sha256': 'hash-b'}
    assert digest(canonical_source_inputs([a, b])) == digest(canonical_source_inputs([b, a, a]))
    changed = {'path': '/owned/a', 'sha256': 'changed'}
    assert digest(canonical_source_inputs([a, b])) != digest(canonical_source_inputs([changed, b]))


def test_duplicate_source_references_reuse_verified_evidence_not_stale_data(tmp_path, monkeypatch):
    source = tmp_path / 'source.json'
    source.write_text('immutable source')
    item = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    monkeypatch.setattr(session, '_audit_supervisor_decision', lambda decision, event: decision)
    monkeypatch.setattr(session.registry, 'execute_dict', lambda name, arguments: {'ok': True})
    verdict = {'next_action': 'advance_dependencies', 'reason': 'verified source', 'evidence_refs': ['source'],
               'scientific_review': {'passed': True, 'issues': []}}
    responses = iter([
        response('convert_physical_units', {'value': 1, 'quantity': 'energy',
                 'from_unit': 'kcal/mol', 'to_unit': 'kJ/mol'}, 'source'),
        response('supervisor_decision', verdict, 'one'),
        response('supervisor_decision', verdict, 'two'),
        response('supervisor_decision', verdict, 'stale'),
        response('supervisor_decision', verdict, 'stale-again'),
        response('supervisor_decision', {'next_action': 'ask_user', 'reason': 'changed input requires reread',
            'evidence_refs': [], 'scientific_review': {'passed': False, 'issues': ['changed input']}}, 'insufficient'),
    ])
    session._call_api = lambda agent: next(responses)
    event = {'kind': 'scientific_review', 'event_id': 'science', 'payload': {'question': 'units',
             'verification_tools': ['convert_physical_units'], 'source_inputs': [item]}}
    first = session.observe_lifecycle_event(event)
    event['payload']['source_inputs'] = [item, item]
    repeated = session.observe_lifecycle_event(event)
    assert first['review_status'] == repeated['review_status'] == 'verified'
    assert first['independent_source_call_ids'] == repeated['independent_source_call_ids']
    assert len(repeated['evidence_calls']) == 1
    source.write_text('changed source')
    stale = session.observe_lifecycle_event(event)
    assert stale['review_status'] == 'insufficient_evidence'
    assert not stale['independent_source_call_ids']


@pytest.mark.parametrize('passed', [True, False])
def test_verdict_requires_independent_source_even_for_rejection(tmp_path, monkeypatch, passed):
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    monkeypatch.setattr(session, '_audit_supervisor_decision', lambda decision, event: decision)
    monkeypatch.setattr(session.registry, 'execute_dict', lambda name, arguments: {'ok': True, 'converted_value': 4.184})
    verdict = {'next_action': 'advance_dependencies' if passed else 'diagnose_and_fix',
               'reason': 'source-backed verdict', 'evidence_refs': ['source'],
               'scientific_review': {'passed': passed, 'issues': [] if passed else ['specific mismatch']}}
    responses = iter([
        response('supervisor_decision', verdict, 'premature'),
        response('convert_physical_units', {'value': 1, 'quantity': 'energy',
                 'from_unit': 'kcal/mol', 'to_unit': 'kJ/mol'}, 'source'),
        response('supervisor_decision', verdict, 'final'),
    ])
    source_required = []
    def call(agent):
        source_required.append(session._scientific_source_required)
        return next(responses)
    session._call_api = call
    receipt = session.observe_lifecycle_event({'kind': 'scientific_review', 'event_id': 'science',
        'payload': {'question': 'units', 'verification_tools': ['convert_physical_units']}})
    assert source_required == [True, True, False]
    assert session.memory.tool_call_log[0]['failed']
    assert receipt['independent_source_call_ids']
    assert receipt['review_status'] == ('verified' if passed else 'rejected_content')
    assert review_action({}, receipt) == ('verified' if passed else 'correct_main')
    assert not session._scientific_source_required


def test_failed_source_read_never_becomes_main_answer_failure(tmp_path, monkeypatch):
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    monkeypatch.setattr(session, '_audit_supervisor_decision', lambda decision, event: decision)
    monkeypatch.setattr(session.registry, 'execute_dict', lambda name, arguments: {'error': 'source unavailable'})
    query = {'value': 1, 'quantity': 'energy', 'from_unit': 'kcal/mol', 'to_unit': 'kJ/mol'}
    responses = iter([
        response('convert_physical_units', query, 'one'),
        response('convert_physical_units', query, 'two'),
        response('supervisor_decision', {'next_action': 'ask_user', 'reason': 'source unavailable',
            'evidence_refs': [], 'scientific_review': {'passed': False, 'issues': ['source unavailable']}}, 'final'),
    ])
    session._call_api = lambda agent: next(responses)
    receipt = session.observe_lifecycle_event({'kind': 'scientific_review', 'event_id': 'science',
        'payload': {'question': 'units', 'verification_tools': ['convert_physical_units']}})
    assert receipt['review_status'] == 'insufficient_evidence'
    assert not receipt['independent_source_call_ids']
    state = {}
    assert review_action(state, receipt) == 'retry_supervisor'
    assert review_action(state, receipt) == 'ask_user'
    assert not state.get('corrections')
    assert not session._scientific_source_required
