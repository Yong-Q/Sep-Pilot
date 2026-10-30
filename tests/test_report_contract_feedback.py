import copy
import pytest
from agents import report_authoring as m

@pytest.fixture
def packet(monkeypatch):
    monkeypatch.setattr(m, 'sources_unchanged', lambda *a: True)
    monkeypatch.setattr(m, 'author_stage_schema', lambda p: {})
    def render(p, d):
        used={b.get('table_id') for b in d['blocks'] if b.get('kind')=='table'}
        if {x['id'] for x in p['tables']}-used:
            raise ValueError('full source table coverage is required')
        return 'rendered'
    monkeypatch.setattr(m, 'render_validated_draft', render)
    return {'root':'/controlled', 'packet_sha256':'controlled', 'tables':[
        {'id':'T1','columns':['material','value']}, {'id':'T2','columns':['material','unit','quality']}]}

def authored(tables):
    return {'actor':'communicator','call_id':'a','draft':{'blocks':[{'kind':'table','table_id':t} for t in tables]}}

def review(payload, passed):
    return {'actor':'report-reviewer','call_id':'r','review':{'passed':passed,'draft_sha256':payload['draft_sha256'],
        'checked_blocks':list(range(len(payload['draft']['blocks']))), 'blocking_issues':[] if passed else ['The instructions conflict with this presentation.']}}

def test_review_then_missing_table_produces_actionable_bounded_feedback(packet):
    calls=[]
    def model(phase,payload):
        calls.append(phase)
        if phase=='review':return review(payload,False)
        return authored(['T1','T2'] if calls.count('author')==1 else ['T1'])
    with pytest.raises(m.ReportContractConflict) as exc:m.author_report(packet,model_call=model)
    details=exc.value.details
    assert details['contract_code']=='possible_report_constraint_conflict'
    assert details['missing_table_ids']==['T2'] and details['source_tables'][1]['available_columns']==['material','unit','quality']
    assert details['independent_review_issues']==['The instructions conflict with this presentation.']
    assert 'Do not fabricate' in details['resolution'] and calls==['author','review','author']

def test_third_round_receives_both_review_and_coverage(packet):
    calls=[]
    def model(phase,payload):
        calls.append(phase)
        if phase=='review':return review(payload,calls.count('review')==2)
        n=calls.count('author')
        if n==3:
            assert 'coverage' in payload['previous_issues'][0]
            assert any('instructions conflict' in x for x in payload['previous_issues'])
        return authored(['T1'] if n==2 else ['T1','T2'])
    assert m.author_report(packet,model_call=model,max_rounds=3)['verified'] is True

def test_coverage_failure_without_independent_review_is_not_labeled_conflict(packet):
    with pytest.raises(ValueError) as exc:m.author_report(packet,model_call=lambda phase,payload:authored(['T1']))
    assert not isinstance(exc.value,m.ReportContractConflict)

def test_ordinary_complete_report_remains_subject_to_review(packet):
    def model(phase,payload):return authored(['T1','T2']) if phase=='author' else review(payload,True)
    assert m.author_report(packet,model_call=model)['verified'] is True
