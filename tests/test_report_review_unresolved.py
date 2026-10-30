import pytest
from tests.test_report_contract_feedback import packet,authored,review
from agents import report_authoring as m


def test_repeated_reviewer_rejection_preserves_feedback_not_success(packet):
    def model(phase,payload):return authored(['T1','T2']) if phase=='author' else review(payload,False)
    with pytest.raises(m.ReportReviewUnresolved) as exc:m.author_report(packet,model_call=model)
    assert not isinstance(exc.value,m.ReportContractConflict)
    assert exc.value.details['contract_code']=='report_review_unresolved'
    assert exc.value.details['independent_review_issues']==['The instructions conflict with this presentation.']
    assert 'not assumed correct' in exc.value.details['resolution']


def test_ordinary_success_still_requires_exact_positive_review(packet):
    def model(phase,payload):return authored(['T1','T2']) if phase=='author' else review(payload,True)
    assert m.author_report(packet,model_call=model)['verified'] is True


def test_authenticated_unresolved_review_reaches_planner(tmp_path,monkeypatch):
    from tests.test_repair_event_isolation import setup
    from tests.test_report_feedback_recovery import install
    from agents.state_io import json_transaction
    machine,store,actor,root=setup(tmp_path);_,proof=install(machine,store)
    with json_transaction(store.path) as data:
        data['workflows']['workflow']['nodes']['one']['result']['contract_feedback']['contract_code']='report_review_unresolved'
    monkeypatch.setattr(machine,'_source',lambda state,step:proof)
    seen=[];machine.run_repair(actor(),'incident-one',4,checkpoint_path=root/'one.json',stage_runner=lambda e:(seen.append(e) or {'status':'blocked'}))
    assert seen[0]['payload']['report_contract_feedback']['contract_code']=='report_review_unresolved'
    assert 'not user instruction' in seen[0]['payload']['report_contract_feedback_trust']


@pytest.mark.parametrize('tamper',['actor','call_id','draft_sha256','checked_blocks'])
def test_invalid_review_never_becomes_authenticated_review_feedback(packet,tamper):
    def model(phase,payload):
        if phase=='author':return authored(['T1','T2'])
        result=review(payload,False)
        if tamper=='actor':result['actor']='other'
        elif tamper=='call_id':result['call_id']=''
        elif tamper=='draft_sha256':result['review']['draft_sha256']='wrong'
        else:result['review']['checked_blocks']=[]
        return result
    with pytest.raises(ValueError) as exc:m.author_report(packet,model_call=model)
    assert not isinstance(exc.value,m.ReportReviewUnresolved)
