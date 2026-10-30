import copy
from agents.report_authoring import author_report
from tests.test_report_fact_pointer_diagnostics import packet,fact


def test_rejected_draft_returns_to_author_with_exact_feedback(tmp_path):
    p=packet(tmp_path);calls=[]
    invalid={'kind':'narrative','text':'The output System_0 is tracked by SHA-256.','source_ids':['S1'],'claim_type':'method'}
    valid={'kind':'narrative','text':'The source verification status has a limited scope.','source_ids':['S1'],'claim_type':'limitation'}
    def model(phase,payload):
        calls.append(phase)
        if phase=='author':
            if payload['previous_draft'] is None:
                return {'actor':'communicator','call_id':'first','draft':{'blocks':[copy.deepcopy(invalid),fact('/native_info/converged')]}}
            assert payload['previous_draft']['blocks'][0]==invalid
            assert 'block[0]' in payload['previous_issues'][0] and 'unsupported numerical claim' in payload['previous_issues'][0]
            assert 'paths, hashes' in payload['required_output']
            return {'actor':'communicator','call_id':'second','draft':{'blocks':[copy.deepcopy(valid),fact('/native_info/converged')]}}
        return {'actor':'report-reviewer','call_id':'review','review':{'passed':True,'draft_sha256':payload['draft_sha256'],'checked_blocks':[0,1],'blocking_issues':[]}}
    result=author_report(p,model_call=model)
    assert calls==['author','author','review'] and result['verified'] is True
    assert 'System_0' not in result['markdown'] and 'SHA-256' in result['markdown']  # native source traceability remains
    assert result['scientific_acceptance'] is False


def test_review_rejection_is_sent_back_with_unchanged_draft(tmp_path):
    p=packet(tmp_path);calls=[]
    narrative={'kind':'narrative','text':'The source verification status has a limited scope.','source_ids':['S1'],'claim_type':'limitation'}
    draft={'blocks':[narrative,fact('/native_info/converged')]}
    def model(phase,payload):
        calls.append(phase)
        if phase=='author':
            if calls.count('author')==2:
                assert payload['previous_draft']==draft
                assert 'independent report review' in payload['previous_issues'][0]
            return {'actor':'communicator','call_id':'author'+str(len(calls)),'draft':copy.deepcopy(draft)}
        first=calls.count('review')==1
        return {'actor':'report-reviewer','call_id':'review'+str(len(calls)),'review':{'passed':not first,
            'draft_sha256':payload['draft_sha256'],'checked_blocks':[0,1],
            'blocking_issues':['Source interpretation needs clarity'] if first else []}}
    result=author_report(p,model_call=model)
    assert calls==['author','review','author','review'] and result['verified'] is True


def test_figure_notes_remain_visible_but_are_not_scientific_acceptance(tmp_path):
    from agents.report_authoring import render_validated_draft
    p=packet(tmp_path)
    p['figures']=[{'id':'F1','source_id':'S1','path':'figures/a.png','visual_review':'verified',
        'visual_evidence':{'status':'verified','scientific_acceptance':False,
            'author_limitations':['Ratio uncertainty assumes independent errors; this assumption is unverified.'],
            'review_limitations':['The underlying uncertainty definition is not supplied.']}}]
    narrative={'kind':'narrative','text':'The source has limited scope.','source_ids':['S1'],'claim_type':'limitation'}
    text=render_validated_draft(p,{'blocks':[narrative,{'kind':'figure','figure_id':'F1'}]})
    assert 'assumes independent errors' in text and 'underlying uncertainty definition' in text
    assert 'Figure author limitation' in text and 'unverified' in text and 'scientific acceptance remains false' in text
