import importlib.util
from pathlib import Path
import pytest
import agents

import agents.report_authoring as m

def packet():
    return {'title':'Report', 'root':'/tmp', 'packet_sha256':'a'*64,
            'sources':[{'id':'S1','step_id':'step','call_id':'call',
                        'source_sha256':'b'*64,'conditions':{}}],
            'tables':[], 'references':[], 'figures':[],
            'independent_review_required':False}

def test_light_author_receives_consistent_instruction_and_keeps_prose(monkeypatch):
    monkeypatch.setattr(m, 'sources_unchanged', lambda *args: True)
    calls=[]
    def model(phase, payload):
        calls.append(phase)
        assert 'source-bound explanations' in payload['required_output']
        assert 'Unverified narrative cannot be delivered' not in payload['required_output']
        return {'actor':'communicator','call_id':'c1','draft':{'blocks':[
            {'kind':'narrative','claim_type':'inference','source_ids':['S1'],
             'text':'The result may indicate a possible effect.'}]}}
    result = m.author_report(packet(), model_call=model)
    assert calls == ['author'] and result['independent_review_performed'] is False
    assert 'possible effect' in result['markdown']

def test_unrequested_table_does_not_block_report():
    p=packet(); p['tables']=[{'id':'T1','source_id':'S1','columns':['value'],
                              'rows':[{'value':1}],'row_count':1,'path':'a.csv'}]
    draft={'blocks':[{'kind':'narrative','claim_type':'limitation','source_ids':['S1'],
                      'text':'The result has limited scope.'}]}
    out = m.render_validated_draft(p, draft)
    assert 'limited scope' in out
    assert 'unavailable (native missing/nonfinite)' not in out
