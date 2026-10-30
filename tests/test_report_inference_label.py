import pytest
from agents.report_authoring import render_validated_draft

def packet():return {'title':'test','sources':[{'id':'S1','step_id':'source','call_id':'call','source_sha256':'a'*64,'conditions':{}}],'tables':[],'references':[],'figures':[]}
def narrative(kind,text):return {'blocks':[{'kind':'narrative','claim_type':kind,'source_ids':['S1'],'text':text}]}

def test_inference_and_hypothesis_have_distinct_nonverified_labels():
 inferred=render_validated_draft(packet(),narrative('inference','The supplied trend is consistent with a possible effect.'))
 hypothesized=render_validated_draft(packet(),narrative('hypothesis','A possible mechanism remains speculative.'))
 assert 'Inference / 推断（待核验）' in inferred
 assert '未验证假设' not in inferred
 assert 'Unverified hypothesis / 未验证假设' in hypothesized

def test_inference_cannot_invent_numbers():
 with pytest.raises(ValueError,match='unsupported numerical claim'):
  render_validated_draft(packet(),narrative('inference','The true value must be 97 mol/kg.'))
