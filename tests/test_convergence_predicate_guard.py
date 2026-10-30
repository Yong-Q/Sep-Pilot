import pytest
from agents.report_authoring import render_validated_draft

def packet(converged=None):return {'title':'test','sources':[{'id':'S1','step_id':'science','call_id':'call','source_sha256':'a'*64,'conditions':{},'scientific_convergence':converged}],'tables':[],'references':[],'figures':[]}
def draft(text):return {'blocks':[{'kind':'narrative','source_ids':['S1'],'claim_type':'limitation','text':text}]}

@pytest.mark.parametrize('text',[
 '完整保留源表全部列以便复核，包括方法、单位、收敛标记、力场与客体定义。',
 '全部列包括收敛字段，状态未知。',
 '全部列包括收敛状态，未验证。',
])
def test_listing_fields_does_not_claim_convergence(text):
 assert 'Source traceability' in render_validated_draft(packet(),draft(text))

@pytest.mark.parametrize('text',[
 '全部结构已经收敛。',
 '所有结果没有收敛，但全部结构已经收敛。',
 '全部计算的收敛状态为已收敛。',
 'The source has not converged, but all results converged.',
])
def test_positive_or_mixed_claim_still_requires_native_convergence(text):
 with pytest.raises(ValueError,match='unsupported convergence claim'):
  render_validated_draft(packet(),draft(text))

def test_positive_claim_requires_all_referenced_sources_to_be_proven():
 result=render_validated_draft(packet(True),draft('全部结构已经收敛。'))
 assert '全部结构已经收敛' in result
