import pytest
from agents.report_authoring import _nonquantitative_literature_denial,render_validated_draft

@pytest.mark.parametrize('text',[
 '本次未做文献比对，无相应证据。证据包中不含参考文献条目。',
 '本次没有做文献比较，来源中没有参考文献记录。',
 '本次未进行文献对照，未包含参考文献条目。',
])
def test_explicit_absence_is_not_a_positive_literature_comparison(text):
 assert _nonquantitative_literature_denial(text)

@pytest.mark.parametrize('text',[
 '本次未做文献比对，但结果与文献一致。',
 '本次未做文献比对并与文献一致。',
 '证据包中不含参考文献条目却与文献相符。',
 '证据包中不含参考文献条目并与其一致。',
 '没有做文献比对却声称与文献吻合。',
 '本次未做 12 项文献比对。',
])
def test_positive_mixed_or_numeric_statements_remain_untrusted(text):
 assert not _nonquantitative_literature_denial(text)

def test_reference_absence_must_match_actual_packet():
 source={'id':'S1','step_id':'read','call_id':'call','source_sha256':'a'*64,'conditions':{}}
 p={'title':'test','sources':[source],'tables':[],'figures':[],'references':[{'id':'R1','support_kind':'supplied_excerpt','snippet':'citation'}]}
 draft={'blocks':[{'kind':'narrative','source_ids':['S1'],'claim_type':'limitation','text':'证据包中不含参考文献条目。'}]}
 with pytest.raises(ValueError,match='claims no references'):render_validated_draft(p,draft)
