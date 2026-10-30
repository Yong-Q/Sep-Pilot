import pytest
from agents.report_authoring import _nonquantitative_literature_denial

def test_provenance_requirement_is_not_a_literature_comparison():
 assert _nonquantitative_literature_denial('必须由对应的原生产出步骤提供证据块，而不是由外部文献值替代。')

@pytest.mark.parametrize('text',[
 '不是由外部文献值替代，但与文献一致。',
 '不是由外部文献值替代，却与文献相符。',
 '不是由外部文献值替代并与文献一致。',
 '不是由外部文献值替代并与其一致。',
 '不是由外部文献值替代并与文献进行比较。',
 '不是由外部文献数值替代其中12个值。',
 '结果与文献一致。',
])
def test_positive_mixed_or_numeric_claims_remain_blocked(text):
 assert not _nonquantitative_literature_denial(text)
