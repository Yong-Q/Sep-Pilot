import pytest
from agents.report_authoring import _positive_normality_or_causality

@pytest.mark.parametrize('text',[
 '在该参数下求解器能否正常启动、运行并退出',
 '检查该结果是否正常', '能否正常运行有待验证', '是否正常运行仍有待验证',
 'Is it normal?',
])
def test_question_is_not_an_assertion(text):assert not _positive_normality_or_causality(text)

@pytest.mark.parametrize('text',[
 '结果正常', '体系符合预期', '结果必然正确', 'The result is normal',
 '能否正常尚未确认，但结果正常',
 '是否正常，结果正常',
 '检查是否正常。结果正常',
 '是否正常且结果正常',
 '是否正常已经得到证明',
 '是否正常运行这一点毋庸置疑', '是否正常运行这一点已得到验证', '是否正常运行获得验证', '是否正常运行得到了验证', '是否正常运行已经确认', '是否正常运行可认定',
 '能否正常而且这次结果正常', '是否正常而结果正常', '是否正常就是正常', '是否正常显示结果正常',
])
def test_positive_mixed_and_cross_sentence_still_blocked(text):assert _positive_normality_or_causality(text)


def test_recommendation_feedback_preserves_numeric_evidence_gate(tmp_path):
    from agents.report_authoring import render_validated_draft
    from tests.test_report_fact_pointer_diagnostics import packet
    p=packet(tmp_path)
    draft={'blocks':[{'kind':'narrative','text':'The observed record has 20 iterations; inspect the outcome further.',
                      'claim_type':'recommendation','source_ids':['S1']}]}
    with pytest.raises(ValueError,match='numeric recommendation lacks') as e:
        render_validated_draft(p,draft)
    assert 'separate source-backed observed conditions' in str(e.value)
    assert 'never invent a numeric recommendation' in str(e.value)
