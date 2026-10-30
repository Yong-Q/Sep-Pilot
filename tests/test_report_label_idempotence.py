import pytest
from agents.report_authoring import render_validated_draft
from tests.test_report_inference_label import packet,narrative

@pytest.mark.parametrize('kind,label',[
 ('recommendation','Proposed follow-up / 建议（未验证）：'),
 ('hypothesis','Unverified hypothesis / 未验证假设：'),
 ('inference','Inference / 推断（待核验）：'),
 ('derived','Derived from cited data / 来源推导：')])
def test_exact_native_label_appears_once(kind,label):
    rendered=render_validated_draft(packet(),narrative(kind,label+'The source has limited scope.'))
    assert rendered.count(label)==1


def test_label_does_not_authorize_unsupported_numbers():
    with pytest.raises(ValueError,match='numeric recommendation lacks'):
        render_validated_draft(packet(),narrative('recommendation','Proposed follow-up / 建议（未验证）：Use 100 cycles.'))


def test_wrong_label_is_not_treated_as_the_current_claim_label():
    result=render_validated_draft(packet(),narrative('recommendation','Inference / 推断（待核验）：Check the evidence later.'))
    assert result.count('Proposed follow-up / 建议（未验证）：')==1
    assert 'Inference / 推断（待核验）：' in result


def test_label_mid_sentence_does_not_remove_required_prefix():
    result=render_validated_draft(packet(),narrative('recommendation','Note: Proposed follow-up / 建议（未验证）：Check evidence.'))
    assert result.count('Proposed follow-up / 建议（未验证）：')==2


def test_leading_space_still_produces_one_label_and_retains_source_citation():
    label='Proposed follow-up / 建议（未验证）：'
    result=render_validated_draft(packet(),narrative('recommendation','  '+label+'Check the source.'))
    assert result.count(label)==1 and '[S1]' in result
