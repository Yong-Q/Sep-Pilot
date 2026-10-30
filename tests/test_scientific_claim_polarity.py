"""Explicit scientific denials cannot erase positive or numerical claims."""
import pytest
from agents.report_authoring import _positive_mixture_selectivity_claim, render_validated_draft


def packet():
    return {'title': 'Controlled report', 'sources': [{'id': 'S1', 'step_id': 'native',
        'call_id': 'controlled', 'source_sha256': 'a' * 64, 'conditions': {}, 'documents': [],
        'structured_data': [], 'text_documents': []}], 'tables': [], 'references': [], 'figures': []}


def render(text):
    return render_validated_draft(packet(), {'blocks': [{'kind': 'narrative', 'text': text,
        'source_ids': ['S1'], 'claim_type': 'limitation'}]})


@pytest.mark.parametrize('text', [
    '表中给出的比值为单组分吸附量之比，其定义列明确说明它并非 IAST 选择性；本片段不将图中比值解读为选择性，也不推断混合物条件下的分离性能。',
    '该比值不是 IAST 选择性。',
    '这是单组分载量比，不能称为混合气选择性。',
    'This ratio is not IAST selectivity.',
    'It does not prove mixture selectivity.',
    '单组分比值为数据表中的指标。IAST 结果并未计算。',
])
def test_explicit_denial_or_separate_sentences_do_not_assert_mixture_selectivity(text):
    assert not _positive_mixture_selectivity_claim(text)
    assert text in render(text)


@pytest.mark.parametrize('text', [
    '该比值是 IAST 选择性。',
    '这为混合气选择性。',
    'The ratio is IAST selectivity.',
    'These data prove mixture selectivity.',
    '不是 IAST 选择性，但是该比值是混合气选择性。',
    '不是 IAST 选择性。该比值为混合气选择性。',
    '不是单组分结果而是 IAST 选择性。',
    'Not IAST, but the ratio represents mixture selectivity.',
    '并非没有证据，结果是 IAST 选择性。',
])
def test_positive_claim_or_mixed_polarity_still_fails(text):
    assert _positive_mixture_selectivity_claim(text)
    with pytest.raises(ValueError, match='unsupported mixture-selectivity interpretation'):
        render(text)


def test_denial_does_not_bypass_independent_number_gate():
    with pytest.raises(ValueError, match='unsupported numerical claim'):
        render('该比值不是 IAST 选择性，数值为 9876。')
