"""No evidence is needed to deny comparability; factual external claims still do."""
import pytest
from agents.report_authoring import render_validated_draft


def packet():
    return {'title':'Bounded inference','sources':[{'id':'S1','step_id':'native','call_id':'native-call',
        'source_sha256':'a'*64,'conditions':{},'documents':[],'structured_data':[],'text_documents':[]}],
        'tables':[],'references':[],'figures':[]}


def render(text,claim='inference'):
    return render_validated_draft(packet(),{'blocks':[{'kind':'narrative','text':text,'source_ids':['S1'],'claim_type':claim}]})


@pytest.mark.parametrize('text,claim',[
    ('上述读数不足以支撑任何跨材料比较或与文献值的一致性判断。','inference'),
    ('单次观察不能与文献进行定量比较。','inference'),
    ('本报告没有进行文献比较。','method'),
    ('本报告不作任何文献对比。','limitation'),
    ('No literature comparison is made.','inference'),
    ('The observations cannot support agreement with published values.','inference'),
    ('Future work needs literature excerpt evidence before comparison.','recommendation'),
    ('未来需要补充文献证据后才能比较。','recommendation'),
])
def test_nonnumeric_denial_does_not_assert_an_external_literature_result(text,claim):
    assert text in render(text,claim)


@pytest.mark.parametrize('text,claim',[
    ('这些结果与文献一致。','inference'),
    ('Published studies establish a higher capacity.','inference'),
    ('No literature comparison is made, but published studies establish a higher capacity.','inference'),
    ('这些读数不能与文献比较，但文献证明该材料更优。','inference'),
    ('No difference from published values is observed.','inference'),
    ('Cannot support agreement with literature reporting a capacity of 9.','inference'),
    ('没有进行文献比较。','external'),
    ('No literature comparison is made; however the values agree.','inference'),
    ('However the values agree; no literature comparison is made.','inference'),
    ('这些观察无法与文献进行比较；然而数值一致。','inference'),
    ('数值一致；但是这些观察不能与文献比较。','inference'),
    ('这些结果与文献一致。','limitation'),
])
def test_positive_mixed_numeric_or_external_claims_still_require_actual_excerpt(text,claim):
    with pytest.raises(ValueError,match='literature comparison lacks actual supporting excerpt evidence'):
        render(text,claim)
