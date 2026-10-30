import pytest
from agents.report_authoring import _nonquantitative_literature_denial

@pytest.mark.parametrize('text',[
 '不将其与其他材料、实验数据或文献数值作定量比较。',
 '后续必须获取文献摘录之后才能对比。', 'Further work needs literature evidence before comparison.',
 'The evidence cannot support agreement with literature results.',
 '没有把结果同先前研究或文献数据作对照。',
 '未与材料、实验和文献结果进行比较。',
 '不对这些材料、文献结果作直接对比。',
])
def test_explicit_negation_covers_coordinated_objects(text):assert _nonquantitative_literature_denial(text)

@pytest.mark.parametrize('text',[
 'not without comparison with literature.', 'No comparison with literature and the result is better.',
 '将其与其他材料、实验数据或文献数值作定量比较。',
 '不是不将其与文献作比较。',
 '并非不将其与文献作比较。',
 '没有任何理由在本次完整研究工作当中不与文献比较。',
 '并非由于研究方法条件限制或者实验资料缺乏而没有做文献比较。',
 '不将结果优于文献这一事实作比较。', '不将其与文献证明可靠的结论作比较。',
 '研究结果优于文献并没有做文献比较。', '结果比文献更好同时没有做文献比较。',
 '文献表明该模型准确且没有做文献比较。',
 '没有做文献比较并声称结果优于文献。', '没有做文献比较并声称结果更优。',
 '不能不将其与文献作比较。', '没有必要不与文献比较。', '并非没有做文献比较。', '不只是没有做文献比较。',
 '本研究未做文献比较，结果优于它。', '本研究未做文献比较。结果优于其数据。',
 '未做文献比较，但本方法更好。',
 '不将其与文献比较，但与文献一致。',
 '不将其与文献比较。结果与文献一致。',
 '不将其与文献比较，而是与文献吻合。',
 '不将其与文献比较，也与文献作了比较。',
 '不与文献比较并与文献对照。', '不做文献比较同时与文献对照。',
 '不将其与文献中的 2.4 作比较。',
 '将其与文献比较，但不与实验比较。',
 '不将其与文献比较，并指出文献一致。',
])
def test_positive_double_negation_mixed_or_numeric_are_not_denials(text):assert not _nonquantitative_literature_denial(text)
