import pytest
from agents.report_authoring import render_validated_draft

def packet():return {'title':'test','sources':[{'id':'S1','step_id':'read','call_id':'call','source_sha256':'a'*64,'conditions':{}}],'tables':[],'references':[],'figures':[]}
def draft(text):return {'blocks':[{'kind':'narrative','source_ids':['S1'],'claim_type':'limitation','text':text}]}

@pytest.mark.parametrize('text',[
 'Original source is /home/person/private/file.pdf.',
 'Owned copy is .../runs/user/case/literature/article.pdf.',
 'Document path literature/article.pdf is present.',
 'PDF is file:///tmp/source.pdf.',
 '相对文件路径 ../docs/article.pdf。',
])
def test_file_paths_are_rejected_before_semantic_comparison(text):
 with pytest.raises(ValueError,match='local path in report prose'):
  render_validated_draft(packet(),draft(text))

def test_ordinary_scientific_units_are_not_treated_as_local_paths():
 result=render_validated_draft(packet(),draft('The units are mol/kg and molecules/unit cell; convergence remains unknown.'))
 assert 'mol/kg' in result

def test_positive_literature_claim_still_requires_excerpt():
 with pytest.raises(ValueError,match='literature comparison lacks'):
  render_validated_draft(packet(),draft('该结果与文献一致。'))

def test_unverified_url_still_rejected():
 with pytest.raises(ValueError,match='narrative links must'):
  render_validated_draft(packet(),draft('See https://example.org/source.pdf.'))
