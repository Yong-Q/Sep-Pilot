import pytest
from agents.report_authoring import render_validated_draft,author_stage_schema

def packet():
 sha='a'*64
 return {'title':'test','report_output_path':'reports/report.md','sources':[{'id':'S1','step_id':'read','call_id':'call','source_sha256':'b'*64,'conditions':{},'native_document':{'path':'literature/doc.pdf','sha256':sha,'blocks':[{'anchor':'page-1','untrusted_content':'Original evidence text.'}]}}], 'file_facts':[{'path':'literature/doc.pdf','sha256':sha}], 'tables':[],'references':[],'figures':[]}
def draft(text='Evidence is bounded.'):
 return {'blocks':[{'kind':'excerpt','source_id':'S1','anchor':'page-1','start':0,'end':8},{'kind':'narrative','source_ids':['S1'],'claim_type':'limitation','text':text}]}

def test_owned_source_document_is_linked_without_proving_publisher():
 result=render_validated_draft(packet(),draft())
 assert '[S1 native source document](../literature/doc.pdf)' in result
 assert 'source origin and scientific claims remain unverified' in result
 schema=author_stage_schema(packet());text=schema['properties']['draft']['properties']['blocks']['items']['oneOf'][0]['properties']['text']['description']
 assert 'Never place URLs' in text

def test_narrative_url_is_still_rejected_with_usable_feedback():
 with pytest.raises(ValueError,match='remove URLs and Markdown links'):
  render_validated_draft(packet(),draft('See https://example.org/unsafe.pdf.'))

@pytest.mark.parametrize('kind',['path','hash','fact','report_path'])
def test_unowned_or_unbound_document_links_are_rejected(kind):
 p=packet()
 if kind=='path':p['sources'][0]['native_document']['path']='../../outside.pdf'
 if kind=='hash':p['sources'][0]['native_document']['sha256']='b'*64
 if kind=='fact':p['file_facts']=[]
 if kind=='report_path':p['report_output_path']='../../report.md'
 with pytest.raises(ValueError):render_validated_draft(p,draft())
