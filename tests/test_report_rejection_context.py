import pytest
from agents.report_authoring import render_validated_draft, _rejected_narrative_excerpt, _nonquantitative_literature_denial

def test_rejected_method_keeps_numeric_guard_and_locatable_context():
 packet={'title':'test','sources':[{'id':'S1','conditions':{}}],'tables':[],'references':[],'figures':[]}
 draft={'blocks':[{'kind':'narrative','claim_type':'method','source_ids':['S1'],
  'text':'Files under System_0 have SHA-256 provenance.'}]}
 with pytest.raises(ValueError) as failure:render_validated_draft(packet,draft)
 message=str(failure.value)
 assert 'unsupported numerical claim' in message
 assert 'System_0' in message and 'SHA-256' in message

def test_feedback_redacts_secrets_paths_urls_and_remains_bounded():
 text=('System_0 uses SHA-256; /home/example/private.csv https://example.org/secret '
       'person@example.org token=abc123 '+'z'*64+' '+'long '*200)
 excerpt=_rejected_narrative_excerpt(text)
 assert 'System_0' in excerpt and 'SHA-256' in excerpt
 for secret in ('/home/example','https://example.org','person@example.org','abc123','z'*64):assert secret not in excerpt
 assert len(excerpt)<=600


def test_explicit_denial_of_literature_data_does_not_permit_positive_comparisons():
 assert _nonquantitative_literature_denial('未引入任何额外模拟或文献数值。')
 assert not _nonquantitative_literature_denial('未引入文献数值，但与文献一致。')
 assert not _nonquantitative_literature_denial('采用文献数据并与文献一致。')
 assert not _nonquantitative_literature_denial('未使用文献中的 12 个结果。')
