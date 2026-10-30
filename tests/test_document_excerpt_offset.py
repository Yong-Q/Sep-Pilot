import pytest
from agents.literature_report_bridge import excerpt_block

def packet():
 return {'sources':[{'id':'S1','native_document':{'path':'doc.pdf','sha256':'a'*64,'blocks':[{'anchor':'page-1','untrusted_content':'Original page evidence.'}]}}],'tables':[]}
def excerpt(start,end):return {'kind':'excerpt','source_id':'S1','anchor':'page-1','start':start,'end':end}

def test_integral_float_offsets_preserve_exact_original_slice():
 _,binding=excerpt_block(packet(),excerpt(0.0,8.0))
 assert binding['start']==0 and type(binding['end']) is int and binding['excerpt']=='Original'

@pytest.mark.parametrize('start,end',[(False,8),(0,8.1),(0,float('nan')),(0,float('inf')),(-1,8),(0,100)])
def test_invalid_offsets_remain_blocked(start,end):
 with pytest.raises(ValueError):excerpt_block(packet(),excerpt(start,end))
