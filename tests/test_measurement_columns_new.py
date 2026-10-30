import importlib.util
from pathlib import Path
import pytest
from jsonschema import Draft7Validator

from agents import report_authoring as m

def packet():
    columns = ['temperature_K','grid_x','grid_y','grid_z','requested_spacing_A',
               'actual_spacing_x_A','actual_spacing_y_A','actual_spacing_z_A','grid_capped']
    table = {'id':'T2','source_id':'S1','sha256':'abc','columns':['material']+columns,
             'rows':[dict(material='framework', **{key:str(i) for i,key in enumerate(columns)})]}
    return {'tables':[table]},columns

def test_nine_realistic_columns_valid_and_source_bound():
    data,cols=packet(); block={'kind':'measurement','table_id':'T2','row_selector':{'material':'framework'},'columns':cols,'format':'.6g'}
    Draft7Validator(m.MEASUREMENT_SCHEMA).validate(block)
    rendered,bindings,source=m._measurement_block(data,block)
    assert len(bindings)==9 and source=='S1' and all(item['source_sha256']=='abc' for item in bindings)
    assert 'actual_spacing_z_A' in rendered

@pytest.mark.parametrize('count',[33,128])
def test_all_existing_columns_can_be_selected(count):
    cols=['field_'+str(i) for i in range(count)]
    table={'id':'T2','source_id':'S1','sha256':'abc','columns':['material']+cols,
           'rows':[dict(material='framework',**{key:str(i) for i,key in enumerate(cols)})]}
    block={'kind':'measurement','table_id':'T2','row_selector':{'material':'framework'},'columns':cols}
    Draft7Validator(m.MEASUREMENT_SCHEMA).validate(block)
    _,bindings,_=m._measurement_block({'tables':[table]},block)
    assert len(bindings)==count and all(item['source_sha256']=='abc' for item in bindings)

def test_unknown_column_rejected():
    data,cols=packet();block={'kind':'measurement','table_id':'T2','row_selector':{'material':'framework'},'columns':cols+['fabricated']}
    with pytest.raises(ValueError):m._measurement_block(data,block)

def test_report_can_use_one_relevant_table_without_optional_material():
    p={'title':'Report','root':'/tmp','packet_sha256':'a'*64,
       'sources':[{'id':'S1','step_id':'step','call_id':'call','source_sha256':'b'*64,'conditions':{}}],
       'tables':[{'id':'T1','source_id':'S1','columns':['value','scientific_convergence'],
                  'rows':[{'value':'1.25','scientific_convergence':None}],'row_count':1,'path':'source.csv'},
                 {'id':'T2','source_id':'S1','columns':['other'],
                  'rows':[{'other':'3'}],'row_count':1,'path':'other.csv'}],
       'references':[],'figures':[],'independent_review_required':False}
    draft={'blocks':[{'kind':'table','table_id':'T1','columns':['value']}]}
    out=m.render_validated_draft(p,draft)
    assert '1.25' in out and 'Unknown convergence remains unknown' in out
    draft['blocks'][0]['columns']=['invented']
    with pytest.raises(ValueError,match='declared source subset'):m.render_validated_draft(p,draft)

def test_explicit_figure_requirement_still_enforced():
    p={'title':'Report','root':'/tmp','packet_sha256':'a'*64,
       'sources':[{'id':'S1','step_id':'step','call_id':'call','source_sha256':'b'*64,'conditions':{}}],
       'tables':[],'references':[],'figures':[],'independent_review_required':False,'minimum_figures':1}
    draft={'blocks':[{'kind':'heading','text':'Results'}]}
    with pytest.raises(ValueError,match='additional figure'):m.render_validated_draft(p,draft)

def test_row_identity_can_use_all_relevant_fields():
    cols=['m','gas','condition','run','phase','value'];row=dict(m='A',gas='CO2',condition='298K',run='1',phase='prod',value='1.25')
    packet={'tables':[{'id':'T','source_id':'S','sha256':'abc','columns':cols,'rows':[row]}]}
    block={'kind':'measurement','table_id':'T','row_selector':{k:row[k] for k in cols[:-1]},'columns':['value']}
    Draft7Validator(m.MEASUREMENT_SCHEMA).validate(block)
    _,bindings,_=m._measurement_block(packet,block)
    assert bindings[0]['source_value']=='1.25'

def test_explicit_requirements_without_fixed_number_cap():
    from agents import presentation_continuation as presentation
    assert presentation.valid_report_requirements({'minimum_figures':17,'section_titles':['section'+str(i) for i in range(20)]})
    assert not presentation.valid_report_requirements({'minimum_figures':-1})

def test_oversize_owned_source_gets_actionable_nontruncation_feedback(tmp_path):
    source=tmp_path/'source.csv'
    source.write_bytes(b'x'*(m.MAX_FILE+1))
    with pytest.raises(ValueError,match='data sandbox.*bounded chunks.*no source data was silently truncated'):
        m._read(tmp_path,source,{'bytes':0})
    assert source.stat().st_size==m.MAX_FILE+1
