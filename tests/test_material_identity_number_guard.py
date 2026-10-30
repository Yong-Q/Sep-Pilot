import pytest
from agents.report_authoring import render_validated_draft

def packet(path='/owner/MOF-5_pacman.cif'):
 return {'title':'test','sources':[{'id':'S1','step_id':'charge','call_id':'call','source_sha256':'a'*64,'conditions':{},'structured_data':[{'id':'J1','untrusted_content':{'cif_path':path}}]}],'tables':[],'references':[],'figures':[]}
def draft(text):return {'blocks':[{'kind':'narrative','source_ids':['S1'],'claim_type':'method','text':text}]}

def test_exact_source_material_identifier_is_not_a_measurement():
 result=render_validated_draft(packet(),draft('MOF-5 骨架的电荷检查尚不足以说明吸附收敛。'))
 assert 'MOF-5' in result

@pytest.mark.parametrize('text',[
 'MOF-5 的吸附量为 5 mol/kg。',
 'MOF-50 骨架的电荷检查尚不足以说明吸附收敛。',
 'MOF-5-new 骨架的电荷检查尚不足以说明吸附收敛。',
 'MOF-5 骨架在 5 K 运行。',
])
def test_same_value_or_changed_identifier_is_not_authorized(text):
 with pytest.raises(ValueError,match='unsupported numerical claim'):
  render_validated_draft(packet(),draft(text))

def test_identifier_from_another_source_does_not_authorize_this_source():
 p=packet('/owner/ZIF-8_pacman.cif')
 with pytest.raises(ValueError,match='unsupported numerical claim'):
  render_validated_draft(p,draft('MOF-5 的结构尚未核验。'))

def test_unknown_filename_suffix_is_not_assumed_to_be_same_material():
 p=packet('/owner/MOF-5_unrelated.cif')
 with pytest.raises(ValueError,match='unsupported numerical claim'):
  render_validated_draft(p,draft('MOF-5 的结构尚未核验。'))
