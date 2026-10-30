import hashlib
from pathlib import Path
import pytest
from agents.md_report_inputs import _attach_executed_script
from agents.report_authoring import _read

def setup(tmp_path):
 root=tmp_path/'u/c';work=root/'md';work.mkdir(parents=True);path=work/'input.lmp';path.write_text('fix npt all npt temp 300 300 100 iso 1 1 1000\n')
 fact={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'verification':'sha256_before_and_after_solver','scientific_acceptance':False}
 return root,work,path,fact

def test_current_exact_input_attached_as_untrusted_source(tmp_path):
 root,work,path,fact=setup(tmp_path);summary={'id':'S1'};facts=[]
 assert _attach_executed_script(root,work,{'execution_input':fact},summary,{},read=_read,budget={'bytes':0},file_facts=facts)==str(path)
 document=summary['text_documents'][0]
 assert document['sha256']==fact['sha256'] and document['historical_sha_verified'] is True
 assert 'fix npt' in document['preview']['text']

def test_legacy_file_is_not_inferred_as_executed_input(tmp_path):
 root,work,path,fact=setup(tmp_path);summary={'id':'S1'};facts=[]
 assert _attach_executed_script(root,work,{},summary,{},read=_read,budget={'bytes':0},file_facts=facts) is None
 assert not summary.get('text_documents') and 'does not attest' in summary['comparison_limits'][0]

@pytest.mark.parametrize('tamper',['bytes','hash','file','outside','invalid_proof','symlink'])
def test_changed_or_untrusted_script_rejected(tmp_path,tamper):
 root,work,path,fact=setup(tmp_path)
 if tamper=='bytes':fact['bytes']+=1
 if tamper=='hash':fact['sha256']='0'*64
 if tamper=='file':path.write_text('changed')
 if tamper=='outside':fact['path']=str(tmp_path/'foreign')
 if tamper=='invalid_proof':fact['verification']='guessed'
 if tamper=='symlink':
  path.rename(work/'real');path.symlink_to(work/'real')
 with pytest.raises((ValueError,OSError,PermissionError)):_attach_executed_script(root,work,{'execution_input':fact},{'id':'S1'},{},read=_read,budget={'bytes':0},file_facts=[])
