import json
from agents import parallel_workflow as m

def create(tmp_path):
 p=tmp_path/'state.json'
 node={'token':'tok','status':'running','updated_at':1,'path_manifest':{'output_files':['a'],'updated_at':1}}
 p.write_text(json.dumps({'workflows':{'w':{'status':'active','nodes':{'s':node}}},'leases':{'tok':{'resources':{}}}}))
 return p,m.WorkflowStore(p)

def test_timestamp_only_manifest_does_not_rewrite(tmp_path):
 p,s=create(tmp_path);before=p.read_bytes();inode=p.stat().st_ino
 assert s.update('w','s','tok',{'path_manifest':{'output_files':['a'],'updated_at':99}})
 assert p.read_bytes()==before and p.stat().st_ino==inode

def test_real_change_or_release_is_preserved(tmp_path):
 p,s=create(tmp_path)
 assert s.update('w','s','tok',{'path_manifest':{'output_files':['b'],'updated_at':99}})
 data=json.loads(p.read_text());assert data['workflows']['w']['nodes']['s']['path_manifest']['output_files']==['b']
 assert s.update('w','s','tok',{'path_manifest':{'output_files':['b'],'updated_at':100}},release=True)
 assert 'tok' not in json.loads(p.read_text())['leases']
 assert not s.update('w','s','wrong',{'path_manifest':{'output_files':['c']}})
 assert json.loads(p.read_text())['workflows']['w']['nodes']['s']['path_manifest']['output_files']==['b']

def test_terminal_housekeeping_not_suppressed(tmp_path):
 p,s=create(tmp_path);data=json.loads(p.read_text());w=data['workflows']['w'];w['nodes']['s']['status']='succeeded';w['branch_blockers']={'s':'old'};p.write_text(json.dumps(data))
 assert s.update('w','s','tok',{'path_manifest':{'output_files':['a'],'updated_at':3}})
 data=json.loads(p.read_text());assert data['workflows']['w']['status']=='completed';assert 's' not in data['workflows']['w']['branch_blockers']
