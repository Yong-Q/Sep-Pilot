import errno,hashlib,json,multiprocessing,os
from pathlib import Path
import pytest
from agents.plot_sandbox import _link_commit_data_outputs
from tests.test_data_sandbox import owned,run,SCRIPT,DECL
from tests.test_parallel_workflow import factory,node,eventually


@pytest.fixture
def force_nfs_rename_unsupported(monkeypatch):
 """Exercise the NFS fallback even when the test workspace is on local disk."""
 from agents import plot_sandbox
 real_cdll=plot_sandbox.ctypes.CDLL
 class NoRename:
  def __init__(self,library):self.library=library
  def __getattr__(self,name):
   if name=='renameat2':raise AttributeError(name)
   return getattr(self.library,name)
 def load(name,*args,**kwargs):
  library=real_cdll(name,*args,**kwargs)
  return NoRename(library) if name is None else library
 monkeypatch.setattr(plot_sandbox.ctypes,'CDLL',load)


def test_real_os_data_sandbox_commits_whole_set_on_shared_filesystem(tmp_path,force_nfs_rename_unsupported):
 root,source=owned(tmp_path);r=run(root,source,SCRIPT,DECL)
 assert r['ok'] is True and r['sandbox']['os_isolated'] is True
 assert r['publication']['protocol']=='nfs-link-commit-v1'
 marker=root/'attempt/artifacts/.publication.json';body=marker.read_bytes()
 assert hashlib.sha256(body).hexdigest()==r['publication']['commit_sha256']
 receipt=json.loads(body);expected=[{k:v[k]for k in ('name','sha256','bytes')}for v in r['artifacts']]
 assert receipt['artifacts']==expected and all(Path(a['path']).stat().st_nlink==1 for a in r['artifacts'])


def _compete(path,stage,gate,queue):
 from agents.plot_sandbox import _link_commit_data_outputs
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY);f=Path(path)/stage/'value.json';data=f.read_bytes();record={'name':'value.json','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
 try:
  gate.wait(timeout=10)
  result=_link_commit_data_outputs(fd,stage,[(record,data)]);queue.put(('ok',result['commit_sha256']))
 except FileExistsError:queue.put(('exists',None))
 except Exception as error:queue.put(('error',type(error).__name__+':'+str(error)))
 finally:os.close(fd)


def test_concurrent_writers_never_replace_existing_artifacts(tmp_path):
 for stage,value in [('first',b'{"winner":1}'),('second',b'{"winner":2}')]:
  p=tmp_path/stage;p.mkdir();(p/'value.json').write_bytes(value)
 ctx=multiprocessing.get_context('fork');barrier=ctx.Barrier(2);queue=ctx.Queue();jobs=[ctx.Process(target=_compete,args=(str(tmp_path),stage,barrier,queue))for stage in ('first','second')]
 for p in jobs:p.start()
 for p in jobs:p.join(15);assert p.exitcode==0
 results=[queue.get(timeout=2)for _ in jobs];assert sorted(x[0]for x in results)==['exists','ok']
 data=(tmp_path/'artifacts/value.json').read_bytes();marker=json.loads((tmp_path/'artifacts/.publication.json').read_text())
 assert data in (b'{"winner":1}',b'{"winner":2}') and marker['artifacts'][0]['sha256']==hashlib.sha256(data).hexdigest()


def test_failed_partial_publication_has_no_commit_and_no_overwrite(tmp_path,monkeypatch):
 stage=tmp_path/'stage';stage.mkdir();data1=b'one';data2=b'two';(stage/'a.txt').write_bytes(data1);(stage/'b.txt').write_bytes(data2)
 def record(name,data):return ({'name':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)},data)
 real=os.link
 def fault(source,destination,*args,**kwargs):
  if destination=='b.txt':raise OSError(errno.EIO,'controlled link failure')
  return real(source,destination,*args,**kwargs)
 monkeypatch.setattr(os,'link',fault);fd=os.open(tmp_path,os.O_RDONLY|os.O_DIRECTORY)
 try:
  with pytest.raises(OSError):_link_commit_data_outputs(fd,'stage',[record('a.txt',data1),record('b.txt',data2)])
 finally:os.close(fd)
 assert (tmp_path/'artifacts/a.txt').read_bytes()==data1 and not (tmp_path/'artifacts/.publication.json').exists()
 assert (stage/'a.txt').read_bytes()==data1 and (stage/'b.txt').read_bytes()==data2


def test_dag_consumer_refuses_tampered_commit_marker(factory,tmp_path,monkeypatch,force_nfs_rename_unsupported):
 from agents import registry,workspace
 root=tmp_path/'runs/tester/nfs-marker';source=root/'source.csv';out=root/'derived';artifact=out/'artifacts/summary.json'
 producer=node('analyze',tool='run_analysis_sandbox',args={'script':"import json,pathlib\npathlib.Path('/output/summary.json').write_text(json.dumps({'mean':3.0}))",'input_files':[{'path':str(source),'input_id':'T1','format':'csv'}],'output_dir':str(out),'declared_outputs':[{'name':'summary.json','format':'json','json_schema':{'type':'object','required':['mean']}}]},outputs=[str(artifact)])
 consumer=node('consume',tool='read_file',args={'path':str(artifact)},deps=['analyze']);consumer['input_bindings']={'path':{'step_id':'analyze','output_index':0}}
 rt=factory([producer,consumer],cid='nfs-marker');monkeypatch.setattr(registry,'get_config',lambda:rt.main.config);monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
 source.write_text('x,y\n1,2\n2,4\n');rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['analyze']['status']in{'succeeded','failed'},timeout=30)
 result=rt.snapshot()['nodes']['analyze'];assert result['status']=='succeeded',result
 assert result['result']['publication']['protocol']=='nfs-link-commit-v1'
 (out/'artifacts/.publication.json').write_text('{}')
 rt.tick();eventually(lambda:rt.snapshot()['nodes']['consume']['status']in{'succeeded','failed','uncertain'},timeout=30)
 assert rt.snapshot()['nodes']['consume']['status']=='failed' and not rt.snapshot()['nodes']['consume'].get('tool_returned')
