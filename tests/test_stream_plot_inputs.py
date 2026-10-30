import hashlib,json,os,shutil,time,tracemalloc
from pathlib import Path
import pytest
from agents.plot_sandbox import plot_result_code,_stream_text,MAX_STREAM_TEXT
from agents.plot_input_adapter import catalogue

SCRIPT='''import hashlib,json,pathlib\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\np=pathlib.Path("/inputs/D1.log")\nh=hashlib.sha256();n=0\nwith p.open("rb") as f:\n while True:\n  b=f.read(65536)\n  if not b: break\n  h.update(b);n+=len(b)\na=json.loads(pathlib.Path("/inputs/audit.json").read_text())[0]\nassert h.hexdigest()==a["sha256"] and n==a["bytes"] and a["inspection_scope"].startswith("bounded preview")\nassert "source_path" not in a\ni=json.loads(pathlib.Path("/inputs/guidance_index.json").read_text())\nassert any(x["path"]=="result-guidance/evidence-closure.md" and pathlib.Path(x["sandbox_path"]).is_file() for x in i)\ntry:\n p.write_text("bad")\n raise AssertionError("read-only input writable")\nexcept OSError:pass\nassert not pathlib.Path("/home/other-user").exists()\nplt.plot([1,2],[1,2]);plt.savefig("/output/result.png")\n'''

@pytest.fixture
def large(tmp_path):
 root=tmp_path/'runs/u/c';root.mkdir(parents=True);source=root/'sample.data'
 with source.open('wb') as f:
  for _ in range(220):f.write(b'native GCMC observation 42\n'*2500)
 assert source.stat().st_size>8*1024**2
 return root,source

def test_real_os_sandbox_reads_full_large_original_and_guidance(large,record_property):
 root,source=large;expected=hashlib.sha256(source.read_bytes()).hexdigest()
 tracemalloc.start();start=time.monotonic();fact=_stream_text(root,source,MAX_STREAM_TEXT);elapsed=time.monotonic()-start;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
 assert fact['sha256']==expected and fact['bytes']==source.stat().st_size and peak<4*1024**2
 record_property('input_bytes',fact['bytes']);record_property('stream_seconds',elapsed);record_property('python_peak_bytes',peak)
 result=plot_result_code(SCRIPT,input_files=[{'path':str(source),'input_id':'D1','format':'text','expected_sha256':expected}],output_dir=str(root/'out'),owned_root=root,guidance_files=['result-guidance/evidence-closure.md'])
 assert result['ok'] is True and result['sandbox']['os_isolated'] is True
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected

def test_wrong_hash_and_foreign_or_linked_input_are_rejected(large,tmp_path):
 root,source=large
 with pytest.raises(ValueError,match='authenticated input SHA'):
  plot_result_code(SCRIPT,input_files=[{'path':str(source),'input_id':'D1','format':'text','expected_sha256':'0'*64}],output_dir=str(root/'wrong'),owned_root=root)
 foreign=tmp_path/'outside.data';foreign.write_text('other user')
 with pytest.raises((ValueError,PermissionError,OSError)):_stream_text(root,foreign,MAX_STREAM_TEXT)
 link=root/'link.data';link.symlink_to(source)
 with pytest.raises((ValueError,PermissionError,OSError)):_stream_text(root,link,MAX_STREAM_TEXT)
 hardlink=root/'hardlink.data';os.link(foreign,hardlink)
 with pytest.raises(ValueError,match='bounded'):_stream_text(root,hardlink,MAX_STREAM_TEXT)

def test_oversized_sparse_source_fails_before_read(large):
 root,source=large;oversize=root/'oversize.data'
 with oversize.open('wb')as f:f.truncate(MAX_STREAM_TEXT+1)
 with pytest.raises(ValueError,match='bounded'):_stream_text(root,oversize,MAX_STREAM_TEXT)

def test_only_bound_relative_raw_sources_enter_catalogue():
 packet={'tables':[],'file_facts':[{'path':'native/output.data','sha256':'a'*64,'size':9000000}],
  'sources':[{'id':'S1','structured_data':[{'id':'J1','path':'view.json','sha256':'b'*64,'untrusted_content':{},
  'raw_input_sources':[{'path':'native/output.data','sha256':'a'*64,'bytes':9000000,'format':'text'},
   {'path':'/outside.data','sha256':'a'*64,'bytes':9000000,'format':'text'},
   {'path':'native/fake.data','sha256':'c'*64,'bytes':5,'format':'text'}]}]}]}
 options=catalogue(packet);raw=[x for x in options if x['id'].startswith('D')]
 assert len(raw)==1 and raw[0]['path']=='native/output.data' and raw[0]['sandbox_path']=='/inputs/D1.log'

def test_visual_review_receives_only_explicitly_truncated_raw_source(large):
 from agents.figure_visual_review import source_contract
 root,source=large;digest=hashlib.sha256(source.read_bytes()).hexdigest()
 item={'id':'D1','format':'text','path':source.name,'sha256':digest,'bytes':source.stat().st_size}
 packet={'root':str(root),'tables':[],'sources':[]}
 value=source_contract(packet,[item],['D1'])
 assert value['sources'][0]['truncated'] is True and len(value['sources'][0]['source_text'])<=6000
 assert 'not the entire source' in value['evidence_rules']['truncated_sources']
 source.write_text('changed')
 with pytest.raises(ValueError,match='changed'):source_contract(packet,[item],['D1'])
