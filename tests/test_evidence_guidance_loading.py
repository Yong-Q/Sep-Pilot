import json,shutil
from pathlib import Path
import pytest
from agents import result_guidance
from agents.config import AgentConfig
from agents.report_authoring_session import NativeReportModelBridge,ReportStageSession

class Captured(Exception):pass

@pytest.mark.parametrize('phase',['author','review','plot','visual'])
def test_owned_bridge_actually_loads_hash_checked_evidence_guidance(tmp_path,monkeypatch,phase):
 root=tmp_path/'runs/u/c';(root/'evidence').mkdir(parents=True);owner={'username':'u','conv_id':'c'}
 pair=('author','review') if phase in {'author','review'} else ('plot','visual')
 paths={key:root/'workers'/key/'session_checkpoint.json' for key in pair}
 def factory(key):
  actor=ReportStageSession(config=AgentConfig(api_key='test',project_root=tmp_path));actor._evidence_root=root/'evidence';actor._on_checkpoint=lambda *args:None;return actor
 bridge=NativeReportModelBridge(root,owner,'a'*64,actor_factory=factory,checkpoint_paths=paths,control_valid=lambda:True,guidance_reader=result_guidance.read_result_guidance)
 bridge._completed_author_revision=1
 seen=[]
 def capture(actor,role,key,phase,payload,packet,label,guidance):
  envelope=json.loads(actor.messages[0]['content']);seen.extend(envelope['guidance']);raise Captured()
 monkeypatch.setattr(bridge,'_request_data',capture)
 packet={'owner':owner,'packet_sha256':'a'*64,'report_step':'step','report_token':'token'}
 with pytest.raises(Captured):bridge(phase,{'packet':packet})
 selected=[item for item in seen if item['path']=='result-guidance/evidence-closure.md']
 assert len(selected)==1 and 'not independent scientific validation' in selected[0]['text']
 assert 'figure ID is not a literature reference' in selected[0]['text']


def test_guidance_tampering_is_rejected(tmp_path,monkeypatch):
 source=result_guidance.bundle_root();root=tmp_path/'skills';(root/'result-guidance').mkdir(parents=True)
 shutil.copyfile(source/'result-guidance-manifest.json',root/'result-guidance-manifest.json')
 (root/'result-guidance/evidence-closure.md').write_text('tampered')
 monkeypatch.setattr(result_guidance,'bundle_root',lambda:root)
 with pytest.raises(ValueError,match='integrity|changed'):result_guidance.read_result_guidance(['result-guidance/evidence-closure.md'])


def test_real_sandbox_reads_original_and_guidance_without_host_path_or_write_access(tmp_path):
 from agents.plot_sandbox import plot_result_code
 root=tmp_path/'runs/u/c';root.mkdir(parents=True);source=root/'data.csv';source.write_text('x,y\n1,2\n2,4\n')
 script='''import json, pathlib\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\np=pathlib.Path("/inputs/T1.csv")\nassert p.read_text()=="x,y\\n1,2\\n2,4\\n"\naudit=json.loads(pathlib.Path("/inputs/audit.json").read_text())\nassert all("source_path" not in x for x in audit)\nindex=json.loads(pathlib.Path("/inputs/guidance_index.json").read_text())\nitem=next(x for x in index if x["path"]=="result-guidance/evidence-closure.md")\nassert "no fixed analysis or chart template" in pathlib.Path(item["sandbox_path"]).read_text()\ntry:\n p.write_text("tampered")\n raise AssertionError("sandbox source was writable")\nexcept OSError: pass\nassert not pathlib.Path("/home/other-user").exists()\nplt.plot([1,2],[2,4]);plt.savefig("/output/plot.png")\n'''
 out=root/'plot-attempt'
 result=plot_result_code(script,input_files=[{'path':str(source),'input_id':'T1','format':'csv'}],output_dir=str(out),owned_root=root,guidance_files=['result-guidance/evidence-closure.md'])
 assert result['ok'] is True and result['sandbox']['os_isolated'] is True
 assert source.read_text()=='x,y\n1,2\n2,4\n' and len(result['artifacts'])==1


from tests.test_figure_visual_review import case as figure_case,SCRIPT

def test_native_plot_author_passes_guidance_into_sandbox(figure_case):
 from agents.figure_authoring import author_figures
 root,packet=figure_case;requests=[]
 def model(phase,payload):
  assert 'guidance_index.json' in payload['required_output']
  return {'actor':'analyst','call_id':'plot','plot':{'script':SCRIPT,'input_ids':['J1'],'rationale':'read owned data','limitations':[]}}
 def sandbox(params):requests.append(params);raise Captured()
 with pytest.raises(Captured):author_figures(packet,root/'figures',model_call=model,sandbox_call=sandbox,control_valid=lambda:True)
 assert requests[0]['input_files'][0]['path']==str(root/'data.json')
 assert 'result-guidance/evidence-closure.md' in requests[0]['guidance_files']
