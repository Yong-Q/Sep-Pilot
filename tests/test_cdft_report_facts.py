import copy,hashlib,json
from pathlib import Path
import pytest
from agents import report_source_metadata as m
from agents.report_source_metadata import _publish as real_publish
from agents.report_authoring import _read
from agents.recovery import RecoveryGate

@pytest.fixture
def case(tmp_path,monkeypatch):
 root=tmp_path/'runs/u/c';root.mkdir(parents=True);original=root/'inputs';original.mkdir();work=root/'run';(work/'input_dir/m').mkdir(parents=True);(work/'input').mkdir()
 sha=lambda b:hashlib.sha256(b).hexdigest();data=b'owned input';digest=sha(data)
 for f in [original/'m.dat',work/'input/m.dat',work/'input_dir/m/input.dat']:f.write_bytes(data)
 receipt={'job_id':'123','exit_code':0,'timed_out':False,'input_sha256':digest,'wall_seconds':0.5}
 rf=work/'input_dir/m/execution.json';rf.write_text(json.dumps(receipt));batch=work/'batch_execution.json';batch.write_text(json.dumps({'job_id':'123','outcomes':{'m.dat':receipt}}));csv=work/'results.csv';csv.write_text('m,value\nm,2\n')
 quality={'status':'observed','expected_materials_verified':True,'complete_material_outputs':True,'collected_csv_sha256':sha(csv.read_bytes()),'original_input_facts':{'inputs':[{'path':'m.dat','bytes':len(data),'sha256':digest}]},'materials':{'m':{'errors':[],'technical_completion':True,'input_facts':{'temperature_K':298,'native_grid':[12,12,12],'requested_spacing_A':2.,'actual_axis_spacing_A':[3.,4.,5.],'grid_capped':True},'artifacts':{str(work/'input/m.dat'):{'sha256':digest,'size':len(data)},str(work/'input_dir/m/input.dat'):{'sha256':digest,'size':len(data)},str(rf):{'sha256':sha(rf.read_bytes()),'size':rf.stat().st_size}}}}}
 from agents import cdft_result_facts
 monkeypatch.setattr(cdft_result_facts,'inspect_cdft_batch_quality',lambda *a,**k:copy.deepcopy(quality))
 monkeypatch.setattr(m,'_publish',lambda root,role,public,binding,**kw:{'sha256':'native-meta','role':role,'untrusted_content':public,'binding':binding})
 args={'action':'submit','input_dir':str(original),'gas':'CH4'};node={'contract':{'step_id':'calc','tool':'run_cdft','arguments':args},'effective_arguments':args,'job_ids':['123'],'attempt_id':'attempt','token':'token','origin_goal_version':2}
 key=RecoveryGate().key('run_cdft',args,1,'calc',tmp_path);node['recovery_key']=key;native={'submitted':True,'job_id':'123','work_dir':str(work)}
 entry={'key':key,'tool':'run_cdft','attempt_id':'attempt','job_ids':['123'],'goal_version':2,'identity_goal_version':1,'params':args,'last_result':native};(root/'recovery_state.json').write_text(json.dumps({key:entry}))
 watch={'job_id':'123','username':'u','conv_id':'c','attempt_id':'attempt','recovery_key':key,'tool':'run_cdft','work_dir':str(work),'state':'COMPLETED'}
 summary={'step_id':'calc','tool':'run_cdft','call_id':'producer','source_sha256':'source'}
 return root,node,native,watch,summary,quality

def scheduler(case):
 root,node,native,watch,summary,_=case
 return m.scheduler_document(root,{'username':'u','conv_id':'c'},{},node,summary,native,job_lookup=lambda j:watch,read=_read,budget={'bytes':0},file_facts=[])

def test_identity_goal_binding_and_sealed_cdft_facts(case):
 root,node,native,watch,summary,_=case;s=scheduler(case);assert s and s['untrusted_content']['job_ids']==['123']
 result=m.cdft_document(root,{'username':'u','conv_id':'c'},node,summary,native,s,read=_read,budget={'bytes':0},file_facts=[])
 material=result['untrusted_content']['materials']['m'];assert material['native_grid']==[12,12,12] and material['native_wall_seconds']==0.5 and material['scientific_convergence'] is None
 assert result['untrusted_content']['producer_build_authenticated'] is False

@pytest.mark.parametrize('change',['identity','owner','attempt','input','receipt','batch','csv'])
def test_changed_owner_or_science_never_yields_facts(case,change):
 root,node,native,watch,summary,_=case
 if change=='identity':
  path=root/'recovery_state.json';d=json.loads(path.read_text());d[node['recovery_key']]['identity_goal_version']=2;path.write_text(json.dumps(d))
 if change=='owner':watch['username']='other'
 if change=='attempt':watch['attempt_id']='other'
 s=scheduler(case)
 if change in {'identity','owner','attempt'}:assert s is None;return
 assert s
 if change=='input':(root/'inputs/m.dat').write_bytes(b'changed')
 if change=='receipt':(root/'run/input_dir/m/execution.json').write_text('{}')
 if change=='batch':(root/'run/batch_execution.json').write_text('{}')
 if change=='csv':(root/'run/results.csv').write_text('changed')
 try:r=m.cdft_document(root,{'username':'u','conv_id':'c'},node,summary,native,s,read=_read,budget={'bytes':0},file_facts=[])
 except (ValueError,KeyError,TypeError):r=None
 assert r is None


def test_figure_ids_cannot_be_returned_as_literature_ids(tmp_path):
 from agents.report_authoring import author_stage_schema
 from tests.test_report_fact_pointer_diagnostics import packet
 p=packet(tmp_path);p['figures']=[{'id':'F1','source_id':'S1'}]
 schema=author_stage_schema(p);narrative=next(v for v in schema['properties']['draft']['properties']['blocks']['items']['oneOf']if v['properties']['kind']['const']=='narrative')
 refs=narrative['properties']['reference_ids'];assert refs['maxItems']==0
 p['references']=[{'id':'R1'}];schema=author_stage_schema(p)
 narrative=next(v for v in schema['properties']['draft']['properties']['blocks']['items']['oneOf']if v['properties']['kind']['const']=='narrative')
 assert narrative['properties']['reference_ids']['items']['enum']==['R1']
 assert 'F1' in narrative['properties']['source_ids']['items']['enum']


def ancestor_case(case):
 root,node,native,watch,summary,quality=case
 from agents.workflow_patch import execution_contract
 node.update(status='succeeded',execution_fingerprint='fp',result=native)
 evidence=root/'evidence';evidence.mkdir(exist_ok=True)
 call={'call_id':'producer','tool':'run_cdft','params':node['effective_arguments'],'failed':False,'result':native}
 path=evidence/'producer.json';path.write_text(json.dumps(call));cp=evidence/'cp.json';cp.write_text('checkpoint')
 node['result_ref']={'call_id':'producer','evidence_path':str(path),'checkpoint_path':str(cp)}
 record={'step_id':'calc','token':node['token'],'attempt_id':node['attempt_id'],'execution_fingerprint':'fp',
         'call_id':'producer','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'contract_sha256':m._digest(execution_contract(node['contract']))}
 figure={'contract':{'tool':'generate_result_figures','depends_on':['calc']}}
 figure_result={'source_steps':['calc'],'source_receipts':[record]}
 attest=lambda node,call,raw:{'checkpoint_verified':True,'checkpoint_path':str(cp),'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}
 return {'username':'u','conv_id':'c','nodes':{'calc':node}},figure,figure_result,attest


def test_figure_gets_only_authenticated_original_cdft_metadata(case):
 root,node,native,watch,summary,_=case;state,figure,result,attest=ancestor_case(case)
 docs=m.cdft_ancestor_documents(root,{'username':'u','conv_id':'c'},state,figure,result,job_lookup=lambda j:watch,attest=attest,read=_read,budget={'bytes':0},file_facts=[])
 assert [doc['role']for doc in docs]==['provenance','native_input_evidence']
 assert docs[1]['untrusted_content']['materials']['m']['scientific_convergence'] is None
 assert docs[1]['binding']['step_id']=='calc' and docs[1]['binding']['input_sources']
 assert docs[1]['native_ancestor_step']=='calc'

@pytest.mark.parametrize('change',['token','attempt','source_sha','contract','dependency','foreign_job','input','checkpoint'])
def test_figure_ancestor_tampering_is_never_promoted(case,change):
 root,node,native,watch,summary,_=case;state,figure,result,attest=ancestor_case(case)
 if change=='token':state['nodes']['calc']['token']='foreign'
 if change=='attempt':state['nodes']['calc']['attempt_id']='foreign'
 if change=='source_sha':result['source_receipts'][0]['source_sha256']='0'*64
 if change=='contract':state['nodes']['calc']['contract']['arguments']['gas']='N2'
 if change=='dependency':figure['contract']['depends_on']=[]
 if change=='foreign_job':watch['username']='other'
 if change=='input':(root/'inputs/m.dat').write_bytes(b'changed')
 if change=='checkpoint':attest=lambda *a:{'checkpoint_verified':False}
 try:docs=m.cdft_ancestor_documents(root,{'username':'u','conv_id':'c'},state,figure,result,job_lookup=lambda j:watch,attest=attest,read=_read,budget={'bytes':0},file_facts=[])
 except (ValueError,KeyError,TypeError,PermissionError):docs=[]
 assert not any(d['role']=='native_input_evidence' for d in docs)


def test_inherited_cdft_projection_revalidates_original_input_bytes(case,monkeypatch):
 root,node,native,watch,summary,_=case;state,figure,result,attest=ancestor_case(case)
 # This is the native immutable envelope writer, not the fixture stand-in.
 monkeypatch.setattr(m,'_publish',real_publish)
 summary['source_sha256']=hashlib.sha256(Path(node['result_ref']['evidence_path']).read_bytes()).hexdigest()
 doc=m.cdft_document(root,{'username':'u','conv_id':'c'},node,summary,native,{'sha256':'scheduler'},read=_read,budget={'bytes':0},file_facts=[])
 path=root/doc['path'];raw=path.read_bytes()
 assert m.restore_metadata(root,{'username':'u','conv_id':'c'},state,path,raw,read=_read,budget={'bytes':0},file_facts=[])['native_ancestor_step']=='calc'
 (root/'inputs/m.dat').write_bytes(b'changed')
 with pytest.raises(ValueError,match='original input changed'):
  m.restore_metadata(root,{'username':'u','conv_id':'c'},state,path,raw,read=_read,budget={'bytes':0},file_facts=[])

def test_real_cdft_native_document_projects_column_typed_table(case):
 root,node,native,watch,summary,_=case;s=scheduler(case)
 doc=m.cdft_document(root,{'username':'u','conv_id':'c'},node,summary,native,s,read=_read,budget={'bytes':0},file_facts=[])
 doc.update(path='evidence/native.json',sha256='a'*64)
 tables=[];m.metadata_tables(doc,'S1',tables)
 assert len(tables)==1 and tables[0]['format']=='native_cdft_input_observation'
 row=tables[0]['rows'][0];assert row['material']=='m' and [row['grid_x'],row['grid_y'],row['grid_z']]==[12,12,12]
 assert row['actual_spacing_x_A']==3. and row['grid_capped'] is True and row['scientific_convergence'] is None
 assert 'native_wall_seconds' not in row and 'exit_code' not in row
 assert tables[0]['native_context']['solver_build_authenticated'] is False

@pytest.mark.parametrize('change',['missing','grid','spacing','temperature','cap','wall','exit','convergence','schema'])
def test_cdft_metadata_malformed_fails_closed(case,change):
 root,node,native,watch,summary,_=case;s=scheduler(case);doc=m.cdft_document(root,{'username':'u','conv_id':'c'},node,summary,native,s,read=_read,budget={'bytes':0},file_facts=[])
 doc.update(path='evidence/native.json',sha256='a'*64);v=doc['untrusted_content'];row=v['materials']['m']
 if change=='missing':del row['native_grid']
 if change=='grid':row['native_grid']=[12,12,True]
 if change=='spacing':row['actual_axis_spacing_A']=[3.,float('nan'),5.]
 if change=='temperature':row['temperature_K']=-1
 if change=='cap':row['grid_capped']=1
 if change=='wall':row['native_wall_seconds']=-1
 if change=='exit':row['exit_code']=1
 if change=='convergence':row['scientific_convergence']=True
 if change=='schema':v['producer_build_authenticated']=True
 with pytest.raises(ValueError):m.metadata_tables(doc,'S1',[])

def test_unknown_native_input_shape_fails_closed_instead_of_keyerror():
 with pytest.raises(ValueError,match='unsupported native input metadata shape'):
  m.metadata_tables({'role':'native_input_evidence','untrusted_content':{'unexpected':{}}},'S1',[])

def test_ambiguous_string_and_cdft_metadata_is_rejected():
 with pytest.raises(ValueError,match='ambiguous native input metadata shape'):
  m.metadata_tables({'role':'native_input_evidence','untrusted_content':{'materials':{},'native_input':{}}},'S1',[])
