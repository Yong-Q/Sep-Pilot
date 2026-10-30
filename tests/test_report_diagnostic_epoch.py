import copy,json
from pathlib import Path
from types import SimpleNamespace
from agents.failure_recovery import FailureRecovery,diagnostic_revision,_same_approved_goal_with_continuations

def test_report_adapter_change_reopens_only_readonly_diagnosis(tmp_path,monkeypatch):
 import agents.report_source_metadata as meta
 old_report=diagnostic_revision('generate_scientific_report');old_cdft=diagnostic_revision('run_cdft')
 contract={'step_id':'report','tool':'generate_scientific_report','agent':'communicator','arguments':{},'depends_on':[],'expected_outputs':[]}
 goal={'version':1,'approved_plan_version':1,'execution_authorized':True,'approved_nodes':[contract]}
 node={'contract':contract,'status':'failed','token':'a'*32,'tool_returned':True,'dispatch_phase':'entered_tool','result':{'error':'old'},'result_ref':{'call_id':'old'}}
 state={'username':'u','conv_id':'c','status':'needs_user','plan_version':1,'goal_contract':goal,'nodes':{'report':node}}
 path=tmp_path/'workflow.json';path.write_text(json.dumps({'workflows':{'w':state}}))
 def snapshot():return json.loads(path.read_text())['workflows']['w']
 rt=SimpleNamespace(username='u',conv_id='c',workflow_id='w',root=tmp_path,futures={},store=SimpleNamespace(path=path),main=SimpleNamespace(goal_contract=SimpleNamespace(to_dict=lambda:copy.deepcopy(goal)),_current_line_id='c'),snapshot=snapshot)
 machine=FailureRecovery(rt);monkeypatch.setattr(machine,'_source',lambda state,step:{'call_id':'old','call_sha256':'x'})
 first=machine.observe('report');assert first
 assert machine.observe('report') is None
 before=snapshot()['nodes']['report']
 def changed_metadata_adapter(*args,**kwargs):return 'changed capability'
 monkeypatch.setattr(meta,'metadata_tables',changed_metadata_adapter)
 assert diagnostic_revision('generate_scientific_report')!=old_report
 assert diagnostic_revision('run_cdft')==old_cdft
 second=machine.observe('report');assert second and second['revision']>first['revision']
 after=snapshot();assert after['nodes']['report']==before
 event=after['outbox'][second['event_id']]
 assert event['payload']['failure_recovery']['stage']=='diagnosis'
 assert event['payload']['failure_recovery']['retry_granted'] is False
 assert after['goal_contract']==goal


def test_only_exact_same_version_approved_scope_with_proven_resume_is_accepted():
 saved={'version':3,'parameters':{'temperature_K':298},'approved_nodes':[],
        'execution_authorized':True,'directives':['原任务']}
 new={**copy.deepcopy(saved),'directives':['原任务','继续原任务，保留既有结果，不重复计算。']}
 assert _same_approved_goal_with_continuations(saved,new)
 changed=copy.deepcopy(new);changed['parameters']['temperature_K']=308
 assert not _same_approved_goal_with_continuations(saved,changed)
 changed=copy.deepcopy(new);changed['directives'].append('继续原任务并改用新结构。')
 assert not _same_approved_goal_with_continuations(saved,changed)
 changed=copy.deepcopy(new);changed['directives'][0]='篡改先前指令'
 assert not _same_approved_goal_with_continuations(saved,changed)
 changed=copy.deepcopy(new);changed['version']=4
 assert not _same_approved_goal_with_continuations(saved,changed)
 changed=copy.deepcopy(new);changed['execution_authorized']=1
 assert not _same_approved_goal_with_continuations(saved,changed)
