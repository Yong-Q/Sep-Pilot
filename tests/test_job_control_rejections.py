import copy,json
from types import SimpleNamespace
import pytest
from agents import job_control
from tests.test_job_retarget_feedback import assessment


def setup_control(tmp_path,monkeypatch,mutation=None,code=1,error='Resource temporarily unavailable for job 123\n'):
    from agents.job_watch import JobWatch
    watch=JobWatch(str(tmp_path/'watch.json'));watch.register('123',username='a',conv_id='c',tool='run_cdft')
    monkeypatch.setattr('agents.job_watch.get_watch',lambda:watch)
    state=assessment();state['effective_constraints']['script_has_memory_directive']=True
    monkeypatch.setattr('agents.resource_review.assess_job_resources',lambda _:copy.deepcopy(state))
    calls=[]
    def update(argv,**kw):
        calls.append(argv)
        if mutation:mutation(state['effective_constraints'])
        return SimpleNamespace(returncode=code,stderr=error)
    monkeypatch.setattr(job_control.subprocess,'run',update)
    def act():return job_control.retarget_queued_job('123','good','compute','a','c',tmp_path,memory_mb=16384)
    return act,state,calls


def test_known_scheduler_rejection_with_unchanged_pending_budget_is_retryable(tmp_path,monkeypatch):
    act,state,calls=setup_control(tmp_path,monkeypatch)
    result=act()
    assert result['status']=='verified_rejected' and result['retryable'] and not result['ok']
    assert not result['requires_user'] and result['not_resubmitted']
    assert json.load(open(result['journal_path']))['rejection_proof']['unchanged_fields']['Partition']=='compute'
    assert len(calls)==3
    act();assert len(calls)==3


@pytest.mark.parametrize('kind',['partial_partition','partial_ram','running','transport','signal','success_no_change'])
def test_ambiguous_partial_or_transport_outcomes_remain_uncertain(tmp_path,monkeypatch,kind):
    mutation=None;code=1;error='Resource temporarily unavailable for job 123\n'
    if kind=='partial_partition':mutation=lambda c:c.update(Partition='other')
    if kind=='partial_ram':mutation=lambda c:c.update(ReqTRES='cpu=1,mem=8G,node=1')
    if kind=='running':mutation=lambda c:c.update(JobState='RUNNING')
    if kind=='transport':error='Socket timed out on send/recv operation'
    if kind=='signal':code=-9
    if kind=='success_no_change':code=0;error=''
    act,state,calls=setup_control(tmp_path,monkeypatch,mutation,code,error)
    result=act();assert result['status']=='uncertain'
    act();assert len(calls)==1


def test_legacy_uncertain_rejection_is_reconciled_only_after_fresh_same_job_readback(tmp_path,monkeypatch):
    act,state,calls=setup_control(tmp_path,monkeypatch)
    root=tmp_path/'runs/a/c/job_control';root.mkdir(parents=True)
    before=copy.deepcopy(state['effective_constraints'])
    old={'job_id':'123','update_id':'old','status':'uncertain','exit_code':1,
         'scheduler_error':'Resource temporarily unavailable for job 123\n',
         'old_constraints':before,'after':copy.deepcopy(before),
         'requested':{'nodelist':'good','partition':'compute','memory_mb':16384},'verified':False}
    path=root/'123_old.json';path.write_text(json.dumps(old))
    result=act();reconciled=json.loads(path.read_text())
    assert len(calls)==2 and result['status']=='verified_rejected'
    assert reconciled['status']=='verified_rejected'
    assert reconciled['old_constraints']==before and reconciled['after']==before
    assert reconciled['reconciled_from_status']=='uncertain'
    assert reconciled['rejection_proof']['proof_sha256']


def test_missing_legacy_control_field_cannot_be_reconciled(tmp_path,monkeypatch):
    act,state,calls=setup_control(tmp_path,monkeypatch)
    root=tmp_path/'runs/a/c/job_control';root.mkdir(parents=True)
    before=copy.deepcopy(state['effective_constraints']);before.pop('TimeLimit')
    (root/'123_old.json').write_text(json.dumps({'job_id':'123','status':'uncertain','exit_code':1,
        'scheduler_error':'Resource temporarily unavailable for job 123\n','old_constraints':before,'after':before}))
    assert act()['blocked'] and not calls


def test_transient_rejection_stops_at_first_verified_success(tmp_path,monkeypatch):
    act,state,calls=setup_control(tmp_path,monkeypatch)
    def run(argv,**kw):
        calls.append(argv)
        if len(calls)==2:
            state['effective_constraints'].update(ReqNodeList='good',Partition='compute')
            return SimpleNamespace(returncode=0,stderr='')
        return SimpleNamespace(returncode=1,stderr='Resource temporarily unavailable for job 123\n')
    monkeypatch.setattr(job_control.subprocess,'run',run)
    result=act()
    assert result['ok'] and result['status']=='verified' and len(calls)==2


def test_each_retry_rechecks_fresh_target_facts(tmp_path,monkeypatch):
    act,state,calls=setup_control(tmp_path,monkeypatch)
    def run(argv,**kw):
        calls.append(argv);state['node_inventory']['nodes']['good']['states']=['mix-']
        return SimpleNamespace(returncode=1,stderr='Resource temporarily unavailable for job 123\n')
    monkeypatch.setattr(job_control.subprocess,'run',run)
    result=act()
    assert result['blocked'] and result['status']=='infrastructure_retry' and len(calls)==1


def test_cancellation_intent_between_rejections_stops_control_retry(tmp_path,monkeypatch):
    act,state,calls=setup_control(tmp_path,monkeypatch)
    from agents.job_watch import get_watch
    def run(argv,**kw):
        calls.append(argv);watch=get_watch();watch._jobs['123']['cancel_request']={'status':'intent'};watch._save()
        return SimpleNamespace(returncode=1,stderr='Resource temporarily unavailable for job 123\n')
    monkeypatch.setattr(job_control.subprocess,'run',run)
    result=act()
    assert result['blocked'] and 'cancellation' in result['reason'] and len(calls)==1


def test_foreign_or_different_job_error_never_reconciles(tmp_path,monkeypatch):
    act,state,calls=setup_control(tmp_path,monkeypatch,error='Resource temporarily unavailable for job 999\n')
    result=act()
    assert result['status']=='uncertain' and len(calls)==1


def test_budget_change_between_retries_is_not_silently_adopted(tmp_path,monkeypatch):
    act,state,calls=setup_control(tmp_path,monkeypatch)
    reads=0
    def assess(_):
        nonlocal reads
        reads+=1
        if reads==3:state['effective_constraints']['NumCPUs']='2'
        return copy.deepcopy(state)
    monkeypatch.setattr('agents.resource_review.assess_job_resources',assess)
    result=act()
    assert result['status']=='uncertain' and len(calls)==1
