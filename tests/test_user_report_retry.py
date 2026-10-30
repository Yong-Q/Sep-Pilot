import copy
from concurrent.futures import ThreadPoolExecutor
import pytest
from tests.test_report_bounded_retry import candidate
from agents.report_bounded_retry import capture_report_retry,bounded_retry_history
from agents.state_io import json_transaction


def auth(runtime,goal):
    n=runtime.snapshot()['nodes']['report']
    return {'kind':'authenticated_user_report_retry','owner':{'username':'owner','conv_id':'case'},
        'step_id':'report','plan_version':goal['approved_plan_version'],'failure_call':n['result_ref']['call_id'],
        'failure_token':n['token'],'request_id':'request_1234567890','confirmed':True}


def prepare(candidate):
    runtime,goal,_,_=candidate
    with json_transaction(runtime.store.path)as data:
        node=data['workflows']['workflow']['nodes']['report']
        node['report_retry_epochs']=[{'capability_revision':'a'*64},{'capability_revision':'b'*64}]
        node['contract']['resources']={}
    return runtime,goal


def capture(runtime,authorization):
    return capture_report_retry(runtime,runtime.snapshot(),'report',control_valid=lambda:True,
                                return_packet=True,user_authorization=authorization)


def commit(runtime,goal,authorization,proof,packet):
    def revalidate(state):return capture_report_retry(runtime,state,'report',control_valid=lambda:True,
        prepared_packet=packet,user_authorization=authorization)
    return runtime.store.requeue_bounded_report('workflow','report',authorization['failure_token'],goal,
        authorization['owner'],proof,revalidate,control_valid=lambda:True)


def test_user_retry_keeps_automatic_budget_and_is_single_use(candidate):
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization)
    assert proof['kind']=='authenticated_user_report_reauthor' and len(bounded_retry_history(runtime.snapshot()['nodes']['report']))==2
    result=commit(runtime,goal,authorization,proof,packet);s=runtime.snapshot();assert result['no_scientific_resubmission']
    assert s['nodes']['report']['status']=='running' and len(bounded_retry_history(s['nodes']['report']))==2
    assert len(__import__('json').loads(runtime.store.path.read_text())['user_report_retry_ledger'])==1 and s['nodes']['source']['status']=='succeeded'
    with pytest.raises(ValueError):commit(runtime,goal,authorization,proof,packet)


@pytest.mark.parametrize('field,value',[('owner',{'username':'other','conv_id':'case'}),('step_id','other'),
    ('plan_version',2),('failure_call','foreign'),('failure_token','f'*32),('request_id','short'),('confirmed',False)])
def test_wrong_scope_or_ambiguous_authorization_is_rejected(candidate,field,value):
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);authorization[field]=value
    with pytest.raises(ValueError):capture(runtime,authorization)


def test_concurrent_replay_one_commit(candidate):
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization)
    def run():
        from agents.watch_context import set_context
        set_context('owner','case','lead-orchestrator','case')
        try:commit(runtime,goal,authorization,proof,packet);return 'ok'
        except ValueError:return 'rejected'
    with ThreadPoolExecutor(max_workers=2)as pool:results=list(pool.map(lambda _:run(),range(2)))
    assert sorted(results)==['ok','rejected'] and len(__import__('json').loads(runtime.store.path.read_text())['user_report_retry_ledger'])==1


def test_source_or_output_tamper_rejected_before_commit(candidate):
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization)
    (runtime.root/'report.md').write_text('existing result')
    with pytest.raises(ValueError):commit(runtime,goal,authorization,proof,packet)
    assert runtime.snapshot()['nodes']['report']['status']=='failed'


def test_authenticated_api_confirmation_and_replay(candidate,monkeypatch,tmp_path):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import api
    from agents.workflow_user_control import bind_api_workflow_control
    runtime,goal=prepare(candidate);actor=runtime.main;actor._lifecycle_store=None
    conv=SimpleNamespace(username='owner',conv_id='case',session=actor,is_processing=False,interrupt_requested=False,
                         interrupt_message='',current_uid='original',_workflow_user_control_epoch=0)
    bind_api_workflow_control(conv,actor)
    monkeypatch.setattr(api.session_manager,'get_user',lambda username:SimpleNamespace(get_conversation=lambda cid:conv if cid=='case' else None))
    monkeypatch.setattr(api,'_ensure_workflow_runtime',lambda c:runtime)
    api.app.dependency_overrides[api._get_user]=lambda:'owner'
    request={'plan_version':1,'expected_failure_call_id':'failedcall','expected_failure_token':'b'*32,
             'request_id':'request_1234567890','confirm_same_report_only':True}
    try:
        client=TestClient(api.app)
        bad=dict(request,confirm_same_report_only=False)
        assert client.post('/api/conversations/case/workflow/nodes/report/retry-report-once',json=bad).status_code==400
        assert runtime.snapshot()['nodes']['report']['status']=='failed'
        response=client.post('/api/conversations/case/workflow/nodes/report/retry-report-once',json=request)
        assert response.status_code==200,response.text
        assert response.json()['no_scientific_resubmission'] is True
        assert client.post('/api/conversations/case/workflow/nodes/report/retry-report-once',json=request).status_code==409
        assert len(__import__('json').loads(runtime.store.path.read_text())['user_report_retry_ledger'])==1
        assert runtime.snapshot()['nodes']['source']['status']=='succeeded'
        # A disconnected user can retrieve the committed receipt without
        # issuing a second authorization or revealing private paths.
        path=tmp_path/'data/state/parallel_workflows.json';path.parent.mkdir(parents=True);path.write_bytes(runtime.store.path.read_bytes())
        monkeypatch.setattr(api,'__file__',str(tmp_path/'api.py'))
        receipt=client.get('/api/conversations/case/workflow/report-retry-receipts/'+request['request_id'])
        assert receipt.status_code==200 and receipt.json()['status']=='claimed'
        assert str(runtime.root) not in receipt.text
        assert client.get('/api/conversations/case/workflow/report-retry-receipts/other_request_12345').status_code==404
    finally:api.app.dependency_overrides.pop(api._get_user,None)


def test_targeted_claim_does_not_tick_or_dispatch_other_work(candidate):
    import threading
    from agents.parallel_workflow import ParallelWorkflow
    runtime,goal=prepare(candidate)
    with json_transaction(runtime.store.path) as data:
        state=data['workflows']['workflow'];state['nodes']['report']['contract']['resources']={}
        state['nodes']['other']={'status':'pending','contract':{'step_id':'other','tool':'run_cdft','agent':'operator',
            'arguments':{},'depends_on':[],'expected_outputs':[],'resources':{}}}
    authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization);receipt=commit(runtime,goal,authorization,proof,packet)
    scheduled=[]
    class Pool:
        def submit(self,func,step,ticket):
            scheduled.append((step,ticket['contract']['tool']));return type('Future',(),{'done':lambda self:False})()
    rt=ParallelWorkflow.__new__(ParallelWorkflow);rt.lock=threading.Lock();rt.store=runtime.store;rt.workflow_id='workflow'
    rt.username='owner';rt.conv_id='case';rt.main=runtime.main;rt.root=runtime.root;rt.job_watch=None;rt.max_workers=4;rt.futures={};rt.pool=Pool()
    result=rt.dispatch_user_report_only('report',control_valid=lambda:True,proof=proof,packet=packet,ticket=receipt['user_claim_ticket'])
    assert result=='report_worker_claimed' and scheduled==[('report','generate_scientific_report')]
    state=rt.snapshot();assert state['nodes']['report']['status']=='running' and state['nodes']['other']['status']=='pending'


def test_pause_revision_changed_even_if_pause_cleared_rejects_commit(candidate):
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization)
    with json_transaction(runtime.store.path)as data:data['workflows']['workflow']['user_pause_revision']=1
    with pytest.raises(ValueError):commit(runtime,goal,authorization,proof,packet)
    assert runtime.snapshot()['nodes']['report']['status']=='failed'


def test_same_output_or_source_cannot_reset_limit_with_new_workflow(candidate):
    import json
    runtime,goal=prepare(candidate);before=runtime.snapshot();authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization)
    commit(runtime,goal,authorization,proof,packet)
    # A fresh workflow and renamed step cannot erase the top-level owner ledger.
    new=copy.deepcopy(before);node=new['nodes'].pop('report');node['contract']['step_id']='renamed';new['nodes']['renamed']=node
    altered=copy.deepcopy(proof);altered['step_id']='renamed';altered['user_authorization']['step_id']='renamed';altered['user_authorization']['request_id']='another_request_1234'
    altered['contract_sha256']='a'*64  # Even if descriptive contract metadata changes, output/lineage still match.
    with json_transaction(runtime.store.path)as data:data['workflows']['new-workflow']=new
    with pytest.raises(ValueError,match='already used'):
        runtime.store.requeue_bounded_report('new-workflow','renamed',node['token'],goal,authorization['owner'],altered,lambda state:altered,control_valid=lambda:True)
    assert json.loads(runtime.store.path.read_text())['workflows']['new-workflow']['nodes']['renamed']['status']=='failed'


def test_same_source_distinct_preapproved_report_can_use_own_allowance(candidate):
    from agents.report_bounded_retry import _digest
    from agents.workflow_patch import execution_contract
    runtime,goal=prepare(candidate);original=runtime.snapshot();second=copy.deepcopy(original['nodes']['report'])
    second['contract']['step_id']='report_other';second['contract']['arguments']['title']='Another report'
    second['contract']['arguments']['output_path']=str(runtime.root/'other.md');second['contract']['expected_outputs']=[str(runtime.root/'other.md')]
    goal['approved_nodes'].append(copy.deepcopy(second['contract']))
    with json_transaction(runtime.store.path)as data:
        state=data['workflows']['workflow'];state['goal_contract']=copy.deepcopy(goal);state['nodes']['report_other']=second
    authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization)
    assert _digest(execution_contract(second['contract'])) in proof['approved_sibling_contract_sha256']
    commit(runtime,goal,authorization,proof,packet)
    secondproof=copy.deepcopy(proof);secondproof['step_id']='report_other';secondproof['output_identity_sha256']='2'*64
    secondproof['contract_sha256']=_digest(execution_contract(second['contract']))
    secondproof['user_authorization']['step_id']='report_other';secondproof['user_authorization']['request_id']='different_request_1234'
    result=runtime.store.requeue_bounded_report('workflow','report_other',second['token'],goal,authorization['owner'],secondproof,lambda s:secondproof,control_valid=lambda:True)
    assert result['no_scientific_resubmission'] is True


def test_title_and_output_churn_cannot_reset_same_source_allowance(candidate):
    runtime,goal=prepare(candidate);before=runtime.snapshot();authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization)
    commit(runtime,goal,authorization,proof,packet)
    new=copy.deepcopy(before);node=new['nodes'].pop('report');node['contract']['step_id']='renamed'
    node['contract']['arguments']['title']='Controlled report revised';node['contract']['arguments']['output_path']=str(runtime.root/'revised.md')
    node['contract']['expected_outputs']=[str(runtime.root/'revised.md')];new['nodes']['renamed']=node
    altered=copy.deepcopy(proof);altered['step_id']='renamed';altered['output_identity_sha256']='3'*64
    altered['user_authorization']['step_id']='renamed';altered['user_authorization']['request_id']='yet_another_request_1234'
    with json_transaction(runtime.store.path)as data:data['workflows']['replacement']=new
    with pytest.raises(ValueError,match='already used'):
        runtime.store.requeue_bounded_report('replacement','renamed',node['token'],goal,authorization['owner'],altered,lambda state:altered,control_valid=lambda:True)


@pytest.mark.parametrize('tamper',['source','output','goal','control'])
def test_targeted_dispatch_rechecks_source_and_control(candidate,tamper):
    import threading
    from agents.parallel_workflow import ParallelWorkflow
    runtime,goal=prepare(candidate)
    with json_transaction(runtime.store.path)as data:data['workflows']['workflow']['nodes']['report']['contract']['resources']={}
    authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization);receipt=commit(runtime,goal,authorization,proof,packet)
    class Pool:
        def submit(self,*args):pytest.fail('stale report was dispatched')
    rt=ParallelWorkflow.__new__(ParallelWorkflow);rt.lock=threading.Lock();rt.store=runtime.store;rt.workflow_id='workflow'
    rt.username='owner';rt.conv_id='case';rt.main=runtime.main;rt.root=runtime.root;rt.job_watch=None;rt.max_workers=4;rt.futures={};rt.pool=Pool()
    if tamper=='source':(runtime.root/'source.csv').write_text('material,value\nfixture,999\n')
    if tamper=='output':(runtime.root/'report.md').write_text('existing result')
    if tamper=='goal':goal['version']=2
    assert rt.dispatch_user_report_only('report',control_valid=lambda:tamper!='control',proof=proof,packet=packet,ticket=receipt['user_claim_ticket'])=='not_started_control_or_source_changed'
    assert rt.snapshot()['nodes']['report']['status']=='failed'


@pytest.mark.parametrize('tamper',['running_future','missing_settlement','wrong_settlement_token','wrong_settlement_call'])
def test_user_retry_rejects_live_or_unsettled_failure(candidate,tamper):
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal)
    if tamper=='running_future':runtime.futures['report']=type('Future',(),{'done':lambda self:False})()
    else:
        with json_transaction(runtime.store.path)as data:
            event=data['workflows']['workflow']['outbox']['failed-event']
            if tamper=='missing_settlement':event['kind']='worker_started'
            if tamper=='wrong_settlement_token':event['payload']['correlation_token']='f'*32
            if tamper=='wrong_settlement_call':event['payload']['evidence_call']['call_id']='other'
    with pytest.raises(ValueError):capture(runtime,authorization)
    assert runtime.snapshot()['nodes']['report']['status']=='failed'


def test_capacity_failure_does_not_consume_user_authorization(candidate):
    import json
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization)
    runtime.store.max_running=0
    with pytest.raises(ValueError,match='capacity or dependencies'):commit(runtime,goal,authorization,proof,packet)
    data=json.loads(runtime.store.path.read_text());assert not data.get('user_report_retry_ledger')
    assert data['workflows']['workflow']['nodes']['report']['status']=='failed'


def test_executor_submit_failure_retains_claim_without_requeue(candidate):
    import threading
    from agents.parallel_workflow import ParallelWorkflow
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization);receipt=commit(runtime,goal,authorization,proof,packet)
    class Pool:
        def submit(self,*args):raise RuntimeError('pool unavailable')
    rt=ParallelWorkflow.__new__(ParallelWorkflow);rt.lock=threading.Lock();rt.store=runtime.store;rt.workflow_id='workflow'
    rt.username='owner';rt.conv_id='case';rt.main=runtime.main;rt.root=runtime.root;rt.job_watch=None;rt.max_workers=4;rt.futures={};rt.pool=Pool()
    assert rt.dispatch_user_report_only('report',control_valid=lambda:True,proof=proof,packet=packet,ticket=receipt['user_claim_ticket'])=='claimed_requires_reconciliation'
    assert rt.snapshot()['nodes']['report']['status']=='uncertain'
    with pytest.raises(ValueError):commit(runtime,goal,authorization,proof,packet)


def test_control_revoked_inside_claim_rolls_back_entire_transaction(candidate):
    import json
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization);calls=[]
    def control():calls.append(True);return len(calls)==1
    def revalidate(state):return capture_report_retry(runtime,state,'report',control_valid=lambda:True,prepared_packet=packet,user_authorization=authorization)
    with pytest.raises(ValueError,match='capacity or dependencies'):
        runtime.store.requeue_bounded_report('workflow','report',authorization['failure_token'],goal,authorization['owner'],proof,revalidate,control_valid=control)
    data=json.loads(runtime.store.path.read_text());assert not data.get('user_report_retry_ledger')
    assert data['workflows']['workflow']['nodes']['report']['status']=='failed' and not data.get('leases')


def test_background_claim_competing_in_another_process_never_duplicates(candidate):
    import multiprocessing,json
    from agents.parallel_workflow import WorkflowStore
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization)
    context=multiprocessing.get_context('fork');start=context.Event();queue=context.Queue()
    def background(path):
        start.wait();queue.put(WorkflowStore(path).claim('workflow','report'))
    worker=context.Process(target=background,args=(runtime.store.path,));worker.start();start.set()
    receipt=commit(runtime,goal,authorization,proof,packet);worker.join(10)
    assert worker.exitcode==0 and queue.get(timeout=2) is None
    data=json.loads(runtime.store.path.read_text());node=data['workflows']['workflow']['nodes']['report']
    assert node['status']=='running' and node['token']==receipt['user_claim_ticket']['token']
    assert list(data['leases'])==[node['token']]


@pytest.mark.parametrize('where',['snapshot','goal','source'])
def test_pre_submit_exception_settles_without_permanent_lease(candidate,monkeypatch,where):
    import threading,json
    from agents.parallel_workflow import ParallelWorkflow
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization);receipt=commit(runtime,goal,authorization,proof,packet)
    class Pool:
        def submit(self,*args):pytest.fail('tool submitted after preflight failure')
    rt=ParallelWorkflow.__new__(ParallelWorkflow);rt.lock=threading.Lock();rt.store=runtime.store;rt.workflow_id='workflow';rt.username='owner';rt.conv_id='case';rt.main=runtime.main;rt.root=runtime.root;rt.job_watch=None;rt.max_workers=4;rt.futures={};rt.pool=Pool()
    if where=='snapshot':monkeypatch.setattr(rt,'snapshot',lambda:(_ for _ in ()).throw(OSError('unavailable')))
    if where=='goal':monkeypatch.setattr(rt.main.goal_contract,'to_dict',lambda:(_ for _ in ()).throw(ValueError('unavailable')))
    if where=='source':
        import agents.report_authoring as report
        monkeypatch.setattr(report,'sources_unchanged',lambda *args:(_ for _ in ()).throw(OSError('unavailable')))
    assert rt.dispatch_user_report_only('report',control_valid=lambda:True,proof=proof,packet=packet,ticket=receipt['user_claim_ticket'])=='not_started_control_or_source_changed'
    data=json.loads(runtime.store.path.read_text());assert data['workflows']['workflow']['nodes']['report']['status']=='failed' and not data['leases']


def test_unstarted_reconciliation_preserves_user_pause(candidate):
    import threading,json
    from agents.parallel_workflow import ParallelWorkflow
    runtime,goal=prepare(candidate);authorization=auth(runtime,goal);proof,packet=capture(runtime,authorization);receipt=commit(runtime,goal,authorization,proof,packet)
    with json_transaction(runtime.store.path)as data:
        state=data['workflows']['workflow'];state.update(user_paused=True,user_pause_revision=7,pause_kind='user_change',pause_reason='user paused')
    rt=ParallelWorkflow.__new__(ParallelWorkflow);rt.lock=threading.Lock();rt.store=runtime.store;rt.workflow_id='workflow';rt.username='owner';rt.conv_id='case';rt.main=runtime.main;rt.root=runtime.root;rt.job_watch=None;rt.max_workers=4;rt.futures={}
    assert rt.dispatch_user_report_only('report',control_valid=lambda:False,proof=proof,packet=packet,ticket=receipt['user_claim_ticket'])=='not_started_control_or_source_changed'
    data=json.loads(runtime.store.path.read_text());state=data['workflows']['workflow']
    assert state['user_paused'] is True and state['user_pause_revision']==7 and state['pause_kind']=='user_change' and state['pause_reason']=='user paused' and not data['leases']


from tests.test_parallel_workflow import factory,node

def test_late_real_worker_never_releases_uncertain_claim(factory,monkeypatch):
    from agents.watch_context import set_context
    set_context('tester','c1','lead-orchestrator','c1')
    rt=factory([node('a')]);rt.start(1);ticket=rt.store.claim(rt.workflow_id,'a')
    assert rt.store.mark_user_report_dispatch_uncertain(rt.workflow_id,'a',ticket['token'])
    monkeypatch.setattr('agents.session.Session._execute_loop',lambda *args:pytest.fail('uncertain claim entered tool execution'))
    rt._worker('a',ticket)
    state=rt.snapshot();assert state['nodes']['a']['status']=='uncertain'
    assert ticket['token'] in rt.store.snapshot()['leases']


def test_worker_exception_after_uncertain_cannot_release_lease(factory,monkeypatch):
    import threading
    from agents.watch_context import set_context
    set_context('tester','c1','lead-orchestrator','c1')
    rt=factory([node('a')]);rt.start(1);ticket=rt.store.claim(rt.workflow_id,'a')
    entered=threading.Event();release=threading.Event()
    def body(step,work):
        entered.set();assert release.wait(5);raise ValueError('preflight witness rejected')
    monkeypatch.setattr(rt,'_worker_body',body)
    thread=threading.Thread(target=rt._worker,args=('a',ticket));thread.start();assert entered.wait(5)
    assert rt.store.mark_user_report_dispatch_uncertain(rt.workflow_id,'a',ticket['token'])
    release.set();thread.join(5);assert not thread.is_alive()
    assert rt.snapshot()['nodes']['a']['status']=='uncertain' and ticket['token'] in rt.store.snapshot()['leases']


def test_real_worker_uncertain_at_tool_entry_never_executes_or_releases(factory,monkeypatch):
    import threading
    from agents.watch_context import set_context
    set_context('tester','c1','lead-orchestrator','c1')
    rt=factory([node('a')]);rt.start(1);ticket=rt.store.claim(rt.workflow_id,'a')
    entered=threading.Event();release=threading.Event();native=rt.store.enter_tool_if_running
    def guarded(*args,**kwargs):entered.set();assert release.wait(5);return native(*args,**kwargs)
    monkeypatch.setattr(rt.store,'enter_tool_if_running',guarded)
    monkeypatch.setattr(rt.main.registry,'execute_dict',lambda *args:pytest.fail('uncertain worker entered tool'))
    thread=threading.Thread(target=rt._worker,args=('a',ticket));thread.start();assert entered.wait(5)
    assert rt.store.mark_user_report_dispatch_uncertain(rt.workflow_id,'a',ticket['token'])
    release.set();thread.join(5);assert not thread.is_alive()
    assert rt.snapshot()['nodes']['a']['status']=='uncertain' and ticket['token'] in rt.store.snapshot()['leases']


def test_real_worker_user_pause_at_entry_releases_unstarted_claim(factory,monkeypatch):
    import threading
    from agents.watch_context import set_context
    set_context('tester','c1','lead-orchestrator','c1')
    rt=factory([node('a')]);rt.start(1);ticket=rt.store.claim(rt.workflow_id,'a')
    entered=threading.Event();release=threading.Event();native=rt.store.enter_tool_if_running
    def guarded(*args,**kwargs):entered.set();assert release.wait(5);return native(*args,**kwargs)
    monkeypatch.setattr(rt.store,'enter_tool_if_running',guarded)
    monkeypatch.setattr(rt.main.registry,'execute_dict',lambda *args:pytest.fail('paused worker entered tool'))
    thread=threading.Thread(target=rt._worker,args=('a',ticket));thread.start();assert entered.wait(5)
    with json_transaction(rt.store.path) as data:
        state=data['workflows'][rt.workflow_id];state.update(user_paused=True,user_pause_revision=9,pause_kind='user_change',pause_reason='user paused')
    release.set();thread.join(5);assert not thread.is_alive()
    state=rt.snapshot();assert state['nodes']['a']['status']=='pending' and ticket['token'] not in rt.store.snapshot()['leases']
    assert state['user_paused'] is True and state['user_pause_revision']==9 and state['pause_kind']=='user_change' and state['pause_reason']=='user paused'
