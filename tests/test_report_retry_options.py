import copy,json
from types import SimpleNamespace
from fastapi.testclient import TestClient
from tests.test_report_bounded_retry import candidate
from agents.report_bounded_retry import _digest
from agents.workflow_patch import execution_contract


def test_only_owner_sees_advisory_owned_failures_and_ledger(candidate,tmp_path,monkeypatch):
    import api
    runtime,goal,_,_=candidate;state=runtime.snapshot();owner={'username':'owner','conv_id':'case'}
    # Include another user's complete workflow; it must never appear.
    foreign=copy.deepcopy(state);foreign.update(username='foreign',conv_id='other')
    node=state['nodes']['report'];contract_key=_digest(execution_contract(node['contract']))
    payload={'workflows':{'workflow':state,'foreign':foreign},'leases':{},
        'user_report_retry_ledger':{_digest(owner):[{'contract_key':contract_key}]}}
    store=tmp_path/'data/state/parallel_workflows.json';store.parent.mkdir(parents=True);store.write_text(json.dumps(payload))
    monkeypatch.setattr(api,'__file__',str(tmp_path/'api.py'))
    conv=SimpleNamespace(username='owner',conv_id='case')
    monkeypatch.setattr(api.session_manager,'get_user',lambda user:SimpleNamespace(get_conversation=lambda cid:conv if cid=='case' else None))
    api.app.dependency_overrides[api._get_user]=lambda:'owner'
    try:
        before=store.read_bytes();client=TestClient(api.app)
        r=client.get('/api/conversations/case/workflow/report-retry-options')
        assert r.status_code==200 and r.json()['advisory'] is True
        options=r.json()['options'];assert len(options)==1 and options[0]['retry_state']=='used'
        assert options[0]['step_id']=='report' and options[0]['expected_failure_call_id']=='failedcall'
        assert 'foreign' not in r.text and str(runtime.root) not in r.text
        assert store.read_bytes()==before
    finally:api.app.dependency_overrides.pop(api._get_user,None)


def test_unverified_option_is_explicitly_check_required(candidate,tmp_path,monkeypatch):
    import api
    runtime,_,_,_=candidate;store=tmp_path/'data/state/parallel_workflows.json';store.parent.mkdir(parents=True);store.write_bytes(runtime.store.path.read_bytes())
    monkeypatch.setattr(api,'__file__',str(tmp_path/'api.py'));conv=SimpleNamespace(username='owner',conv_id='case')
    monkeypatch.setattr(api.session_manager,'get_user',lambda user:SimpleNamespace(get_conversation=lambda cid:conv if cid=='case' else None))
    api.app.dependency_overrides[api._get_user]=lambda:'owner'
    try:
        r=TestClient(api.app).get('/api/conversations/case/workflow/report-retry-options')
        assert r.status_code==200 and r.json()['options'][0]['retry_state']=='check_required'
    finally:api.app.dependency_overrides.pop(api._get_user,None)
