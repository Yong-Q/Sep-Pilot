"""Correlation is distinct from content, authority and retry idempotency."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest


def fixture_conv():
    return SimpleNamespace(username='owner', conv_id='case', session=SimpleNamespace(_current_line_id='line'),
        is_processing=False, current_uid='old', interrupt_requested=False, interrupt_message='',
        execution_logs=[], messages_history=[], current_answer='', current_reasoning='',
        current_status='', partial_results=[], current_agent='lead')


@pytest.mark.parametrize('value', ['', 'contains space', '中文', 'x'*129, 'line\nbreak', 'slash/path'])
def test_client_turn_id_rejects_invalid_ascii_ids(value, monkeypatch):
    import api
    conv=fixture_conv(); monkeypatch.setattr(api,'_get_conv',lambda *a:conv)
    async def unused(*a,**kw):pass
    monkeypatch.setattr(api,'run_agent_background',unused)
    api.app.dependency_overrides[api._get_user]=lambda:'owner'
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://test') as client:
            response=await client.post('/api/query',json={'query':'same text','conv_id':'case','client_turn_id':value})
            assert response.status_code==422
    try:asyncio.run(scenario())
    finally:api.app.dependency_overrides.clear()


def test_same_text_different_client_turns_echo_distinct_ids_and_old_clients_keep_call_shape(monkeypatch):
    import api
    conv=fixture_conv(); calls=[]
    async def background(*args,**kwargs):
        calls.append((args,kwargs));conv.is_processing=False
    monkeypatch.setattr(api,'_get_conv',lambda *a:conv)
    monkeypatch.setattr(api,'run_agent_background',background)
    api.app.dependency_overrides[api._get_user]=lambda:'owner'
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://test') as client:
            for client_id in ('click_A','click_B',None):
                body={'query':'same text','conv_id':'case'}
                if client_id is not None:body['client_turn_id']=client_id
                response=await client.post('/api/query',json=body)
                assert response.status_code==200
                assert response.json().get('client_turn_id')==client_id
                await asyncio.sleep(0)
    try:asyncio.run(scenario())
    finally:api.app.dependency_overrides.clear()
    assert [c[0] for c in calls]==[('owner','same text','case')]*3
    assert calls[0][1]=={'client_turn_id':'click_A'}
    assert calls[1][1]=={'client_turn_id':'click_B'}
    assert calls[2][1]=={}


def test_redirect_preserves_correlation_through_the_actual_queue(monkeypatch):
    import api
    import threading
    conv=fixture_conv();conv.is_processing=True;conv.interrupt_requested=True
    user=SimpleNamespace(get_conversation=lambda _:conv)
    monkeypatch.setattr(api,'session_manager',SimpleNamespace(get_user=lambda _:user))
    monkeypatch.setattr(api,'_get_conv',lambda *a:conv)
    monkeypatch.setattr(api,'_get_conv_lock',lambda *a:threading.Lock())
    calls=[]
    async def background(*a,**kw):calls.append((a,kw))
    monkeypatch.setattr(api,'run_agent_background',background)
    async def scenario():
        response=await api.process_query(api.QueryRequest(query='redirect',conv_id='case',client_turn_id='redirect_1'),'owner')
        assert response.client_turn_id=='redirect_1'
        await asyncio.sleep(0)
    asyncio.run(scenario())
    assert calls==[(('owner','redirect','case'),{'client_turn_id':'redirect_1'})]


def test_actual_user_archive_has_server_uid_correlation_timestamp_even_when_first_persist_is_cancelled(monkeypatch):
    import api
    conv=fixture_conv();conv.current_uid='server-turn-1'
    user=SimpleNamespace(get_conversation=lambda _:conv)
    monkeypatch.setattr(api,'session_manager',SimpleNamespace(get_user=lambda _:user,persist_conversations=lambda *a,**k:None))
    monkeypatch.setattr(api,'_ensure_session',lambda _:None)
    async def cancelled(*a,**kw):raise asyncio.CancelledError
    monkeypatch.setattr(api,'_durable_io',cancelled)
    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            await api.run_agent_background('owner','same text','case',client_turn_id='click_1')
    asyncio.run(scenario())
    message=conv.messages_history[0]
    assert message['role']=='user' and message['content']=='same text'
    assert message['turn_uid']=='server-turn-1'
    assert message['client_turn_id']=='click_1'
    assert message['timestamp']
    assert conv.is_processing is False


def test_history_keeps_new_correlation_and_old_records_unchanged(monkeypatch):
    import api
    conv=fixture_conv();old={'role':'user','content':'same text'}
    new={'role':'user','content':'same text','turn_uid':'server','client_turn_id':'click','timestamp':'now'}
    conv.messages_history=[old,new];conv.title='test'
    monkeypatch.setattr(api,'session_manager',SimpleNamespace(get_user=lambda _:SimpleNamespace(get_conversation=lambda _:conv)))
    async def light(_):return {}
    monkeypatch.setattr(api,'_workflow_state_async',light)
    result=asyncio.run(api.get_conversation('case','owner'))
    assert result['messages']==[old,new]
    assert set(old)=={'role','content'}
