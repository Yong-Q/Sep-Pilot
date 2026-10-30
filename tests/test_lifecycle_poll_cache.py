import json
import os
import time

from agents import lifecycle as m


def test_pending_cache_invalidation_and_claim(tmp_path, monkeypatch):
    store=m.LifecycleStore(tmp_path/'lifecycle.json')
    store.initialize()
    store.emit('science', {'value': 1}, event_id='one')
    count=0
    original=store._poll_snapshot
    def snapshot():
        nonlocal count
        count+=1
        return original()
    monkeypatch.setattr(store, '_poll_snapshot', snapshot)
    first=store.pending()
    first[0]['payload']['value']=999
    assert store.pending()[0]['payload']['value']==1
    assert count==1
    store.emit('science', {'value': 2}, event_id='two')
    assert {e['event_id'] for e in store.pending()}=={'one','two'}
    assert count==2
    assert store.claim('one','main_chat')
    assert next(e for e in store.pending() if e['event_id']=='one')['main_chat']=='processing'
    store.receipt('one','main_chat','delivered')
    assert next(e for e in store.pending() if e['event_id']=='one')['main_chat']=='delivered'
    store.receipt('one','supervisor','delivered')
    assert {e['event_id'] for e in store.pending()}=={'two'}


def test_heartbeat_uses_bounded_index_and_still_warns(tmp_path,monkeypatch):
    store=m.LifecycleStore(tmp_path/'lifecycle.json')
    store.initialize();store.emit('science',{},event_id='one');assert store.claim('one','main_chat')
    with m.json_transaction(store.path) as data:
        data['events']['one']['main_chat_claimed_at']=time.time()-359
    store.pending()
    assert len(json.dumps(store._poll_cache[2]))<4096
    monkeypatch.setattr(store,'snapshot',lambda: (_ for _ in ()).throw(AssertionError('unnecessary history read')))
    store.heartbeat()
    # Same immutable file, but elapsed wall time must still cross threshold.
    wall_time=time.time
    monkeypatch.setattr(m.time,'time',lambda: wall_time()+3)
    store._last_heartbeat=0
    store.heartbeat()
    data=json.loads(store.path.read_text())
    assert 'role-unresponsive:main_chat:one' in data['events']


def test_replaced_file_invalidates_and_large_pending_is_not_cached(tmp_path):
    store=m.LifecycleStore(tmp_path/'lifecycle.json');store.initialize()
    store.emit('science',{'value':'a'},event_id='one')
    assert store.pending()[0]['payload']['value']=='a'
    data=json.loads(store.path.read_text()); data['events']['one']['payload']['value']='b'
    other=tmp_path/'replacement';other.write_text(json.dumps(data));os.replace(other,store.path)
    assert store.pending()[0]['payload']['value']=='b'
    store.emit('science',{'blob':'x'*(1024*1024+1)},event_id='large')
    store.pending()
    assert store._poll_cache[0] != store._file_identity()


def test_paused_mailbox_heartbeat_caches_without_pending(tmp_path, monkeypatch):
    store=m.LifecycleStore(tmp_path/'lifecycle.json');store.initialize()
    count=0
    original=store._poll_snapshot
    def snapshot():
        nonlocal count
        count+=1
        return original()
    monkeypatch.setattr(store, '_poll_snapshot', snapshot)
    store.heartbeat();store._last_heartbeat=0;store.heartbeat()
    assert count==1
    store.emit('science',{},event_id='one')
    store._last_heartbeat=0;store.heartbeat()
    assert count==2


def test_receipt_during_heartbeat_observation_cannot_generate_stale_warning(tmp_path, monkeypatch):
    store=m.LifecycleStore(tmp_path/'lifecycle.json');store.initialize()
    store.emit('science',{},event_id='one');assert store.claim('one','main_chat')
    with m.json_transaction(store.path) as data:
        data['events']['one']['main_chat_claimed_at']=time.time()-400
    original=store._poll_snapshot
    def receipt_after_read():
        result=original()
        store.receipt('one','main_chat','delivered')
        return result
    monkeypatch.setattr(store,'_poll_snapshot',receipt_after_read)
    store.heartbeat()
    assert not any(e.get('kind')=='role_unresponsive' for e in json.loads(store.path.read_text())['events'].values())
