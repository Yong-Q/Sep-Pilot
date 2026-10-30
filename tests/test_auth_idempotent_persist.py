import json
import auth

def setup(tmp_path,monkeypatch):
 user_db=tmp_path/'users.json';token_db=tmp_path/'tokens.json';monkeypatch.setattr(auth,'USER_DB',user_db);monkeypatch.setattr(auth,'TOKEN_DB',token_db)
 user_db.write_text(json.dumps({'u':{'display_name':'u','conversations':{}}}));manager=auth.UserSessionManager();user=manager.get_user('u');conv=user.new_conversation();return user_db,manager,user,conv

def test_identical_snapshot_preserves_inode_and_bytes(tmp_path,monkeypatch):
 path,manager,user,conv=setup(tmp_path,monkeypatch);manager.persist_conversations('u',conv_id=conv.conv_id);before=path.read_bytes();ino=path.stat().st_ino
 manager.persist_conversations('u',conv_id=conv.conv_id)
 assert path.read_bytes()==before and path.stat().st_ino==ino

def test_changed_checkpoint_and_type_are_not_dropped(tmp_path,monkeypatch):
 path,manager,user,conv=setup(tmp_path,monkeypatch);conv.session_state={'checkpoint_at':1,'value':True};manager.persist_conversations('u',conv_id=conv.conv_id)
 conv.session_state={'checkpoint_at':2,'value':1};manager.persist_conversations('u',conv_id=conv.conv_id)
 data=json.loads(path.read_text())['u']['conversations'][conv.conv_id]['session_state']
 assert data['checkpoint_at']==2 and type(data['value']) is int

def test_concurrent_external_conversation_survives_same_snapshot(tmp_path,monkeypatch):
 path,manager,user,conv=setup(tmp_path,monkeypatch);manager.persist_conversations('u',conv_id=conv.conv_id)
 disk=json.loads(path.read_text());disk['u']['conversations']['other']={'title':'external'};path.write_text(json.dumps(disk))
 manager.persist_conversations('u',conv_id=conv.conv_id)
 assert json.loads(path.read_text())['u']['conversations']['other']=={'title':'external'}

def test_explicit_tombstone_not_ignored(tmp_path,monkeypatch):
 path,manager,user,conv=setup(tmp_path,monkeypatch);manager.persist_conversations('u',conv_id=conv.conv_id)
 user.conversations.pop(conv.conv_id)
 manager.persist_conversations('u',delete_conv_id=conv.conv_id)
 assert conv.conv_id not in json.loads(path.read_text())['u']['conversations']

def test_json_compatible_mixed_key_types_do_not_break_persistence(tmp_path,monkeypatch):
 path,manager,user,conv=setup(tmp_path,monkeypatch);conv.session_state={'mixed':{1:'a','2':'b'}}
 manager.persist_conversations('u',conv_id=conv.conv_id)
 manager.persist_conversations('u',conv_id=conv.conv_id)
 assert json.loads(path.read_text())['u']['conversations'][conv.conv_id]['session_state']['mixed']=={'1':'a','2':'b'}

def test_unchanged_small_conversation_does_not_rewrite_large_other_history(tmp_path,monkeypatch):
 path,manager,user,conv=setup(tmp_path,monkeypatch);manager.persist_conversations('u',conv_id=conv.conv_id)
 disk=json.loads(path.read_text());disk['u']['conversations']['other']={'history':'x'*(12*1024*1024)};path.write_text(json.dumps(disk))
 before=path.stat();manager.persist_conversations('u',conv_id=conv.conv_id);after=path.stat()
 assert after.st_ino==before.st_ino and after.st_size==before.st_size
