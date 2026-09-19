"""Operator-authorized release pause/restore, preserving preexisting user pauses.

Uses short-lived owner-scoped credentials, never logs or persists their values.
The durable manifest records only exact conversations changed by this release.
"""
import argparse
import json
from pathlib import Path
import secrets
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auth import load_users, session_manager
from agents.state_io import write_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['pause', 'restore'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--all-active', action='store_true', help='Also pause idle conversations with active durable workflows')
    args = parser.parse_args()
    path = Path(args.manifest).resolve()
    if not path.is_relative_to(ROOT / 'logs' / 'releases'):
        raise ValueError('release manifest must be in project logs/releases')
    manifest = json.loads(path.read_text()) if path.exists() else {'scopes': []}
    if args.action == 'pause':
        users = load_users()
        scopes = {(name, cid) for name, user in users.items()
                  for cid, conv in user.get('conversations', {}).items() if conv.get('is_processing')}
        if args.all_active:
            scopes.update((name, cid) for name, user in users.items()
                          for cid, conv in user.get('conversations', {}).items() if conv.get('conversation_active'))
        for lifecycle in (ROOT / 'runs').glob('*/*/lifecycle.json'):
            if any(role.get('busy_event_id') for role in json.loads(lifecycle.read_text()).get('agents', {}).values()):
                scopes.add(tuple(lifecycle.relative_to(ROOT / 'runs').parts[:2]))
        for name, cid in sorted(scopes):
            conv = users.get(name, {}).get('conversations', {}).get(cid)
            if not conv or conv.get('interrupt_requested'):
                continue
            if [name, cid] not in manifest['scopes']:
                manifest['scopes'].append([name, cid])
                write_checkpoint(path, manifest)
            request(name, cid, 'interrupt')
    else:
        for name, cid in manifest['scopes']:
            request(name, cid, 'interrupt_clear')
        manifest['restored'] = True
        write_checkpoint(path, manifest)
    print(json.dumps({'action': args.action, 'scopes': manifest['scopes']}))


def request(username, cid, endpoint):
    token = secrets.token_urlsafe(40)
    session_manager.register_token(token, username)
    try:
        req = urllib.request.Request(f'http://127.0.0.1:8000/api/conversations/{cid}/{endpoint}',
            data=b'{}', headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            if not json.load(response).get('ok'):
                raise RuntimeError('conversation maintenance failed')
    finally:
        try:
            logout = urllib.request.Request('http://127.0.0.1:8000/api/logout', data=b'{}',
                headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(logout, timeout=10) as response:
                response.read()
        finally:
            session_manager.remove_token(token)


if __name__ == '__main__':
    main()
