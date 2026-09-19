"""Continue an explicitly authorized user's live API conversation.

Uses the existing credential in memory only; never prints/persists tokens.
Messages and real SDK actions remain in the normal frontend conversation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auth import load_tokens


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--user',required=True)
    parser.add_argument('--conv',required=True)
    parser.add_argument('--message',required=True)
    parser.add_argument('--wait-seconds',type=int,default=180)
    args=parser.parse_args()
    token=next((key for key,value in load_tokens().items() if value==args.user),None)
    if not token:raise RuntimeError('No existing credential for the explicitly authorized user')
    def request(path,data=None):
        body=json.dumps(data,ensure_ascii=False).encode() if data is not None else None
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}
        with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/'+path,data=body,headers=headers),timeout=15) as stream:
            return json.load(stream)
    state=request('api/conversations/'+args.conv)
    if state.get('is_processing'):raise RuntimeError('Conversation already processing; do not inject a concurrent turn')
    started=request('api/query',{'conv_id':args.conv,'query':args.message})
    print(json.dumps({'username':args.user,'conv_id':args.conv,'started':True,'response':started.get('answer','')},ensure_ascii=False),flush=True)
    deadline=time.monotonic()+args.wait_seconds
    while time.monotonic()<deadline:
        state=request('api/conversations/'+args.conv)
        if not state.get('is_processing',False):
            answer=request('api/latest_answer?conv_id='+args.conv)
            print(json.dumps({'processing':False,'answer':answer.get('answer',''),'agent_name':answer.get('agent_name'),
                              'status':answer.get('status'),'logs':answer.get('logs',[])[-5:]},ensure_ascii=False),flush=True)
            return
        time.sleep(2)
    print(json.dumps({'processing':True,'note':'SDK turn continues in frontend; no cancellation or duplicate message'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
