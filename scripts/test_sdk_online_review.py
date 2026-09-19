"""Exercise deployed authenticated API and real SDK audit in one owned test user."""
import json
import argparse
from pathlib import Path
import secrets
import sys
import time
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.state_io import write_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expected-version',default='3.4.8')
    parser.add_argument('--case',choices=['opls_exact_atom','tip4p_water','session_directory'],default='opls_exact_atom')
    args=parser.parse_args()
    username='sdk_online_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    output=ROOT/'reports'/username;output.mkdir(exist_ok=False)
    token=''
    def request(path,data=None):
        body=json.dumps(data).encode() if data is not None else None
        headers={'Content-Type':'application/json'}
        if token:headers['Authorization']='Bearer '+token
        with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/'+path,data=body,headers=headers),timeout=10) as stream:
            return json.load(stream)
    result={'kind':'DEPLOYED_API_REAL_SDK_SCIENTIFIC_REVIEW','username':username,'case_id':args.case}
    try:
        health=request('health')
        if health.get('version')!=args.expected_version:raise RuntimeError('Expected deployed '+args.expected_version)
        result['version']=health['version']
        registered=request('api/register',{'username':username,'password':secrets.token_urlsafe(32),'display_name':'SDK release validation'})
        token=registered['token']  # Never print/save credentials.
        cases=json.loads((ROOT/'tests/fixtures/forcefield_agent_cases.json').read_text())['cases']
        if args.case=='session_directory':
            query='只读检查当前会话工作目录。根据系统提供的真实username/conv_id，自主用run_bash执行ls runs/用户名/会话ID/（替换为真实值），再用inspect_path检查同一目录的绝对路径。报告真实结果，不提交计算、不修改文件；不要把目录当read_file文件。'
        else:
            query=next(case['prompt'] for case in cases if case['id']==args.case)+' 自主调用正式科学工具，不提交计算、不修改文件。'
        started=request('api/query',{'query':query});cid=started['conv_id'];result['conv_id']=cid
        deadline=time.monotonic()+480
        root=ROOT/'runs'/username/cid
        while time.monotonic()<deadline:
            state=request('api/conversations/'+cid)
            if args.case=='session_directory' and not state.get('is_processing',True) and (root/'session_checkpoint.json').exists():
                checkpoint=json.loads((root/'session_checkpoint.json').read_text())
                calls=checkpoint.get('memory',{}).get('tool_call_log',[])
                def fact(call):
                    value=call.get('result')
                    if value is None and call.get('evidence_path'):
                        path=Path(call['evidence_path']).resolve()
                        if not path.is_relative_to(root/'evidence'):raise ValueError('Online evidence escaped owned scope')
                        value=json.loads(path.read_text()).get('result')
                    return json.loads(value) if isinstance(value,str) else value or {}
                shell=[c for c in calls if c['tool']=='run_bash']
                inspect=[c for c in calls if c['tool']=='inspect_path']
                if shell and inspect:
                    result['checks']={'online_version_correct':True,
                        'sdk_supplied_real_project_prefixed_shell':any(f'runs/{username}/{cid}' in c['params'].get('command','') and fact(c).get('exit_code')==0 for c in shell),
                        'sdk_inspected_own_absolute_path':any(Path(c['params'].get('path','')).is_absolute() and not c.get('failed') for c in inspect),
                        'no_computation_submitted':not (root/'recovery_state.json').exists() or not json.loads((root/'recovery_state.json').read_text()).get('entries')}
                    result['tool_calls']=calls;result['passed']=all(result['checks'].values());break
            mailbox=root/'lifecycle.json'
            if mailbox.exists():
                data=json.loads(mailbox.read_text())
                reviews=[e for e in data.get('events',{}).values() if e['kind']=='scientific_review']
                finished=[e for e in reviews if e.get('supervisor_receipt',{}).get('review_action')=='verified']
                if finished and not state.get('is_processing',True):
                    result['checks']={'online_version_correct':True,'real_main_created_review':bool(reviews),
                        'independent_supervisor_verified':bool(finished),
                        'independent_supervisor_used_source':all(e['supervisor_receipt'].get('independent_source_call_ids') for e in finished),
                        'no_computation_submitted':not (root/'recovery_state.json').exists() or not json.loads((root/'recovery_state.json').read_text()).get('entries')}
                    result['reviews']=reviews;result['passed']=all(result['checks'].values());break
            time.sleep(2)
        else:raise TimeoutError('Deployed SDK review did not reach verified within bounded validation window')
    except Exception as error:result.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        write_checkpoint(output/'test_result.json',result)
        print(json.dumps({'passed':result.get('passed'),'error':result.get('error'),'report':str(output/'test_result.json')},ensure_ascii=False),flush=True)
    return 0 if result.get('passed') else 1


if __name__=='__main__':raise SystemExit(main())
