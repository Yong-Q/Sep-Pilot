"""Real UI/API/Session recovery acceptance in an isolated source snapshot.

No API or LLM responses are mocked. Missing input is an explicitly synthetic
filesystem fault. Only read-only evidence tools are allowed; no HPC is contacted.
Requires marionette_driver, Firefox, and existing Codex CLI login.
"""
import argparse
import asyncio
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

from marionette_driver.marionette import Marionette

ROOT = Path(__file__).resolve().parents[2]


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')


def wait_browser(client, expression, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.execute_script('return ' + expression)
        if result:
            return result
        time.sleep(.2)
    raise AssertionError('Browser condition timed out: ' + expression)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=480)
    parser.add_argument('--resume-isolated', type=Path, help='Resume only a prior task-owned isolated test root')
    parser.add_argument('--allow-one-ui-approval', action='store_true', help='Exercise one real UI reply if planning asks for approval; count it in the receipt')
    args = parser.parse_args()
    evidence = args.evidence.resolve(); evidence.mkdir(parents=True, exist_ok=False)
    evidence.chmod(0o700)
    isolated = args.resume_isolated.resolve() if args.resume_isolated else evidence / 'isolated'
    if args.resume_isolated:
        assert isolated.parent.parent == ROOT / 'reports' and isolated.name == 'isolated'
    else:
        isolated.mkdir()
    copied = [ROOT / 'api.py', ROOT / 'auth.py'] + sorted((ROOT / 'agents').glob('*.py'))
    sources = {}
    for source in copied:
        relative = source.relative_to(ROOT); target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not args.resume_isolated: shutil.copyfile(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if not args.resume_isolated: assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        sources[str(relative)] = digest
    if not args.resume_isolated:
        (isolated / 'frontend').mkdir()
        (isolated / 'frontend/build').symlink_to(ROOT / 'frontend/build', target_is_directory=True)
    state = isolated / 'data/state'; state.mkdir(parents=True, exist_ok=True)
    if not args.resume_isolated:
        for name in ('task_lines.json', 'job_watch.json'):
            (state / name).write_text('{}')
    # Existing import-time stores are now scoped before ANY production import.
    os.environ['BIMEM_BACKGROUND_MONITOR'] = '0'
    os.environ['BIMEM_JOB_WATCH_FILE'] = str(state / 'job_watch.json')
    os.environ['BIMEM_MODEL_PROVIDER'] = 'codex_cli'
    os.environ['BIMEM_MODEL_CALL_CONCURRENCY'] = '1'
    def protect_real_state(event, values):
        if event != 'open' or not values or not isinstance(values[0], (str, bytes)):
            return
        candidate = Path(os.fsdecode(values[0])).resolve()
        forbidden_files = {ROOT / name for name in ('users.json', 'tokens.json', 'conversation_logs.jsonl', 'env/settings.json')}
        forbidden_files.add(Path('/tmp/bimem_task_line.json'))
        if candidate in forbidden_files or candidate.is_relative_to(ROOT / 'data/state') or candidate.is_relative_to(ROOT / 'runs'):
            raise PermissionError('Integration harness refuses access to real user/runtime state')
    sys.addaudithook(protect_real_state)
    sys.path.insert(0, str(isolated))
    from agents.config import AgentConfig, set_config
    set_config(AgentConfig(api_key='test-unused-local-provider', model_provider='codex_cli',
                           project_root=isolated, legacy_root=isolated,
                           max_orchestrator_turns=12, max_specialist_turns=12))
    import auth
    import api
    import agents.task_line as task_line
    import agents.registry as registry_module
    from agents.session import Session, ExecutionBudgetExceeded
    assert Path(auth.USER_DB).parent == isolated
    assert Path(auth.TOKEN_DB).parent == isolated
    assert Path(task_line.get_store()._file).is_relative_to(isolated)
    assert Path(api._watch._watch_file).is_relative_to(isolated)
    assert Path(api.__file__).parent == isolated
    events = json.loads((isolated.parent / 'events.json').read_text()) if args.resume_isolated else []
    def record(kind, **values):
        events.append({'time': time.time(), 'kind': kind, **values})
        dump(evidence / 'events.json', events)
    fixture = isolated / 'fixture'; fixture.mkdir(exist_ok=True)
    (fixture / 'current').mkdir(exist_ok=True)
    if not args.resume_isolated:
        (fixture / 'index.json').write_text(json.dumps({
            'fixture_kind': 'SYNTHETIC TEST ONLY; no scientific computation performed',
            'old_path': 'stale_summary.json', 'current_result': 'current/summary.json',
            'reason': 'Result was moved during a controlled filesystem recovery test.',
        }))
        (fixture / 'current/summary.json').write_text(json.dumps({
            'fixture_kind': 'SYNTHETIC TEST ONLY', 'fixture_case_tag': 'ui-recovery-proof',
            'gas': 'CO2', 'temperature_K': 298, 'uptake_mmol_g': 2.75,
            'scientific_limit': 'Synthetic adsorption evidence cannot establish membrane permeability.',
        }))
    allowed = {'read_file', 'inspect_path', 'grep_search', 'get_tool_schema',
               'task_line_query', 'lifecycle_state', 'recovery_state'}
    def instrument_executor(original, structured):
        def observed_execute(self, name, params):
            denial = ''
            if name not in allowed:
                denial = 'This read-only acceptance permits only inspecting declared evidence; no jobs or mutation.'
            for key in ('path', 'cwd'):
                if params.get(key):
                    path = Path(params[key]); path = path if path.is_absolute() else isolated / path
                    if not path.resolve().is_relative_to(fixture):
                        denial = 'Read only the declared fixture directory.'
            if denial:
                record('tool_guard', tool=name, reason=denial)
                return {'error': denial} if structured else json.dumps({'error': denial})
            record('tool_start', tool=name, params=params)
            result = original(self, name, params)
            record('tool_result', tool=name, params=params,
                   result=json.dumps(result, ensure_ascii=False) if structured else result)
            return result
        return observed_execute
    registry_module.ToolRegistry.execute = instrument_executor(registry_module.ToolRegistry.execute, False)
    registry_module.ToolRegistry.execute_dict = instrument_executor(registry_module.ToolRegistry.execute_dict, True)
    calls = []
    original_call = Session._call_api
    started = time.monotonic()
    def observed_call(self, agent):
        if len(calls) >= 12 or time.monotonic() - started > args.timeout:
            raise ExecutionBudgetExceeded('UI integration acceptance budget reached')
        calls.append(agent.name); record('model_start', agent=agent.name, call=len(calls))
        response = original_call(self, agent)
        record('model_finish', agent=agent.name, call=len(calls))
        return response
    Session._call_api = observed_call
    # Register an isolated test identity through the real auth implementation.
    synthetic_password = 'test-synthetic-ui-protocol-password'
    if args.resume_isolated:
        synthetic_password = synthetic_password.removeprefix('test-')
    else:
        assert auth.register_user('ui_protocol_test', synthetic_password)['ok']
    @api.app.on_event('startup')
    async def start_local_lifecycle():
        # Run the real dispatcher without starting the external scheduler watcher.
        asyncio.create_task(api._drain_lifecycle_events())
        record('local_lifecycle_dispatcher_enabled')
    prompt = (
        f'这是合成协议验收，不是真实科学结果。请直接读取 {fixture / "stale_summary.json"}，'
        '检查这份 CO2 吸附摘要，并向我报告 fixture_case_tag、温度、吸附量和单位。'
        '路径若读取失败，请自己检查同目录的来源记录找回当前产物继续完成，不要向我索要路径。'
        '说明恢复依据，并区分吸附量与膜渗透性能。只允许读取此任务目录，不修改文件，不运行计算或shell，不提交作业。'
    )
    dump(evidence / 'input.json', {'prompt': prompt, 'source_sha256': sources,
        'kind': 'REAL_REACT_API_SESSION_MODEL_SYNTHETIC_MISSING_FILE',
        'boundaries': ['byte-identical copied API/auth/agent sources', 'isolated auth/stores',
                       'native missing-file tool error', 'read-only tool guard', 'background HPC watcher disabled; real lifecycle dispatcher enabled']})
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(api.app, host='127.0.0.1', port=0, log_level='error'))
    sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    server_thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    server_thread.start()
    deadline = time.monotonic()+15
    while not server.started and time.monotonic()<deadline: time.sleep(.1)
    assert server.started
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); browser_port = probe.getsockname()[1]
    receipt = {'passed': False}
    conv = None
    with tempfile.TemporaryDirectory(prefix='browser-profile-', dir=evidence) as profile:
        Path(profile, 'user.js').write_text(f'user_pref("marionette.port", {browser_port});\n')
        browser = subprocess.Popen(['firefox', '--headless', '--no-remote', '--profile', profile, '--marionette'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        client = Marionette(port=browser_port, socket_timeout=10)
        try:
            client.start_session(timeout=30); client.set_window_rect(width=1440, height=1000)
            client.navigate(f'http://127.0.0.1:{port}/')
            wait_browser(client, '!!document.querySelector(".auth-field input")')
            client.find_element('css selector', '.auth-field input').send_keys('ui_protocol_test')
            client.find_element('css selector', 'input[type=password]').send_keys(synthetic_password)
            client.find_element('css selector', '.auth-btn').click()
            wait_browser(client, '!!document.querySelector("textarea")')
            if not args.resume_isolated:
                client.find_element('css selector', 'textarea').send_keys(prompt)
                record('ui_submit')
                client.find_element('css selector', '.send-btn').click()
            else:
                record('ui_resume_observation')
            deadline = time.monotonic()+args.timeout
            conv = None
            ui_approvals = 0
            while time.monotonic()<deadline:
                user = api.session_manager.get_user('ui_protocol_test')
                conv = user.get_current_conversation()
                if conv.session and conv.current_uid and not conv.is_processing and conv.current_answer:
                    pending = conv.session._pending_user_interaction or {}
                    if (args.allow_one_ui_approval and ui_approvals == 0
                            and pending.get('tool') == 'workflow_patch_decision'):
                        record('unexpected_approval_prompt', prompt=pending.get('prompt'))
                        wait_browser(client, '!document.querySelector(".send-btn").textContent.includes("处理中")')
                        client.find_element('css selector', 'textarea').send_keys('批准这个仅检查指定目录的只读方案，请继续完成；路径错误请自己沿来源记录修复，不需要我补路径。')
                        record('ui_approval_submit')
                        client.find_element('css selector', '.send-btn').click()
                        ui_approvals += 1
                        wait_browser(client, 'document.querySelector(".send-btn").disabled')
                        time.sleep(1)
                        continue
                    if 'ui-recovery-proof' in conv.current_answer or pending or conv.session.task_complete:
                        break
                time.sleep(1)
            else:
                if conv: conv.interrupt_requested = True
                raise AssertionError('Actual API/Session did not finish within acceptance timeout')
            answer = conv.current_answer
            wait_browser(client, 'Array.from(document.querySelectorAll(".agent-message .message-content")).some(el => el.textContent.includes("ui-recovery-proof"))', timeout=35)
            ui_answer = client.execute_script('return Array.from(document.querySelectorAll(".agent-message .message-content")).map(el => el.textContent).join("\\n")')
            failed = [event for event in events if event['kind']=='tool_result' and event['tool']=='read_file'
                      and 'File not found:' in event['result']]
            recovered = [event for event in events if event['kind']=='tool_result' and event['tool']=='read_file'
                         and 'ui-recovery-proof' in event['result']]
            assert failed, 'No native missing-file read was observed; not an error-recovery pass'
            assert recovered and failed[0]['time'] < recovered[-1]['time']
            assert 'ui-recovery-proof' in answer and '2.75' in answer and '2.75' in ui_answer
            (evidence / 'final-ui.png').write_bytes(client.screenshot(format='binary'))
            dump(evidence / 'final_workflow.json', api._workflow_state(conv))
            dump(evidence / 'session_state.json', conv.session.export_state())
            record('ui_result_visible', conv_id=conv.conv_id)
            receipt = {'passed': True, 'model_calls': len(calls), 'agents': calls, 'resumed_from': str(args.resume_isolated or ''),
                       'native_read_failure_at': failed[0]['time'], 'recovered_read_at': recovered[-1]['time'],
                       'answer': answer, 'ui_answer': ui_answer,
                       'task_complete': conv.session.task_complete, 'extra_ui_approvals': ui_approvals,
                       'dag_node_count': client.execute_script('return document.querySelectorAll(".stable-dag-node").length'),
                       'scope': 'Actual React -> actual local API -> real Codex Session -> native missing-file error -> autonomous evidence recovery -> browser-visible final answer. Synthetic fixture only; no HPC submission.'}
        except Exception as exc:
            receipt['error'] = f'{type(exc).__name__}: {exc}'
            if conv and conv.session:
                receipt['pending_user_interaction'] = conv.session._pending_user_interaction
                receipt['answer'] = conv.current_answer
                dump(evidence / 'final_workflow.json', api._workflow_state(conv))
                dump(evidence / 'session_state.json', conv.session.export_state())
            try: (evidence / 'failure-ui.png').write_bytes(client.screenshot(format='binary'))
            except Exception: pass
            raise
        finally:
            dump(evidence / 'receipt.json', receipt)
            try: client.delete_session()
            except Exception: pass
            browser.terminate()
            try: browser.wait(timeout=10)
            except subprocess.TimeoutExpired: browser.kill(); browser.wait()
            server.should_exit = True; server_thread.join(timeout=10); sock.close()
    print(json.dumps({'passed': receipt['passed'], 'model_calls': len(calls), 'evidence': str(evidence)}))


if __name__ == '__main__':
    main()
