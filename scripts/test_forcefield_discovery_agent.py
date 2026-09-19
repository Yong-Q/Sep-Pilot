"""Real model + formal SDK catalog tools. Read-only science, no jobs."""
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import uuid
import argparse
import subprocess
import shutil
import hashlib
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
RELEASE_SOURCES = ('api.py', 'agents/session.py', 'agents/scientific_review.py', 'agents/defns.py',
                   'agents/goal_contract.py', 'agents/charge_contract.py', 'agents/unit_contract.py',
                   'agents/forcefield_catalog.py', 'env/forcefield_sources.json', 'env/physical_units.json',
                   'registry/catalog.json', 'agents/workspace.py', 'agents/workflow_patch.py',
                   'agents/parallel_workflow.py', 'agents/registry.py', 'agents/recovery.py',
                   'agents/state_io.py', 'agents/task_line.py')
sys.path.insert(0, str(ROOT))
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.registry import _build_default_registry
from agents.session import Session, ExecutionBudgetExceeded
from agents.state_io import write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
from agents.task_line import TaskLineStore
from agents.watch_context import set_context, clear_context


def release_source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in RELEASE_SOURCES}


def normalize_evaluation_text(text):
    return unicodedata.normalize('NFKC', text).casefold().translate(str.maketrans({c:'-' for c in '‐‑‒–—−'})).replace('未被修改','未修改')


def restore_replay_evidence(state, source_root, destination_root):
    """Copy owned archived SDK evidence; never relax Session scope isolation."""
    source = (source_root / 'evidence').resolve()
    target = destination_root / 'evidence'
    target.mkdir(exist_ok=True)
    memory = state.get('memory', {})
    logs = [memory.get('tool_call_log', [])]
    logs.extend(item.get('tool_call_log', []) for item in memory.get('agent_memories', {}).values())
    for log in logs:
        for call in log:
            if not call.get('evidence_path'):
                continue
            call_id = call.get('call_id', '')
            archived = Path(call['evidence_path']).resolve()
            if not call_id.isalnum() or archived != source / (call_id + '.json'):
                raise ValueError('replay evidence is not in the exact owned SDK archive')
            fact = json.loads(archived.read_text())
            if any(fact.get(key) != call.get(key) for key in ('call_id', 'tool', 'agent', 'params')):
                raise ValueError('replay evidence identity differs from checkpoint')
            copied = target / archived.name
            if not copied.exists():
                shutil.copyfile(archived, copied)
            elif copied.read_bytes() != archived.read_bytes():
                raise ValueError('replay evidence collision')
            call['evidence_path'] = str(copied)
    return state


def run_case(case, replay_failed=None):
    cid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    root = ROOT / 'runs/sdk_forcefield_probe' / cid
    root.mkdir(parents=True, exist_ok=False)
    registry = _build_default_registry()
    lines = TaskLineStore(str(root / 'task_lines.json'))
    line = lines.begin_line(cid, username='sdk_forcefield_probe', conv_id=cid)
    s = Session(config=AgentConfig.from_env(max_tokens=8192, temperature=.2), registry=registry)
    s.client = s.client.with_options(timeout=90, max_retries=0)
    s._current_line_id = line['line_id']
    s._evidence_root = root / 'evidence'
    def checkpoint(state, reason):
        refs = compact_checkpoint_evidence(state, root / 'evidence')
        write_checkpoint(root / 'session_checkpoint.json', state)
        apply_evidence_references(s.memory, refs)
    s._on_checkpoint = checkpoint
    agent = copy.copy(ORCHESTRATOR)
    names = {'discover_forcefield', 'inspect_forcefield', 'validate_framework_charges', 'convert_physical_units', 'get_tool_schema', 'request_user_decision', 'task_line_query'}
    agent.functions = [fn for fn in agent.functions if fn.__name__ in names]
    calls = []
    call_limit = 6
    real = s._call_api
    def bounded(current):
        if len(calls) >= call_limit: raise ExecutionBudgetExceeded('forcefield inspection LLM budget reached')
        s.context['execution_budget'] = {'max_model_calls':call_limit,'used_model_calls':len(calls),
            'remaining_model_calls':call_limit-len(calls),'reserve_final_response':True,'scope':'read_only_parameter_investigation'}
        response = real(current)
        calls.append({'agent': current.name, 'tools': [{'name': b.name, 'arguments': b.input} for b in response.content if getattr(b,'type','')=='tool_use'],
            'stop_reason': response.stop_reason, 'output_tokens': response.usage.output_tokens})
        return response
    s._call_api = bounded
    report = {'kind': 'REAL_MODEL_READONLY_FORCEFIELD_DISCOVERY', 'case_id': case['id'], 'scope': str(root),
              'model_calls': calls, 'source_sha256': release_source_hashes()}
    try:
        with patch('agents.defns._registry', registry), patch('agents.defns.ORCHESTRATOR', agent), patch('agents.task_line.get_store', lambda: lines):
            set_context('sdk_forcefield_probe', cid, 'lead-orchestrator', line['line_id'])
            evidence_instruction = 'validate_framework_charges' if case.get('kind') == 'charge' else 'convert_physical_units' if case.get('kind') == 'units' else 'discover_forcefield/inspect_forcefield'
            task = case['prompt'] + f'\n必须使用项目正式{evidence_instruction}工具获取证据，不依赖模型记忆猜参数。'
            if replay_failed:
                source = Path(replay_failed).resolve()
                if not source.is_relative_to(ROOT/'runs/sdk_forcefield_probe') or source.name != 'test_result.json':
                    raise ValueError('replay must reference a retained SDK forcefield test_result.json')
                previous = json.loads(source.read_text())
                if previous.get('case_id') != case['id'] or previous.get('passed') is not False:
                    raise ValueError('replay requires an actual failed SDK result for this case')
                restored = restore_replay_evidence(json.loads((source.parent/'session_checkpoint.json').read_text()), source.parent, root)
                if not s.import_state(restored):
                    raise ValueError('failed SDK checkpoint could not be restored')
                # Apply the original real user restriction using the current
                # contract parser; do not fabricate new scientific arguments.
                s.goal_contract.apply_user_message(case['prompt'])
                s._current_line_id = line['line_id']
                report['kind'] = 'REAL_MODEL_READONLY_FORCEFIELD_CORRECTION_REPLAY'
                report['replay_source'] = str(source)
                report['answer'] = previous['answer']
            else:
                report['answer'] = s.run_until_complete(task, agent=agent, max_rounds=6, verbose=False)
            s._checkpoint('forcefield_probe_finished')
            def evaluate():
                main_calls = [c for c in s.memory.tool_call_log if c.get('agent') == 'lead-orchestrator']
                used = [c['tool'] for c in main_calls]
                query_texts = [c.get('params',{}).get('query','').casefold() for c in main_calls if c['tool']=='discover_forcefield']
                answer = normalize_evaluation_text(report['answer'])
                report['checks'] = {'real_agent_discovered_project_assets': ('validate_framework_charges' if case.get('kind') == 'charge' else 'convert_physical_units' if case.get('kind') == 'units' else 'discover_forcefield') in used,
                    'no_calculation_or_submission': not any(c['tool'].startswith(('run_', 'submit')) for c in s.memory.tool_call_log),
                    'main_owns_scientific_confirmation': s.current_agent.name == 'lead-orchestrator',
                    'requested_family_actually_queried': all(any(q.casefold() in t or t in q.casefold() for t in query_texts) for q in case['queries']),
                    'specific_atom_evidence_read': not case.get('inspect_required') or 'inspect_forcefield' in used,
                    'answer_contains_required_evidence_topics': all(any(normalize_evaluation_text(term) in answer for term in group) for group in case['terms'])}
                report['missing_evidence_topics'] = [group for group in case['terms'] if not any(normalize_evaluation_text(term) in answer for term in group)]
            evaluate()
            reviewer = Session(config=s.config, registry=registry)
            reviewer.client = reviewer.client.with_options(timeout=90,max_retries=0)
            reviewer._current_line_id = line['line_id']
            reviewer.goal_contract = copy.deepcopy(s.goal_contract)
            def review_answer(attempt):
                set_context('sdk_forcefield_probe',cid,'supervisor',line['line_id'])
                event = s.scientific_review_event(case['prompt'], report['answer'])
                if event is None:
                    raise ValueError('Main SDK did not obtain any successful scientific source evidence.')
                receipt = s.audit_scientific_answer(reviewer, event)
                write_checkpoint(root / f'supervisor_checkpoint_attempt_{attempt}.json', reviewer.export_state())
                return receipt
            review = review_answer(1)
            report['initial_attempt'] = {'answer': report['answer'], 'checks': copy.deepcopy(report['checks']),
                'model_calls': copy.deepcopy(calls), 'supervisor_receipt': copy.deepcopy(review)}
            report['correction_rounds'] = 0
            report['supervisor_verification_retries'] = 0
            final_review_attempt = 2
            state = s.context['scientific_reviews'][review['scope_id']]
            report['supervisor_verification_retries'] = state.get('verification_retries', 0)
            # A genuine structured rejection goes back through the formal SDK
            # lifecycle path. Infrastructure/budget failures are not scientific
            # correction, and human decisions must never be auto-approved.
            structured_rejection = any(c['tool']=='supervisor_decision' and c.get('params',{}).get('scientific_review',{}).get('passed') is False
                                       for c in review.get('evidence_calls', []))
            if (structured_rejection and review.get('review_action') == 'correct_main'
                    and not s._waiting_for_user_input):
                call_limit = min(9, len(calls) + 3)
                set_context('sdk_forcefield_probe',cid,'lead-orchestrator',line['line_id'])
                report['answer'] = s.resume_lifecycle_event({'event_id':'forcefield-correction-1', 'kind':'scientific_review_failed',
                    'supervisor_receipt':review,'payload':{'question':case['prompt'],'instruction':
                        '按监督指出的具体错误自主修正答案；只修正有证据的主张，保留缺项和协商边界。不新增计算、不替换力场、不扩大目标。'}})
                s._checkpoint('forcefield_correction_finished')
                evaluate()
                report['correction_rounds'] = 1
                review = review_answer(final_review_attempt)
            report['scientific_review'] = review.get('scientific_review', {'passed':False,'issues':['reviewer did not provide structured scientific verdict']})
            report['supervisor_receipt'] = review
            write_checkpoint(root / 'supervisor_checkpoint.json', reviewer.export_state())
            report['checks']['independent_scientific_review_passed'] = report['scientific_review']['passed']
            report['checks']['independent_supervisor_used_formal_tools'] = any(
                c['tool'] in {'discover_forcefield','inspect_forcefield','validate_framework_charges','convert_physical_units'}
                and (not c.get('failed') or c['call_id'] in review.get('independent_source_call_ids', [])) for c in reviewer.memory.tool_call_log)
            report['supervisor_evidence_reused_after_correction'] = report['correction_rounds'] > 0 and not any(
                c['tool'] in {'discover_forcefield','inspect_forcefield'} for c in review.get('evidence_calls', []))
            set_context('sdk_forcefield_probe',cid,'lead-orchestrator',line['line_id'])
            if case.get('kind') == 'charge':
                from agents.scientific_review import is_source_result
                measured = [json.loads(s._load_evidence_call(c)['result']) for c in s.memory.tool_call_log
                            if c['tool']=='validate_framework_charges' and is_source_result(c['tool'],json.loads(s._load_evidence_call(c)['result']))]
                report['checks']['charge_source_bytes_unchanged'] = all(__import__('hashlib').sha256(Path(c['cif_path']).read_bytes()).hexdigest()==c['sha256'] for c in measured)
            report['passed'] = all(report['checks'].values())
            report['pending_user_interaction'] = s._pending_user_interaction
    except Exception as error:
        report.update(passed=False, error=type(error).__name__+': '+str(error))
    finally:
        clear_context()
        report['build_consistent'] = report['source_sha256'] == release_source_hashes()
        if not report['build_consistent']:
            report.update(passed=False, error='Release sources changed during SDK evaluation; rerun the frozen build')
        write_checkpoint(root / 'test_result.json', report)
        print(json.dumps({'passed':report.get('passed'),'checks':report.get('checks'),'error':report.get('error'),
                          'report': str(root/'test_result.json')},ensure_ascii=False),flush=True)
    return 0 if report.get('passed') else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='opls_ambiguous')
    parser.add_argument('--suite', action='store_true')
    parser.add_argument('--replay-failed', help='Restore a retained failed SDK case and let the SDK review/correct it; no hand-written answer.')
    parser.add_argument('--workers', type=int, default=2, choices=[1,2])
    args = parser.parse_args()
    cases = json.loads((ROOT/'tests/fixtures/forcefield_agent_cases.json').read_text())['cases']
    if not args.suite:
        case = next((c for c in cases if c['id']==args.case),None)
        if not case: raise ValueError('unknown case ID')
        return run_case(case, args.replay_failed)
    if args.replay_failed:
        raise ValueError('--replay-failed is case-specific and cannot be combined with --suite')
    output = ROOT/'reports'/('forcefield_agent_suite_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    summaries = []
    frozen_sources = release_source_hashes()
    def launch(case):
        log = output/(case['id']+'.log')
        with log.open('wb') as stream:
            process = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--case',case['id']],cwd=ROOT,stdout=stream,stderr=stream,timeout=650)
        lines = log.read_text().splitlines()
        summary = next((json.loads(line) for line in reversed(lines) if line.startswith('{"passed":')),None)
        return {'case_id':case['id'],'exit_code':process.returncode,'summary':summary,'log':str(log)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(launch,c):c['id'] for c in cases}
        for future in as_completed(futures):
            try: row = future.result()
            except Exception as error: row={'case_id':futures[future],'error':str(error),'summary':None}
            summaries.append(row)
            write_checkpoint(output/'suite_result.json',{'cases':summaries,'complete':False})
            print(json.dumps(row,ensure_ascii=False),flush=True)
    consistent = frozen_sources == release_source_hashes()
    passed = consistent and all(row.get('summary',{}).get('passed') for row in summaries if row.get('summary')) and all(row.get('summary') for row in summaries)
    write_checkpoint(output/'suite_result.json',{'cases':summaries,'complete':True,'passed':bool(passed),
        'build_consistent': consistent, 'source_sha256': frozen_sources,
        'evaluation_note':'Topic/trace checks are smoke assertions, not a complete scientific-validity assessment; individual answers require review.'})
    print(json.dumps({'passed':bool(passed),'report':str(output/'suite_result.json')},ensure_ascii=False),flush=True)
    return 0 if passed else 1


if __name__ == '__main__': raise SystemExit(main())
