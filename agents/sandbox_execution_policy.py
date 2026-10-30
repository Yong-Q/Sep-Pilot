"""Honor an explicit user isolation requirement at plan and execution gates."""
import re
import shlex
from pathlib import Path

_SUBJECT=r'(?:沙盒|隔离环境|隔离执行|isolated\s+(?:environment|sandbox)|sandbox)'
_REVOKE=re.compile(r'(?:不要求|不需要|无需|不用|不使用|不采用|禁止使用)\s*(?:使用|采用|在)?\s*'+_SUBJECT+r'|\b(?:do\s+not|don\x27t|no\s+need\s+to)\s+(?:use\s+)?(?:the\s+)?sandbox\b',re.I)
_REQUIRE=re.compile(r'(?:请|必须|需要|使用|采用|通过|在|放进|交给|\buse\b|\bin\b|\bwithin\b)\s*(?:的|到|要|an?\b|the\b)?\s*'+_SUBJECT+
    r'|'+_SUBJECT+r'\s*(?:中|里|内|执行|运行|处理|分析|编写|读取|作图|绘图|only|execute|run)',re.I)

def sandbox_requested(goal):
    """Only explicit user wording asserts a sandbox requirement; no model grant."""
    data=goal.to_dict() if hasattr(goal,'to_dict') else (goal or {})
    texts=[data.get('original_goal') or '']+list(data.get('directives') or [])
    if not texts[-1] and data.get('active_goal'):texts.append(data['active_goal'])
    required=False
    for text in texts:
        for clause in re.split(r'[，,。；;\n]',str(text)):
            if _REVOKE.search(clause):required=False
            elif _REQUIRE.search(clause):required=True
    return required

_SAFE={'pwd','ls','stat','head','tail','wc','cat','sha256sum','squeue','sacct','sinfo'}
_SAFE_OPTIONS=re.compile(r'^-[A-Za-z0-9]+$')

def safe_readonly_shell(command):
    """Permit simple diagnostics, never shell composition or code interpreters."""
    if not isinstance(command,str) or not command.strip() or len(command)>4096 or re.search(r'[\n\r\x00;&|<>`$()]',command):return False
    try:parts=shlex.split(command,posix=True)
    except ValueError:return False
    if not parts:return False
    exe=Path(parts[0]).name
    if exe not in _SAFE or (parts[0]!=exe and parts[0] not in {'/bin/'+exe,'/usr/bin/'+exe}):return False
    for part in parts[1:]:
        if part.startswith('-') and not _SAFE_OPTIONS.fullmatch(part):return False
    return True

def sandbox_violation(goal,tool,arguments):
    if not sandbox_requested(goal):return ''
    arguments=arguments or {}
    if tool=='run_bash':
        if safe_readonly_shell(arguments.get('command','')):return ''
    elif tool=='write_file':
        path=arguments.get('path')
        if not isinstance(path,str) or Path(path).suffix.lower() not in {'.csv','.json','.tsv','.txt','.dat','.data'}:return ''
    else:return ''
    return ('用户要求隔离环境；此操作将在隔离外执行代码或写入数据文件，未执行。'
            '请查询 run_analysis_sandbox 的实际 schema，由 analyst/communicator 在隔离环境处理当前会话输入，'
            '声明新输出并按正常 DAG 绑定和校验；只读状态检查可使用专门工具。')

def validate_sandbox_changes(goal,changes):
    from .workflow_patch import WorkflowContractError
    for change in changes:
        node=change.get('node') or {}
        violation=sandbox_violation(goal,node.get('tool'),node.get('arguments') or {})
        if change.get('operation')=='upsert' and violation:
            raise WorkflowContractError('sandbox-required workflow cannot use a non-isolated code command',{
                'step_id':change.get('step_id'),'contract_code':'sandbox_required','executed':False,
                'next_action':violation})
