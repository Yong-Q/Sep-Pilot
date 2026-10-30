"""Conservative resource controls derived only from authenticated goal text.

Model tool arguments are never evidence of a user limit. Unrecognized phrasing
does not create a limit. The original directive digest remains auditable.
"""
import hashlib
import re

_TOOLS = {'run_cdft': r'cDFT', 'run_md_optimize': r'(?<![a-z])MD(?![a-z])|LAMMPS|分子动力学|NPT',
          'run_gcmc_batch': r'GCMC|蒙特卡洛', 'run_gcmc_isotherm': r'GCMC|蒙特卡洛',
          'run_string_tst': r'String|TST', 'run_pore_analysis': r'Zeo|孔隙'}
_CONTROL = r'(?:只用|仅用|固定(?:使用|为|用)?|必须使用|使用|不超过|最多(?:使用|用)?|至多(?:使用|用)?|改为|改成|调整为|调整到|设为|设置为|用|\buse(?:\s+exactly)?\s+|\bat most\s+|\bup to\s+)'


def resource_intent(goal, nodes):
    """Return step-scoped hard controls; no model-supplied provenance accepted.

    Explicit numeric CPU/RAM controls and explicit node/partition restrictions
    are supported. A later automatic-CPU directive releases an old CPU limit.
    Ambiguous suggestions and questions do not establish a hard constraint.
    """
    from .goal_contract import _is_automated_message
    messages = list(getattr(goal, 'directives', []) or [])
    original = getattr(goal, 'original_goal', '')
    if original and (not messages or messages[0] != original):
        messages.insert(0, original)
    numeric, pins = {}, {}
    for index, message in enumerate(messages):
        if not isinstance(message, str) or _is_automated_message(message):
            continue
        basis = 'authenticated_goal:' + str(index) + ':' + hashlib.sha256(message.encode()).hexdigest()
        message_tools = {tool for tool, pattern in _TOOLS.items() if re.search(pattern, message, re.I)}
        for clause in re.split(r'[，,。；;\n]', message):
            if re.search(r'不要|不用|不是|不需要|禁止|不准|不应|为什么|为何|怎么|是否|[？?]|\b(?:why|should|do not|don.t)\b', clause, re.I):
                continue
            tools = {tool for tool, pattern in _TOOLS.items() if re.search(pattern, clause, re.I)}
            if not tools and len(message_tools) == 1:
                tools = message_tools
            targets = [n['step_id'] for n in nodes if not tools or n.get('tool') in tools]
            if re.search(r'(?:核数|CPU|并行数|资源).{0,12}自动|(?:自动|智能(?:化)?)调度', clause, re.I):
                for key in list(numeric):
                    if key[0] in targets and key[1] in {'cpus','workers'}:
                        numeric.pop(key)
            cpu = re.search(_CONTROL + r'\s*(\d+)\s*(?:个\s*)?(?:核|CPUs?|cores?|MPI\s*(?:进程|ranks?))(?![a-z])', clause, re.I)
            if cpu:
                kind = 'max' if re.search(r'最多|至多|不超过|上限|at most|up to', clause, re.I) else 'exact'
                for step in targets:
                    for key in list(numeric):
                        if key[:2] == (step,'cpus'):
                            numeric.pop(key)
                    numeric[step,'cpus',kind] = {'step_id':step,'source':'user','field':'cpus',
                        'kind':kind,'value':int(cpu[1]),'basis':basis}
            memory = re.search(r'(?:(?:内存|RAM|memory)\s*(?:最多|至多|不超过|上限|为|:|：|=)?\s*|'
                               + _CONTROL + r'\s*)(\d+(?:\.\d+)?)\s*(GiB|GB|MiB|MB)', clause, re.I)
            if memory and (re.search(r'内存|RAM|memory',clause,re.I)):
                kind = 'max' if re.search(r'最多|至多|不超过|上限|at most|up to', clause, re.I) else 'exact'
                scale = 1024 if memory[2].lower() in {'gib','gb'} else 1
                value = float(memory[1])*scale
                if value.is_integer():
                    for step in targets:
                        for key in list(numeric):
                            if key[:2] == (step,'memory_mb'):
                                numeric.pop(key)
                        numeric[step,'memory_mb',kind] = {'step_id':step,'source':'user',
                            'field':'memory_mb','kind':kind,'value':int(value),'basis':basis}
            for field, marker in (('nodelist',r'节点|nodes?'),('partition',r'分区|partition')):
                match = re.search(r'(?:限定|指定|只(?:能)?(?:使用|用)|固定(?:使用|为)?|必须使用)\s*(?:在|到)?\s*'
                                  + r'(?:'+marker+r')\s*[:：=]?\s*([a-z][a-z0-9_-]*(?:(?:或|或者|、|\s+or\s+|\s*,\s*)[a-z][a-z0-9_-]*)*)', clause,re.I)
                if match:
                    values = re.split(r'或者|或|、|\s+or\s+|\s*,\s*',match[1],flags=re.I)
                    for step in targets:
                        pins.setdefault(step,{})[field] = values
    return {'resource_constraints':list(numeric.values()), 'resource_pins':pins}
