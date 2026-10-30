"""Fresh registered sandbox contracts and bounded, advisory path validation."""
import ast
import json
from pathlib import PurePosixPath


def current_sandbox_contract(registry,goal,agent_name):
    from .sandbox_execution_policy import sandbox_requested
    if agent_name!='lead-orchestrator' or not sandbox_requested(goal):return ''
    sandbox=registry.get('run_analysis_sandbox');patch=registry.get('propose_workflow_patch')
    if sandbox is None or patch is None:return ''
    try:bindings=patch.input_schema['properties']['changes']['items']['properties']['node']['properties']['input_bindings']
    except (KeyError,TypeError):return ''
    value={'tool':sandbox.to_claude_schema(),'input_bindings_schema':bindings,
        'scope':'Current registry contract, not an execution grant. Bind a source-derived scalar only when the user approved that dataflow; never invent evidence or override explicit conditions.'}
    return '\n[CURRENT SANDBOX DATA CONTRACT]\n'+json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n'


def obvious_sandbox_path_issues(script):
    """Catch only statically obvious file operations; sandbox remains authoritative."""
    if not isinstance(script,str):return []
    try:tree=ast.parse(script)
    except (SyntaxError,ValueError):return []
    values={}
    stores={}
    for item in ast.walk(tree):
        if isinstance(item,ast.Name) and isinstance(item.ctx,(ast.Store,ast.Del)):
            stores[item.id]=stores.get(item.id,0)+1
        elif isinstance(item,ast.arg):stores[item.arg]=stores.get(item.arg,0)+1
    def literal(node):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):return node.value
        if isinstance(node,ast.Name):return values.get(node.id)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Path' and len(node.args)==1:return literal(node.args[0])
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
            left,right=literal(node.left),literal(node.right)
            if isinstance(left,str) and isinstance(right,str):return str(PurePosixPath(left)/right)
        return None
    for item in tree.body:
        if (isinstance(item,ast.Assign) and len(item.targets)==1 and isinstance(item.targets[0],ast.Name)
                and stores.get(item.targets[0].id)==1):
            value=literal(item.value)
            if isinstance(value,str):values[item.targets[0].id]=value
    issues=[];guarded=set()
    for item in ast.walk(tree):
        if isinstance(item,ast.Try) and any(
                handler.type is None or isinstance(handler.type,ast.Name)
                and handler.type.id in {'OSError','PermissionError','Exception'} for handler in item.handlers):
            for body_item in item.body:guarded.update(id(child) for child in ast.walk(body_item))
    operations={'open','read_text','read_bytes','write_text','write_bytes','mkdir','unlink','rename','replace'}
    for item in ast.walk(tree):
        if not isinstance(item,ast.Call) or id(item) in guarded:continue
        function=item.func
        name=function.id if isinstance(function,ast.Name) else function.attr if isinstance(function,ast.Attribute) else ''
        if name not in operations:continue
        path=(literal(function.value) if isinstance(function,ast.Attribute) and name!='open' else
              literal(item.args[0]) if item.args else None)
        if not isinstance(path,str) or not path.startswith('/'):continue
        p=PurePosixPath(path)
        if p==PurePosixPath('/output/artifacts') or PurePosixPath('/output/artifacts') in p.parents:
            issue='sandbox output files must be written directly under /output; the host publishes the artifacts directory'
        elif p==PurePosixPath('/output') or PurePosixPath('/output') in p.parents or p==PurePosixPath('/inputs') or PurePosixPath('/inputs') in p.parents or p==PurePosixPath('/tmp') or PurePosixPath('/tmp') in p.parents:
            continue
        elif any(p==PurePosixPath(prefix) or PurePosixPath(prefix) in p.parents for prefix in ('/usr','/lib','/lib64','/dev')):
            # These runtime mounts can exist in the namespace; permissions
            # and the isolated worker remain authoritative for each access.
            continue
        else:
            issue='host absolute paths are unavailable inside the sandbox; use the selected /inputs path'
        if issue not in issues:issues.append(issue)
    return issues
