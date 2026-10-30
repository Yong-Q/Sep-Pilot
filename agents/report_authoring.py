"""Evidence packet -> bounded scientific author -> reviewed immutable report.

Experimental integration candidate. It never runs a scientific calculation,
changes a workflow status, or silently treats a submission receipt as data.
The caller owns current workflow/goal/token control and the native checkpoint
witness. Model callbacks are server capabilities, never model tool parameters.
"""
import copy
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid
from urllib.parse import quote, urlsplit, parse_qsl

MAX_FILE = 2 * 1024 * 1024
MAX_TOTAL = 16 * 1024 * 1024
MAX_TABLE_ROWS = 2000
MAX_PACKET = 384 * 1024
MAX_DRAFT = 192 * 1024
NUMBER = re.compile(r'(?<![\w])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w])')
PROVENANCE_COLUMN = re.compile(r'(?:_path|_sha256|_source_csv|_json)$', re.I)
AUTHOR_SCHEMA={'type':'object','required':['draft'],'properties':{'draft':{'type':'object',
    'required':['title','blocks'],'properties':{'title':{'type':'string'},'blocks':{'type':'array',
    'minItems':1,'items':{'oneOf':[
        {'type':'object','required':['kind','text','source_ids','claim_type'],'properties':{
            'kind':{'const':'narrative'},'text':{'type':'string','description':'Narrative prose. For a data_ref id a, only {{a.identity}}, {{a.value}}, {{a.unit}} and {{a.column}} tokens exist. Numeric measured/derived prose must bind both identity and value. Table/figure/hash macros do not exist. Use table and figure blocks for complete source data rather than copying metadata into prose.'},'source_ids':{'type':'array','items':{'type':'string'}},
            'reference_ids':{'type':'array','items':{'type':'string'}},
            'data_refs':{'type':'array','description':'Bindings for numeric prose. Each id is a short identifier such as a. Bind each row identity with {{a.identity}} and its quantity with {{a.value}}; additional quantities/uncertainty from the same exact row may share that preceding identity anchor until another row identity appears. Do not put values, material names, hashes or unbound numeric statistics directly in measured prose.',
                'items':{'type':'object','additionalProperties':False,'required':['id','table_id','row_selector','column'],
                'properties':{'id':{'type':'string','pattern':'^[A-Za-z][A-Za-z0-9_]{0,15}$'},'table_id':{'type':'string'},
                    'row_selector':{'type':'object','minProperties':1,'description':'Exact existing string cell values identifying one unique table row.', 'additionalProperties':{'type':'string'}},
                    'column':{'type':'string'},'format':{'type':'string','pattern':'^\\.(?:[0-9]|1[0-2])[fge]$'}}}},
            'claim_type':{'enum':['measured','derived','hypothesis','external','recommendation',
                                   'limitation','method','result','inference']}}},
        {'type':'object','required':['kind','table_id'],'properties':{'kind':{'const':'table'},
            'table_id':{'type':'string'},'columns':{'type':'array','items':{'type':'string'}}}},
        {'type':'object','required':['kind','reference_id'],'properties':{'kind':{'const':'citation'},'reference_id':{'type':'string'}}},
        {'type':'object','required':['kind','figure_id'],'properties':{'kind':{'const':'figure'},'figure_id':{'type':'string'}}},
        {'type':'object','required':['kind','text'],'properties':{'kind':{'const':'heading'},'text':{'type':'string'}}}
    ]}}}}}}
REVIEW_SCHEMA={'type':'object','required':['review'],'properties':{'review':{'type':'object',
    'required':['passed','draft_sha256','checked_blocks','blocking_issues'],'properties':{
        'passed':{'type':'boolean'},'draft_sha256':{'type':'string'},
        'checked_blocks':{'type':'array','items':{'type':'integer'}},
        'blocking_issues':{'type':'array','maxItems':128,
            'description':'Only actual defects that prevent publication. Must be empty when passed is true.',
            'items':{'type':'string'}},
        'notes':{'type':'array','maxItems':128,
            'description':'Non-blocking observations or minor style suggestions, separate from blocking defects.',
            'items':{'type':'string'}}}}}}
MEASUREMENT_SCHEMA={'type':'object','additionalProperties':False,
    'required':['kind','table_id','row_selector','columns'],
    'description':'Optional source-row result paragraph. Select quantities and associated metadata columns; native code renders their exact original identity, values and units. Formatting applies only to finite numeric cells. Missing/nonfinite cells remain explicitly unavailable. No text templates or value transcription.',
    'properties':{'kind':{'const':'measurement'},'table_id':{'type':'string'},
        'row_selector':{'type':'object','minProperties':1,
            'additionalProperties':{'type':'string'}},
        'columns':{'type':'array','minItems':1,'uniqueItems':True,'items':{'type':'string'},
            'description':'Select the relevant existing public columns from the source row.'},
        'format':{'type':'string','pattern':'^\\.(?:[0-9]|1[0-2])[fge]$'}}}
FACT_SCHEMA={'type':'object','additionalProperties':False,
    'required':['kind','source_id','document_id','pointer'],
    'description':'Optional status/coverage fact or separately typed scheduler provenance from an owned JSON result. Select one existing scalar with RFC6901 JSON Pointer; native code renders its exact value. Job IDs/resource requests belong in provenance facts, never scientific numerical prose.',
    'properties':{'kind':{'const':'fact'},'source_id':{'type':'string'},
        'document_id':{'type':'string'},'pointer':{'type':'string','maxLength':1024}}}


def author_stage_schema(packet=None):
    """Expose source-cell selection, not an application-specific text DSL."""
    schema=copy.deepcopy(AUTHOR_SCHEMA)
    draft=schema['properties']['draft'];draft['required']=['blocks']
    draft['properties'].pop('title',None)
    variants=draft['properties']['blocks']['items']['oneOf']
    narrative=variants[0]['properties']
    narrative.pop('data_refs')
    # New actors express scalar observations through native fact/row blocks.
    # Legacy measured/derived narrative binding remains readable, but is not a
    # second private numerical transcription protocol for new model output.
    narrative['claim_type']['enum']=['method','limitation','hypothesis','inference','external','recommendation']
    narrative['text']['description']=('Scientific interpretation, methods and limitations. '
        'Quantitative values are written by table or optional measurement blocks. '
        'Do not transcribe measurement numbers, use template macros or repeat metadata in prose. '
        'Never place URLs, Markdown links or local paths in narrative. Use an available citation block '
        'for verified references; owned native source documents are linked automatically.')
    variants.append(copy.deepcopy(MEASUREMENT_SCHEMA))
    variants.append(copy.deepcopy(FACT_SCHEMA))
    native_sources=[source['id']for source in(packet or{}).get('sources',[])
        if source.get('native_document')and any(block.get('anchor')and block.get('untrusted_content')
            for block in source['native_document'].get('blocks',[]))]
    if native_sources:
        from .literature_report_bridge import EXCERPT_SCHEMA
        excerpt=copy.deepcopy(EXCERPT_SCHEMA)
        excerpt['properties']['source_id']['enum']=native_sources
        variants.append(excerpt)
        excerpt_ref=copy.deepcopy(excerpt)
        excerpt_ref['required'].remove('kind');excerpt_ref['properties'].pop('kind')
        narrative['excerpt_refs']={'type':'array','items':excerpt_ref,
            'description':'Exact native excerpts supporting source-content interpretations. Source origin and scientific claims remain unverified; do not cite declared URLs as proof.'}
    else:
        # An owned JSON/log/PNG is not an extracted native document page.
        # New actors cannot invent excerpt refs; legacy rendering is unchanged.
        variants[0]['additionalProperties']=False
    if packet is not None:
        legal=list(source_aliases(packet))
        variants[0]['properties']['source_ids']['items']['enum']=legal
        variants[0]['properties']['source_ids']['description']='Known native source IDs or owned artifact IDs; artifact citations resolve to their native producer source.'
        available_tables=[table['id']for table in packet.get('tables',[])]
        available_figures=[figure['id']for figure in packet.get('figures',[])]
        available_references=[reference['id']for reference in packet.get('references',[])]
        reference_ids=variants[0]['properties']['reference_ids']
        reference_ids['description']='Literature reference IDs only; figures use figure blocks or source_ids.'
        if available_references:
            reference_ids['items']['enum']=available_references
        else:
            reference_ids['maxItems']=0
        available_documents=[document['id']for source in packet['sources']for document in source.get('structured_data',[])if document.get('id')]
        kept=[]
        for variant in variants:
            properties=variant['properties'];kind=properties['kind']['const']
            if kind in{'table','measurement'}:
                if not available_tables:continue
                properties['table_id']['enum']=available_tables
            if kind=='figure':
                if not available_figures:continue
                properties['figure_id']['enum']=available_figures
            if kind=='citation':
                if not available_references:continue
                properties['reference_id']['enum']=available_references
            if kind=='fact':
                if not available_documents:continue
                properties['document_id']['enum']=available_documents
                properties['source_id']['enum']=legal
                from .report_fact_diagnostics import fact_pointer_catalogue
                properties['pointer']['description']=(
                    'Exact RFC6901 pointer relative to that document.untrusted_content. '
                    'Do not prepend json_root_pointer. Missing fields stay missing; never '
                    'guess a name or substitute another status. Bounded key-only examples '
                    '(not exhaustive, no source values): '+json.dumps(fact_pointer_catalogue(packet),ensure_ascii=False))
            kept.append(variant)
        draft['properties']['blocks']['items']['oneOf']=kept
    return schema


def source_aliases(packet):
    """One owned citation namespace for native sources and their artifacts."""
    result={source['id']:source['id']for source in packet['sources']}
    def add(name,source):
        if not isinstance(name,str)or source not in result.values():raise ValueError('invalid owned citation namespace')
        if name in result and result[name]!=source:raise ValueError('ambiguous owned artifact citation ID')
        result[name]=source
    for table in packet.get('tables',[]):add(table['id'],table['source_id'])
    for figure in packet.get('figures',[]):add(figure['id'],figure['source_id'])
    for source in packet['sources']:
        for group in('structured_data','text_documents'):
            for document in source.get(group,[]):
                if document.get('id'):add(document['id'],source['id'])
    return result


def resolved_source_ids(packet,ids):
    if not isinstance(ids,list)or not ids or any(not isinstance(value,str)for value in ids):
        raise ValueError('actual source references require known owned source/artifact IDs')
    aliases=source_aliases(packet)
    if not set(ids)<=set(aliases):raise ValueError('actual source references require known owned source/artifact IDs')
    return list(dict.fromkeys(aliases[value]for value in ids))


class ReportStageBudgetExceeded(ValueError):
    """The shared bounded model-call budget is exhausted; retain partial artifacts."""


class ReportCapabilityGap(ValueError):
    def __init__(self,message,**details):
        super().__init__(message)
        self.details={'error_kind':'report_capability_gap','executed':False,
            'scientific_acceptance':False,'retry_authorized':False,**details}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _control(check):
    if check is not None and check() is not True:
        raise ValueError('report control was superseded or cancelled')


def _relative(root, value):
    root = Path(root).absolute()
    path = Path(value)
    if not path.is_absolute(): path = root / path
    # A lexical check precedes descriptor traversal; resolve must not hide a
    # symlink that happens to point back into the same session.
    try: rel = path.relative_to(root)
    except ValueError: raise PermissionError('report source is outside the owned session')
    if not rel.parts or '..' in rel.parts or any(part in {'', '.'} for part in rel.parts):
        raise PermissionError('invalid report source path')
    return rel


def _open_parent(root, relative, *, create=False):
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents) if path != Path('/')):
        raise PermissionError('owned root contains a symlink')
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in relative.parts[:-1]:
            try: next_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not create: raise
                os.mkdir(name, 0o700, dir_fd=fd)
                next_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = next_fd
        return fd, relative.name
    except BaseException:
        os.close(fd); raise


def _read(root, path, budget):
    rel = _relative(root, path)
    parent, name = _open_parent(root, rel)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or before.st_size < 0):
                raise ValueError('report source is not bounded owned regular data')
            if before.st_size > MAX_FILE:
                raise ValueError('report source exceeds the in-memory file budget; use the data sandbox to read the owned source in bounded chunks and create a traceable task-relevant result, or request suitable resources; no source data was silently truncated')
            if budget['bytes'] + before.st_size > MAX_TOTAL:
                raise ValueError('report source byte budget exceeded; use the data sandbox to process owned sources in bounded chunks, or request suitable resources; no source data was silently truncated')
            with os.fdopen(os.dup(fd), 'rb') as stream: raw = stream.read(MAX_FILE + 1)
            after = os.fstat(fd); linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
            identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_mode)
            if len(raw) != before.st_size or identity(before) != identity(after) or identity(after) != identity(linked):
                raise ValueError('report source changed while reading')
        finally: os.close(fd)
    finally: os.close(parent)
    budget['bytes'] += len(raw)
    return raw, {'path':str(rel), 'size':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(),
                 'device':before.st_dev, 'inode':before.st_ino,
                 'mtime_ns':before.st_mtime_ns, 'ctime_ns':before.st_ctime_ns}


def _result(value):
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict): raise ValueError('native result must be an object')
    return parsed


def _references(value, *, native_tool=None):
    """Only bibliographic identifiers actually present in the native result.

    A title or DOI is a citation candidate, not support for a quantitative
    scientific claim. Verbatim supplied abstracts/snippets stay clearly scoped.
    """
    result = []
    if native_tool not in {'query_literature','read_scientific_document','download_scientific_file'}:return result
    entries = value.get('results') or []
    if not isinstance(entries, list): return result
    if not entries and native_tool in {'read_scientific_document','download_scientific_file'}:
        # Preserve the task-provided URL as bibliographic metadata only. This
        # does not verify source origin or support a scientific claim.
        url=value.get('source_url_declared') or value.get('source_url') or value.get('final_url')
        if isinstance(url,str) and url.startswith('https://'):
            entries=[{'url':url,'title':value.get('title') or url}]

    for row in entries[:64]:
        if not isinstance(row, dict): continue
        doi = str(row.get('doi') or '').strip()
        link = str(row.get('url') or '').strip()
        if doi and re.fullmatch(r'10\.\d{4,9}/[^\s<>]+', doi): link = 'https://doi.org/' + doi
        if not link.startswith('https://') or any(x in link for x in ('\n', '\r', '<', '>')): continue
        try:
            parsed=urlsplit(link)
            secret_names={'token','access_token','api_key','apikey','key','password','secret','signature','sig','credential','authorization','x-amz-signature'}
            if (parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password
                    or any(key.lower() in secret_names for key,_ in parse_qsl(parsed.query,keep_blank_values=True)+parse_qsl(parsed.fragment,keep_blank_values=True))):continue
        except ValueError:continue
        title = str(row.get('title') or '').strip()
        # An already obtained DOI or link is itself an honest label; never
        # invent a missing paper title or require another network lookup.
        if not title:title=doi if doi and link=='https://doi.org/'+doi else link
        snippet = str(row.get('abstract') or row.get('snippet') or '')[:8000]
        reference={'title':title[:1000], 'url':link, 'snippet':snippet,
            'support_kind':'supplied_excerpt' if snippet and native_tool in {'query_literature','read_scientific_document'} else 'bibliographic_only'}
        result.append(reference)
    return result


def _user_goal_bibliography(goal):
    """Use only literal identifiers from the original user goal, no network."""
    text=goal.get('original_goal') if isinstance(goal,dict) else None
    if not isinstance(text,str) or len(text)>200000:return []
    rows=[]
    for match in re.finditer(r'https://[^\s<>\"\']{1,2048}|(?<![A-Za-z0-9])10\.\d{4,9}/[^\s<>\"\']{1,512}',text):
        value=match.group(0).rstrip('.,;:，。；：)]}）】')
        if value.startswith('https://'):rows.append({'url':value})
        else:rows.append({'doi':value})
        if len(rows)>=64:break
    # Reuse the task-source syntax/privacy filter, but do not assert origin
    # verification, content evidence, or a title the user did not provide.
    refs=_references({'results':rows},native_tool='query_literature')
    unique={}
    for ref in refs:unique.setdefault(ref['url'],ref)
    return list(unique.values())


def source_receipts(packet):
    """Native producer identities, never values supplied by a model."""
    return [{key:source.get(key) for key in ('step_id','token','attempt_id',
        'execution_fingerprint','call_id','source_sha256','contract_sha256')}
        for source in packet['sources']]


def _validate_source_receipts(root, state, result, *, attest, budget, file_facts,
                              cache, trail=()):
    """Bind an authored intermediate to the *current* original producers.

    File equality alone is insufficient: a new source attempt can legitimately
    produce identical bytes. A stale fragment must not erase that distinction.
    """
    records=result.get('source_receipts')
    legacy=False
    if records is None:
        if result.get('source_step') and len(result.get('source_steps')or[])==1:
            legacy=True
            records=[{'step_id':result['source_step'],
                'call_id':result.get('source_evidence_call_id'),
                'execution_fingerprint':result.get('source_attempt_fingerprint'),
                'source_sha256':result.get('evidence_fingerprint')}]
        else:
            raise ReportCapabilityGap('authored intermediate lacks original producer identities',
                required_action='reauthor_in_new_immutable_output')
    if (not isinstance(records,list) or not 1<=len(records)<=16
            or any(not isinstance(record,dict)for record in records)
            or len({record.get('step_id')for record in records})!=len(records)
            or {record.get('step_id')for record in records}!=set(result.get('source_steps')or[])):
        raise ValueError('authored intermediate source identities differ from declared dependencies')
    from agents.workflow_patch import execution_contract
    for record in records:
        step=record.get('step_id')
        if not isinstance(step,str)or step in trail or len(trail)>=16:
            raise ValueError('authored intermediate has cyclic or excessive provenance')
        node=state.get('nodes',{}).get(step,{})
        ref=node.get('result_ref')or{};contract=node.get('contract')or{}
        keys=('execution_fingerprint',)if legacy else('token','attempt_id','execution_fingerprint')
        if (node.get('status')!='succeeded' or not node.get('token')
                or any(record.get(key)!=node.get(key)for key in keys)
                or not legacy and (not isinstance(record.get('token'),str)or not record['token'])
                or record.get('call_id')!=ref.get('call_id')
                or not re.fullmatch('[a-f0-9]{64}',str(record.get('source_sha256','')))
                or record.get('contract_sha256')is not None and
                    record['contract_sha256']!=_digest(execution_contract(contract))):
            raise ValueError('stale source attempt: authored intermediate original source attempt changed')
        call_id=record['call_id']
        if not isinstance(call_id,str)or not re.fullmatch('[a-zA-Z0-9_-]{1,128}',call_id):
            raise ValueError('authored intermediate has no native source call')
        expected=Path(root)/'evidence'/(call_id+'.json')
        if ref.get('evidence_path')!=str(expected):
            raise PermissionError('authored intermediate source evidence escaped owner')
        identity=_digest({'step_id':step,'token':node.get('token'),'attempt_id':node.get('attempt_id'),
            'execution_fingerprint':node.get('execution_fingerprint'),'call_id':call_id,
            'source_sha256':record['source_sha256'],'contract_sha256':_digest(execution_contract(contract))})
        if identity in cache:continue
        raw,fact=_read(root,expected,budget);call=json.loads(raw)
        if (fact['sha256']!=record['source_sha256']or call.get('call_id')!=call_id
                or call.get('tool')!=contract.get('tool')or call.get('failed')is not False
                or call.get('params')!=(node.get('effective_arguments')or contract.get('arguments')or{})
                or _result(call.get('result'))!=_result(node.get('result'))):
            raise ValueError('authored intermediate original source evidence changed')
        witness=attest(node,call,raw)
        if not isinstance(witness,dict)or witness.get('checkpoint_verified')is not True:
            raise ValueError('authored intermediate original producer checkpoint is missing')
        checkpoint=witness.get('checkpoint_path')or ref.get('checkpoint_path')
        _,checkpoint_fact=_read(root,checkpoint,budget)
        if checkpoint_fact['sha256']!=witness.get('checkpoint_sha256'):
            raise ValueError('authored intermediate original producer checkpoint changed')
        file_facts.extend([fact,checkpoint_fact]);cache.add(identity)
        original=_result(call['result'])
        if contract.get('tool')in{'generate_scientific_report','generate_result_figures'}:
            _validate_source_receipts(root,state,original,attest=attest,budget=budget,
                file_facts=file_facts,cache=cache,trail=(*trail,step))


def build_result_packet(root, owner, state, report_step, *, attest, control_valid=None, job_lookup=None):
    """Read actual completed artifacts, not just source tool return text.

    attest(node, call, raw) must be a trusted native callback validating the
    original producer/checkpoint/attempt. The report cannot supply this callback.
    """
    _control(control_valid)
    if (state.get('username'), state.get('conv_id')) != (owner.get('username'), owner.get('conv_id')):
        raise PermissionError('report workflow belongs to a different owner')
    root = Path(root).absolute()
    if (root.parent.name, root.name) != (owner.get('username'), owner.get('conv_id')):
        raise PermissionError('report root owner mismatch')
    if not callable(attest): raise ValueError('native producer witness is required')
    node = state.get('nodes', {}).get(report_step, {})
    contract = node.get('contract') or {}
    args = contract.get('arguments') or {}
    source_steps = args.get('source_steps') or []
    tool=contract.get('tool');agent=contract.get('agent')
    if (tool not in {'generate_scientific_report','generate_result_figures'}
            or (tool=='generate_scientific_report'and agent!='communicator')
            or (tool=='generate_result_figures'and agent not in {'analyst','communicator'})
            or not source_steps or len(set(source_steps)) != len(source_steps)
            or not set(source_steps) <= set(contract.get('depends_on') or [])):
        raise ValueError('report must use its declared direct successful dependencies')
    budget = {'bytes':0}; file_facts = []; sources = []; tables = []; references = []; figures = []
    provenance_cache=set();visual_cache={}
    for index, step in enumerate(source_steps, 1):
        _control(control_valid)
        source = state['nodes'].get(step, {})
        if source.get('status') != 'succeeded': raise ValueError('report source is not completed')
        source_id = 'S' + str(index)
        ref = source.get('result_ref') or {}
        call_id = ref.get('call_id')
        if not isinstance(call_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', call_id):
            raise ValueError('source lacks a native producer')
        expected = root / 'evidence' / (call_id + '.json')
        if Path(ref.get('evidence_path') or '') != expected:
            raise PermissionError('producer evidence path does not match owned call ID')
        raw, fact = _read(root, expected, budget); call = json.loads(raw)
        source_contract = source.get('contract') or {}
        actual_args = source.get('effective_arguments') or source_contract.get('arguments') or {}
        if (call.get('call_id') != call_id or call.get('tool') != source_contract.get('tool')
                or call.get('failed') is not False or call.get('params') != actual_args
                or _result(call.get('result')) != _result(source.get('result'))):
            raise ValueError('producer does not attest the current source attempt')
        witness = attest(source, call, raw)
        if not isinstance(witness, dict) or witness.get('checkpoint_verified') is not True:
            raise ValueError('producer checkpoint witness is missing')
        file_facts.append(fact)
        checkpoint_path=witness.get('checkpoint_path')or ref.get('checkpoint_path')
        if not checkpoint_path:raise ValueError('producer checkpoint fact is missing')
        _,checkpoint_fact=_read(root,checkpoint_path,budget)
        if checkpoint_fact['sha256']!=witness.get('checkpoint_sha256'):
            raise ValueError('producer checkpoint changed after native witness')
        file_facts.append(checkpoint_fact)
        for extra in ('fresh_inspection_evidence','fresh_inspection_checkpoint'):
            expected_fact=witness.get(extra)
            if expected_fact is not None:
                if not isinstance(expected_fact,dict):raise ValueError('invalid fresh inspection witness')
                _,actual_fact=_read(root,expected_fact['path'],budget)
                if actual_fact['sha256']!=expected_fact['sha256']:
                    raise ValueError('fresh source inspection changed before report packet')
                file_facts.append(actual_fact)
        native_result = _result(call['result'])
        authored_source=source_contract.get('tool')=='generate_scientific_report'
        from .report_validation_receipt import report_validation_satisfied
        source_requirements=(source_contract.get('arguments')or{}).get('report_requirements')or{}
        from .report_review_intent import explicit_strict_review
        if authored_source and (not report_validation_satisfied(native_result,strict_required=(source_requirements.get('independent_review') is True or explicit_strict_review(state.get('goal_contract')or{})))
                or not native_result.get('report_sha256')):
            raise ReportCapabilityGap('assembly source is an evidence index, not an authored report',
                source_step=step,required_action='reauthor_source_in_new_immutable_output')
        if authored_source or source_contract.get('tool')=='generate_result_figures':
            _validate_source_receipts(root,state,native_result,attest=attest,budget=budget,
                file_facts=file_facts,cache=provenance_cache,trail=(step,))
        from agents.workflow_patch import execution_contract
        summary = {'id':source_id, 'step_id':step, 'tool':call['tool'], 'call_id':call_id,
            'token':source.get('token'), 'attempt_id':source.get('attempt_id'),
            'execution_fingerprint':source.get('execution_fingerprint'),
            'contract_sha256':_digest(execution_contract(source_contract)),
            'job_ids':source.get('job_ids') or [], 'method':native_result.get('method'),
            'source_sha256':fact['sha256'], 'checkpoint_witness':witness,
            'completion_status':source['status'], 'scientific_convergence':native_result.get('scientific_convergence','unknown'),
            'comparison_limits':native_result.get('comparison_limits') or [],
            'quantity_contract':witness.get('quantity_contract')or{}}
        provenance_cache.add(_digest(source_receipts({'sources':[summary]})[0]))
        # Source IDs, job IDs, file sizes and hashes are provenance, never a
        # pool of possible scientific numbers that an author may cite as data.
        metadata_keys={'resource_review_id','memory_mb','cpus_per_task','num_processes',
                       'nodelist','partition','walltime','output_dir','output_csv','input','path',
                       'command','script','content','body','instructions','task','context'}
        from .workspace import PATH_ARGUMENTS
        metadata_keys.update(PATH_ARGUMENTS)
        summary['conditions']={key:value for key,value in actual_args.items()
            if key not in metadata_keys and not key.endswith(('_path','_dir','_id'))
            and isinstance(value,(str,int,float,bool,list))}
        artifacts = source.get('artifacts') or {}
        candidates = [];sealed_inputs={}
        for name, artifact in artifacts.items():
            if not isinstance(artifact, dict): continue
            if isinstance(artifact.get('files'), list):
                for entry in artifact['files']:
                    if isinstance(entry, list) and entry and isinstance(entry[0], str): candidates.append(entry[0])
            else: candidates.append(name)
        if source_contract.get('tool')in{'plot_result_code','generate_result_figures'}:
            for image in native_result.get('artifacts')or[]:
                if not isinstance(image,dict)or image.get('kind')!='PNG':raise ValueError('invalid native figure manifest')
                image_path=str(root/_relative(root,image['path']))
                approved_directory=(source_contract.get('arguments')or{}).get('output_dir')
                if not approved_directory or not Path(image_path).is_relative_to(Path(approved_directory)):
                    raise PermissionError('figure outside approved output branch')
                candidates.append(image_path);sealed_inputs[image_path]=image['sha256']
            if source_contract.get('tool')=='generate_result_figures':
                # The figure node is an explicit dataflow stage. A report that
                # depends on it must still consume the original complete source
                # tables, not infer values by looking at PNG pixels or log JSON.
                for table_source in native_result.get('source_tables')or[]:
                    if not isinstance(table_source,dict)or not isinstance(table_source.get('path'),str):
                        raise ValueError('figure source table manifest is invalid')
                    original=str(root/_relative(root,table_source['path']))
                    candidates.append(original);sealed_inputs[original]=table_source['sha256']
                for source_file in native_result.get('source_inputs')or[]:
                    if (not isinstance(source_file,dict)or source_file.get('format')not in{'csv','json','text'}
                            or not re.fullmatch('[a-f0-9]{64}',str(source_file.get('sha256','')))):
                        raise ValueError('figure source input manifest is invalid')
                    original=str(root/_relative(root,source_file['path']))
                    candidates.append(original);sealed_inputs[original]=source_file['sha256']
        if authored_source:
            # Earlier report stages retain references to the original data,
            # rather than making an assembly reparse a JSON index or trust
            # rounded numbers extracted from prose.
            for source_file in native_result.get('data_sources')or[]:
                if (not isinstance(source_file,dict) or not isinstance(source_file.get('path'),str)
                        or not re.fullmatch('[a-f0-9]{64}',str(source_file.get('sha256','')))):
                    raise ValueError('authored fragment data manifest is invalid')
                path=str(root/_relative(root,source_file['path']))
                sealed_inputs[path]=source_file['sha256'];candidates.append(path)
            if not sealed_inputs and not native_result.get('reference_records'):
                raise ValueError('authored fragment has no traceable original result data')
        from .gcmc_report_inputs import attach_gcmc_inputs
        native_gcmc_document=attach_gcmc_inputs(root,owner,state,source,candidates,sealed_inputs,
            read=_read,budget=budget,file_facts=file_facts,attest=attest)
        from .string_result_inputs import attach_string_inputs
        native_string_inputs=attach_string_inputs(root,source_contract,native_result,
            candidates,sealed_inputs,budget,file_facts,_read)
        from .md_report_inputs import attach_md_inputs
        native_md_inputs=attach_md_inputs(root,owner,state,source,native_result,summary,tables,sealed_inputs,
            report_tool=tool,read=_read,budget=budget,file_facts=file_facts)
        from .report_source_metadata import string_input_documents,scheduler_document,metadata_tables
        metadata_documents=string_input_documents(root,owner,state,source,summary,native_result,candidates,
            read=_read,budget=budget,file_facts=file_facts)
        if tool=='generate_scientific_report':
            try:
                provenance=scheduler_document(root,owner,state,source,summary,native_result,
                    job_lookup=job_lookup,read=_read,budget=budget,file_facts=file_facts)
            except (OSError,ValueError,KeyError,TypeError):
                provenance=None
                limits=summary['comparison_limits']
                summary['comparison_limits']=(limits if isinstance(limits,list)else[str(limits)])+[
                    'Native scheduler provenance was unavailable; no resource or identifier facts were inferred.']
            if provenance:
                metadata_documents.append(provenance)
                if source_contract.get('tool')=='run_cdft':
                    from .report_source_metadata import cdft_document
                    try:
                        native=cdft_document(root,owner,source,summary,native_result,provenance,
                            read=_read,budget=budget,file_facts=file_facts)
                    except (OSError,ValueError,KeyError,TypeError):
                        native=None
                    if native:metadata_documents.append(native)
            if source_contract.get('tool')=='generate_result_figures':
                from .report_source_metadata import cdft_ancestor_documents
                metadata_documents.extend(cdft_ancestor_documents(root,owner,state,source,native_result,
                    job_lookup=job_lookup,attest=attest,read=_read,budget=budget,file_facts=file_facts))
        if len(set(candidates)) > 64: raise ValueError('report source artifact count exceeds budget')
        attached = 0
        if authored_source:
            from .literature_report_bridge import attach_reader_ancestor
            attached += attach_reader_ancestor(root,state,native_result,summary,budget,file_facts,_read)
        if source_contract.get("tool")=="read_scientific_document":
            from .literature_report_bridge import attach_native_document
            attached += attach_native_document(root,native_result,summary,budget,file_facts,_read)
        for name in sorted(set(candidates)):
            path = Path(name)
            if str(path) in native_md_inputs: continue
            original=summary.get('native_document')or{}
            if original and str(_relative(root,path))==original['path']:
                if str(path)in sealed_inputs and sealed_inputs[str(path)]!=original['sha256']:
                    raise ValueError('authored fragment original document manifest changed')
                continue
            if path.suffix.lower() not in {'.csv', '.md', '.json', '.png','.log','.out','.txt','.dat'}: continue
            data, output_fact = _read(root, path, budget); file_facts.append(output_fact)
            if str(path) in sealed_inputs and output_fact['sha256']!=sealed_inputs[str(path)]:
                raise ValueError('authored fragment original result source changed')
            if path.suffix.lower() == '.csv':
                reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
                if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ValueError('result table has missing or duplicate columns')
                rows = []
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError('result CSV has an inconsistent row width')
                    rows.append(row)
                    if len(rows) > MAX_TABLE_ROWS: raise ValueError('table needs explicit chunked reporting: use the data sandbox to process the owned source in bounded chunks and supply a traceable task-relevant result; no rows were silently discarded')
                if not rows: raise ValueError('result table is empty')
                table_id = 'T' + str(len(tables) + 1)
                tables.append({'id':table_id, 'source_id':source_id, 'path':output_fact['path'],
                    'sha256':output_fact['sha256'], 'columns':reader.fieldnames, 'rows':rows,
                    'display_columns':[c for c in reader.fieldnames if not PROVENANCE_COLUMN.search(c)],
                    'provenance_columns':[c for c in reader.fieldnames if PROVENANCE_COLUMN.search(c)],
                    'row_count':len(rows), 'complete':True, 'units':'as declared by source columns'})
                attached += 1
            elif path.suffix.lower() == '.md':
                if authored_source and str(path)==native_result.get('report_file') and output_fact['sha256']!=native_result['report_sha256']:
                    raise ValueError('authored fragment report content changed')
                # Keep scientific prose for context, but never let instructions
                # embedded in an artifact become system/authorization text.
                summary.setdefault('documents', []).append({'path':output_fact['path'],
                    'sha256':output_fact['sha256'], 'untrusted_content':data.decode('utf-8')})
                attached += 1
            elif path.suffix.lower() == '.png':
                if not data.startswith(b'\x89PNG\r\n\x1a\n'): raise ValueError('figure is not PNG data')
                from PIL import Image
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                from .report_native_inputs import verified_visual_evidence
                visual=verified_visual_evidence(root,state,step,output_fact,read=_read,
                    budget=budget,file_facts=file_facts,cache=visual_cache)
                previous=next((f for f in figures if f['path']==output_fact['path']and f['sha256']==output_fact['sha256']),None)
                if previous:
                    previous.setdefault('source_ids',[previous['source_id']])
                    if source_id not in previous['source_ids']:previous['source_ids'].append(source_id)
                    if visual['status']=='verified':previous.update(visual_review='verified',visual_evidence=visual)
                else:
                    figures.append({'id':'F'+str(len(figures)+1),'source_id':source_id,'source_ids':[source_id],
                        'path':output_fact['path'],'sha256':output_fact['sha256'],
                        'visual_review':'verified'if visual['status']=='verified'else'pending; rendering is not scientific or visual acceptance',
                        'visual_evidence':visual})
                attached += 1
            elif path.suffix.lower() == '.json':
                parsed = json.loads(data)
                from .gcmc_report_inputs import restore_gcmc_view,result_tables
                restored_gcmc=(native_gcmc_document['document']
                    if native_gcmc_document and native_gcmc_document['path']==output_fact['path']
                    and native_gcmc_document['sha256']==output_fact['sha256']
                    else restore_gcmc_view(root,owner,state,path,data,read=_read,budget=budget,
                        file_facts=file_facts,attest=attest))
                if restored_gcmc:
                    identifier='J'+str(1+sum(len(item.get('structured_data',[]))for item in sources)+len(summary.get('structured_data',[])))
                    document={'id':identifier,'path':output_fact['path'],'sha256':output_fact['sha256'],**restored_gcmc}
                    summary.setdefault('structured_data',[]).append(document)
                    if tool=='generate_scientific_report':result_tables(document,source_id,tables)
                    attached+=1
                    continue
                from .report_source_metadata import restore_metadata
                restored_metadata=restore_metadata(root,owner,state,path,data,
                    read=_read,budget=budget,file_facts=file_facts)
                if restored_metadata:
                    identifier='J'+str(1+sum(len(item.get('structured_data',[]))for item in sources)+len(summary.get('structured_data',[])))
                    document={'id':identifier,'path':output_fact['path'],'sha256':output_fact['sha256'],**restored_metadata}
                    summary.setdefault('structured_data',[]).append(document)
                    if tool=='generate_scientific_report':metadata_tables(document,source_id,tables)
                    attached+=1
                    continue
                from .plot_input_adapter import preview
                json_id='J'+str(1+sum(len(item.get('structured_data',[]))for item in sources)+len(summary.get('structured_data',[])))
                summary.setdefault('structured_data', []).append({'id':json_id,'path':output_fact['path'],
                    'sha256':output_fact['sha256'], 'untrusted_content':parsed,'preview':preview(data)})
                attached += 1
            elif path.suffix.lower()in{'.log','.out','.txt','.dat'}:
                from .plot_input_adapter import preview,text_format_hints
                identifier='L'+str(1+sum(len(item.get('text_documents',[]))for item in sources)+len(summary.get('text_documents',[])))
                summary.setdefault('text_documents',[]).append({'id':identifier,'path':output_fact['path'],
                    'sha256':output_fact['sha256'],'format':'text','preview':preview(data),'format_hints':text_format_hints(data),
                    **({'native_string':native_string_inputs[str(path)]}if str(path)in native_string_inputs else{})})
                if tool=='generate_scientific_report':
                    from .report_native_inputs import append_thermo_tables
                    native_summary=append_thermo_tables(data,output_fact,source_id,tables,max_rows=MAX_TABLE_ROWS)
                    if native_summary:summary['text_documents'][-1]['native_thermo']=native_summary
                attached += 1
        for document in metadata_documents:
            document={**document,'id':'J'+str(1+sum(len(item.get('structured_data',[]))for item in sources)+len(summary.get('structured_data',[])))}
            summary.setdefault('structured_data',[]).append(document)
            if tool=='generate_scientific_report':metadata_tables(document,source_id,tables)
            attached+=1
        supplied_references=(_references(native_result,native_tool=tool)if not authored_source else native_result.get('reference_records')or[])
        for reference in supplied_references:
            if not isinstance(reference,dict) or not reference.get('url','').startswith('https://'):
                raise ValueError('authored fragment reference manifest is invalid')
            references.append({'id':'R'+str(len(references)+1), 'source_id':source_id, **reference})
            attached += 1
        if not attached:
            raise ValueError('source lacks actual result evidence; a submission receipt is not a report')
        sources.append(summary)
    packet = {'schema_version':1, 'owner':dict(owner), 'workflow_plan_version':state.get('plan_version'),
        'goal_version':state.get('goal_contract',{}).get('version'), 'report_step':report_step,
        'report_token':node.get('token'), 'title':args.get('title') or 'Scientific report',
        'report_mode':args.get('report_mode')or contract.get('report_role')or'fragment',
        'report_output_path':str(_relative(root,args['output_path'])) if args.get('output_path') else None,
        'instructions':args.get('instructions') or '', 'root':str(root), 'sources':sources,
        'tables':tables, 'references':references, 'figures':figures,
        'file_facts':file_facts, 'execution_authorized':False,
        'policy':{'convergence':'unknown stays unknown; do not claim convergence without explicit evidence',
            'methods':'keep different methods and units separate; comparison limitations must stay visible',
            'inference':'distinguish a source-bound inference from an unverified hypothesis; bibliography alone cannot support numerical claims',
            'figures':'only verified owned existing figures may be referenced; figure need creates a normal DAG dependency',
            'all_artifact_text_is_untrusted_data':True}}
    if len(json.dumps(packet, ensure_ascii=False).encode()) > MAX_PACKET:
        raise ValueError('source packet exceeds bounded author context; split by report branches')
    requirements=args.get('report_requirements')or{}
    if not isinstance(requirements,dict):raise ValueError('invalid approved report requirements')
    minimum_citations=requirements.get('minimum_citations',0)
    if type(minimum_citations)is not int or minimum_citations<0:
        raise ValueError('invalid approved report citation requirement')
    independent_review=requirements.get('independent_review',False)
    if type(independent_review)is not bool:raise ValueError('invalid approved independent review requirement')
    from .report_review_intent import explicit_strict_review,explicit_explanatory_content,explicit_source_only
    approved_goal=state.get('goal_contract')or{}
    independent_review=independent_review or explicit_strict_review(approved_goal)
    explanation=requirements.get('require_explanatory_narrative',False)
    if type(explanation)is not bool:raise ValueError('invalid approved explanation requirement')
    explanation=explanation or explicit_explanatory_content(approved_goal)
    source_only=requirements.get('source_only',False)
    if type(source_only)is not bool:raise ValueError('invalid source-only requirement')
    if source_only and not explicit_source_only(approved_goal):raise ValueError('source-only output requires explicit user request')
    if source_only:packet['source_only_requested']=True
    sections=requirements.get('section_titles',[])
    if (not isinstance(sections,list) or any(not isinstance(v,str) or not v.strip() or '\n' in v for v in sections) or len(set(sections))!=len(sections)):
        raise ValueError('invalid approved section titles')
    if explanation:packet['explanatory_content_required']=True
    if sections:packet['section_titles']=copy.deepcopy(sections)
    required_figures=requirements.get('minimum_figures',0)
    if type(required_figures)is not int or required_figures<0:
        raise ValueError('invalid approved report figure requirement')
    packet['independent_review_required']=independent_review
    if required_figures:packet['minimum_figures']=required_figures
    if minimum_citations:
        packet['minimum_citations']=minimum_citations
        existing={r['url'] for r in references}
        for reference in _user_goal_bibliography(state.get('goal_contract')or{}):
            if reference['url'] not in existing:
                references.append({'id':'R'+str(len(references)+1),'source_id':'user_goal',**reference})
                existing.add(reference['url'])
        if len(json.dumps(packet,ensure_ascii=False).encode())>MAX_PACKET:
            raise ValueError('source packet exceeds bounded author context; split by report branches')
    packet['packet_sha256'] = _digest(packet)
    if minimum_citations and len(references)<minimum_citations:
        raise ReportCapabilityGap('approved report requires task-sourced bibliographic records',
            required_action='add_approved_literature_lookup_dependency',missing_citations=minimum_citations-len(references),
            no_fulltext_required_for_bibliography=True)
    if len(figures)<required_figures:
        raise ReportCapabilityGap('approved report requires missing figure artifacts',
            required_action='add_approved_plot_dependency',missing_figures=required_figures-len(figures),
            available_tables=[{'table_id':t['id'],'path':t['path'],'sha256':t['sha256']}for t in tables])
    _control(control_valid)
    if not sources_unchanged(root, packet): raise ValueError('report source changed during packet creation')
    return packet


def sources_unchanged(root, packet):
    try:
        if _digest({k:v for k,v in packet.items() if k != 'packet_sha256'}) != packet.get('packet_sha256'):
            return False
        budget = {'bytes':0}
        from .document_source_facts import stream_document_fact,DOCUMENT_VERIFICATION
        from .gcmc_report_inputs import STREAM_VERIFICATION,DIRECTORY_VERIFICATION,stream_fact,directory_fact,implementation_sha256,SCHEMA
        for source in packet.get('sources',[]):
            for document in source.get('structured_data',[]):
                if document.get('metadata_schema')==SCHEMA and document.get('parser_sha256')!=implementation_sha256():return False
        for expected in packet['file_facts']:
            if expected.get('verification')==STREAM_VERIFICATION:
                actual=stream_fact(root,expected['path'],budget)
            elif expected.get('verification')==DIRECTORY_VERIFICATION:
                actual=directory_fact(root,expected['path'])
            elif expected.get('verification')==DOCUMENT_VERIFICATION:
                _,actual=stream_document_fact(root,expected['path'],budget)
            else:
                _,actual=_read(root,expected['path'],budget)
            if actual['sha256'] != expected['sha256'] or actual['size'] != expected['size']:
                return False
        # Final linked identities after all content reads catch normal atomic
        # replacement or modification of an earlier source during later I/O.
        for expected in packet['file_facts']:
            parent,name=_open_parent(root,_relative(root,expected['path']))
            try:current=os.stat(name,dir_fd=parent,follow_symlinks=False)
            finally:os.close(parent)
            if (not (stat.S_ISDIR(current.st_mode)if expected.get('verification')==DIRECTORY_VERIFICATION else stat.S_ISREG(current.st_mode)) or current.st_dev!=expected['device']
                    or current.st_ino!=expected['inode'] or current.st_size!=expected['size']
                    or current.st_mtime_ns!=expected['mtime_ns'] or current.st_ctime_ns!=expected['ctime_ns']):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError): return False



def _numbers(value):
    result = set()
    for raw in NUMBER.findall(str(value)):
        try:
            number = Decimal(raw)
            if number.is_finite(): result.add(number.normalize())
        except InvalidOperation: pass
    return result


def _escape(value):
    return _plain(value).replace('|','\\|').replace('\r',' ').replace('\n',' ')


def _plain(value):
    """Untrusted labels are text, never Markdown links/images or HTML."""
    text=html.escape(str(value),quote=False)
    return re.sub(r'([\\`*{}\[\]()#+!_>])',r'\\\1',text)


def _uncertainty_column(name):
    return bool(re.search(r'(?:^|_)(?:std|stderr|standard_error|error|uncertainty|ci|lower_ci|upper_ci)(?:_|$)',name,re.I))


def _unit_for(table,row,column):
    declared=table.get('column_units')or{}
    if column in declared:return declared[column]if declared[column]is not None else'unspecified'
    for suffix,unit in (('_mol_kg','mol/kg'),('_mol_L','mol/L'),('_K','K'),('_Pa','Pa'),
                        ('_cm3_g','cm³/g'),('_g_cm3','g/cm³'),('_A','Å')):
        if column.endswith(suffix):return unit
    return row.get('loading_unit')or row.get('unit')or row.get('units')or'unspecified'


def _row_selector_matches(row,selector):
    """Match typed native numeric cells without normalizing textual identities."""
    for key,wanted in selector.items():
        actual=row.get(key)
        if isinstance(actual,str):
            if actual!=wanted:return False
        elif isinstance(actual,(int,float,Decimal))and not isinstance(actual,bool):
            try:
                number=Decimal(str(actual));selected=Decimal(wanted)
            except (InvalidOperation,TypeError,ValueError):return False
            if not number.is_finite()or not selected.is_finite()or number!=selected:return False
        else:return False
    return True


def _bind_measurements(packet,block):
    text=block['text'];references=block.get('data_refs')or[]
    if not isinstance(references,list):raise ValueError('invalid measurement binding list')
    tables={t['id']:t for t in packet['tables']};bindings=[];seen=set()
    original=text;anchors=[]
    # A sentence naturally shares one subject across a value and its std/CI.
    # The anchor is still a native row reference, never a guessed material name
    # or a number pool. A later material identity ends that shared context.
    for candidate in references:
        if (not isinstance(candidate,dict)or not isinstance(candidate.get('id'),str)
                or not isinstance(candidate.get('row_selector'),dict)
                or any(not isinstance(k,str)or not isinstance(v,str)
                       for k,v in candidate['row_selector'].items())):continue
        row=(candidate.get('table_id'),tuple(sorted(candidate['row_selector'].items())))
        token='{{'+candidate['id']+'.identity}}'
        anchors.extend((match.start(),row)for match in re.finditer(re.escape(token),original))
    anchors.sort(key=lambda item:item[0])
    for ref in references:
        if not isinstance(ref,dict)or set(ref)-{'id','table_id','row_selector','column','format'}:
            raise ValueError('invalid measurement binding')
        name=ref.get('id');selector=ref.get('row_selector');table=tables.get(ref.get('table_id'))if isinstance(ref.get('table_id'),str)else None
        if (not isinstance(name,str)or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,15}',name)
                or name in seen or table is None or table['source_id']not in resolved_source_ids(packet,block['source_ids'])
                or not isinstance(selector,dict)or not selector
                or any(k not in table['columns']or not isinstance(v,str)for k,v in selector.items())):
            raise ValueError('measurement binding lacks a unique cited table identity')
        seen.add(name);column=ref.get('column')
        if not isinstance(column,str)or column not in (table.get('display_columns')or table['columns']):
            raise ValueError('measurement column is not a public result quantity')
        matches=[row for row in table['rows']if _row_selector_matches(row,selector)]
        if len(matches)!=1:raise ValueError('measurement selector is absent or ambiguous')
        row=matches[0];raw=row[column]
        try:number=Decimal(raw)
        except (InvalidOperation,TypeError):raise ValueError('measurement value is not numeric')
        if not number.is_finite():raise ValueError('missing/nonfinite measurement cannot be invented')
        form=ref.get('format')
        if form is not None and (not isinstance(form,str)or not re.fullmatch(r'\.(?:[0-9]|1[0-2])[fge]',form)):
            raise ValueError('unsupported source-bound numerical formatting')
        value=format(number,form)if form is not None else str(raw)
        identity=next(iter(selector.values()))if len(selector)==1 else ', '.join(k+'='+v for k,v in selector.items())
        identity_token='{{'+name+'.identity}}';value_token='{{'+name+'.value}}'
        if value_token not in original:
            raise ValueError('measurement must bind both actual material identity and value in prose')
        if identity_token not in original:
            row_identity=(ref['table_id'],tuple(sorted(selector.items())))
            for position in (match.start()for match in re.finditer(re.escape(value_token),original)):
                preceding=[anchor for anchor in anchors if anchor[0]<position]
                if not preceding or preceding[-1][1]!=row_identity:
                    raise ValueError('measurement must bind both actual material identity and value in prose')
        for kind,content in (('identity',identity),('value',value),('column',column),('unit',_unit_for(table,row,column))):
            text=text.replace('{{'+name+'.'+kind+'}}',str(content))
        bindings.append({'id':name,'table_id':table['id'],'source_id':table['source_id'],
            'row_selector':dict(selector),'column':column,'source_value':raw,'display_value':value,
            'format':form,'unit':_unit_for(table,row,column),'source_sha256':table['sha256']})
    if re.search(r'\{\{[^}]+\}\}',text):raise ValueError('unresolved or unknown source-bound value token')
    return text,bindings


def _measurement_block(packet,block):
    """Render selected source cells with native identity, type and units."""
    if not isinstance(block,dict)or set(block)-{'kind','table_id','row_selector','columns','format'}:
        raise ValueError('invalid native measurement block')
    table=next((item for item in packet['tables']if item['id']==block.get('table_id')),None)
    selector=block.get('row_selector');columns=block.get('columns')
    if (table is None or not isinstance(selector,dict)or not selector
            or any(key not in table['columns']or not isinstance(value,str)for key,value in selector.items())
            or not isinstance(columns,list)or not columns
            or any(not isinstance(column,str)for column in columns)or len(set(columns))!=len(columns)
            or not set(columns)<=set(table.get('display_columns')or table['columns'])):
        raise ValueError('native measurement requires a unique source row and public columns')
    matches=[row for row in table['rows']if _row_selector_matches(row,selector)]
    if len(matches)!=1:raise ValueError('native measurement row is absent or ambiguous')
    row=matches[0];form=block.get('format')
    if form is not None and (not isinstance(form,str)or not re.fullmatch(r'\.(?:[0-9]|1[0-2])[fge]',form)):
        raise ValueError('unsupported native measurement formatting')
    identity=next(iter(selector.values()))if len(selector)==1 else ', '.join(key+'='+value for key,value in selector.items())
    shown=[];bindings=[]
    for column in columns:
        raw=row[column]
        try:number=Decimal(raw)
        except (InvalidOperation,TypeError):number=None
        kind='number'if number is not None and number.is_finite()else'nonfinite'if number is not None else'missing'if raw is None or not str(raw).strip()else'text'
        if kind=='number':value=format(number,form)if form is not None else str(raw)
        elif kind=='nonfinite':value='unavailable (nonfinite source cell: '+str(raw)+')'
        elif kind=='missing':value='unavailable (missing source cell)'
        else:value=str(raw)
        unit=_unit_for(table,row,column)if kind!='text'else'unspecified'
        shown.append(column+' = '+value+(' '+unit if unit!='unspecified'else''))
        bindings.append({'table_id':table['id'],'source_id':table['source_id'],
            'row_selector':dict(selector),'column':column,'source_value':raw,'display_value':value,
            'format':form if kind=='number'else None,'unit':unit,'value_kind':kind,
            'source_sha256':table['sha256']})
    return identity+': '+'; '.join(shown)+'.',bindings,table['source_id']


def _fact_block(packet,block):
    if not isinstance(block,dict)or set(block)!={'kind','source_id','document_id','pointer'}:
        raise ValueError('invalid native structured fact block')
    requested=resolved_source_ids(packet,[block['source_id']])[0]
    source=next((item for item in packet['sources']if item['id']==requested),None)
    document=next((item for item in (source or{}).get('structured_data',[])
                   if item.get('id')==block['document_id']),None)
    pointer=block['pointer']
    if (document is None or not isinstance(pointer,str)or len(pointer)>1024
            or pointer and not pointer.startswith('/')):
        raise ValueError('structured fact must select an owned result document and JSON pointer')
    from .report_fact_diagnostics import validate_fact_projection, missing_fact_pointer_diagnostic
    base_pointer=validate_fact_projection(document)
    value=document['untrusted_content']
    parts=pointer.split('/')[1:]if pointer else[]
    if len(parts)>32:raise ValueError('structured fact pointer is too deep')
    for part in parts:
        if re.search(r'~(?![01])',part):raise ValueError('invalid JSON pointer escape')
        key=part.replace('~1','/').replace('~0','~')
        if isinstance(value,dict)and key in value:value=value[key]
        elif isinstance(value,list)and re.fullmatch(r'0|[1-9][0-9]*',key)and int(key)<len(value):value=value[int(key)]
        else:raise ValueError('structured fact pointer does not identify a source value; '+missing_fact_pointer_diagnostic(document,value))
    if isinstance(value,(list,dict))or isinstance(value,float)and not math.isfinite(value):
        raise ValueError('structured fact requires a finite scalar')
    rendered=json.dumps(value,ensure_ascii=False,allow_nan=False)
    if len(rendered)>2000:raise ValueError('structured fact exceeds bounded scalar size')
    ancestor=document.get('native_ancestor_step')
    text=((str(ancestor)+' · ') if isinstance(ancestor,str) and ancestor else '')+Path(document['path']).name+' · '+(pointer or'/')+' = '+rendered
    binding={'source_id':source['id'],'document_id':document['id'],'path':document['path'],
        'pointer':base_pointer+pointer,'value':value,'source_sha256':document['sha256'],
        **({'quantity_role':document['role']}if document.get('role')in{'provenance','native_input_evidence'}else{})}
    return text,binding,source['id']


def _rejected_narrative_excerpt(text):
    """Bound correction context without exposing paths, credentials or URLs."""
    if not isinstance(text, str): return ''
    text = re.sub(r'https?://\S+|(?<![\w])(?:file://)?/(?:[^\s/]+/)*[^\s/]+', '[path]', text)
    text = re.sub(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b', '[email]', text)
    text = re.sub(r'(?i)\b(?:bearer\s+|(?:api[_-]?key|token|password)\s*[:=]\s*)\S+', '[credential]', text)
    text = re.sub(r'\b[A-Za-z0-9_+=-]{32,}\b', '[identifier]', text)
    return ' '.join(text.split())[:600]


def _source_material_labels(source):
    """Return bounded source-attested labels, never numbers as allowed facts."""
    labels=set()
    def add(value, *, path=False):
        if not isinstance(value,str) or not value or len(value)>2048:return
        name=Path(value).name if path else value
        stem=Path(name).stem if path else name
        candidates=[stem]
        # Only the known charge-tool filename suffixes have a defined
        # relationship to the original material identity. Unknown suffixes
        # must never silently turn another material into a matching label.
        if path:
            original=re.sub(r'(?i)(?:_pacman|_pacmof)+$','',stem)
            if original!=stem:candidates.append(original)
        for label in candidates:
            if len(label)<=128 and re.fullmatch(r'[A-Za-z][A-Za-z0-9]*-\d+[A-Za-z0-9-]*',label):
                labels.add(label)
    for document in source.get('structured_data',[]):
        data=document.get('untrusted_content')
        if not isinstance(data,dict):continue
        for key in ('material','material_id','cif_name','MOF','framework'):
            add(data.get(key))
        for key in ('cif_path','input_cif','output_cif'):
            add(data.get(key),path=True)
    return labels


def _mask_source_material_labels(text, sources, ids):
    labels=set().union(*(_source_material_labels(sources[key]) for key in ids))
    for label in sorted(labels,key=len,reverse=True):
        text=re.sub(r'(?<![A-Za-z0-9_])'+re.escape(label)+r'(?![A-Za-z0-9_-])',
                    lambda match:' '*len(match.group()),text)
    return text


def _scientific_claim_clauses(text):
    """Separate sentence, punctuation and contrast boundaries for claims."""
    return [part for part in re.split(
        r'[。！？!?；;\n，,]|(?<!\d)\.(?!\d)|\b(?:but|however|nevertheless|nonetheless|yet)\b|但是|然而|不过|反而|但|却|(?=而是)',
        str(text).lower()) if part]


def _positive_mixture_selectivity_claim(text):
    """Reject positive interpretations; a local explicit denial is not one.

    Evaluate each overlapping predicate separately so a later affirmative
    clause cannot be hidden by an earlier denial. Numeric and provenance
    checks are independent and always run before this gate.
    """
    pattern = r'(?=((?:\b(?:demonstrates?|proves?|is|are|represents?|indicates?)\b|为|是).{0,25}(?:iast|mixture selectivity|混合气选择性)))'
    for clause in _scientific_claim_clauses(text):
        for match in re.finditer(pattern, clause):
            claim = match.group(1); prefix = clause[max(0, match.start()-24):match.start()]
            immediate = re.search(r'(?:不|非|未|没|没有|并非|不是|不能|不可|无法|不能称|不应称|不称|不能视|不能认定|不能解读)\s*$', prefix)
            english = re.search(r'\b(?:not|never|cannot|does\s+not|do\s+not)\s*$', prefix)
            internal = re.search(r'(?:并非|不是|不等于|不属于|不构成|非|不能|不可|无法|\bnot\b|\bnever\b)', claim, re.I)
            if not (immediate or english or internal):
                return True
    return False


def render_validated_draft(packet, draft):
    if (not isinstance(draft, dict) or not isinstance(draft.get('blocks'), list)
            or not draft['blocks']
            or len(json.dumps(draft, ensure_ascii=False).encode()) > MAX_DRAFT):
        raise ValueError('invalid bounded report draft')
    sources = {s['id']:s for s in packet['sources']}
    tables = {t['id']:t for t in packet['tables']}
    references = {r['id']:r for r in packet['references']}
    figures = {f['id']:f for f in packet['figures']}
    allowed_numbers = {key:set() for key in sources}
    for key, source in sources.items():
        allowed_numbers[key] |= _numbers(json.dumps(source.get('conditions',{}), ensure_ascii=False))
        for document in source.get('documents',[]):
            allowed_numbers[key] |= _numbers(document.get('untrusted_content',''))
        for document in source.get('structured_data',[]):
            if document.get('role')in{'provenance','native_input_evidence'}:continue
            allowed_numbers[key] |= _numbers(json.dumps(document.get('untrusted_content'),ensure_ascii=False))
        for document in source.get('text_documents',[]):
            allowed_numbers[key] |= _numbers((document.get('preview')or{}).get('text',''))
    for table in tables.values():
        display=table.get('display_columns')or table['columns']
        shown=[{key:row[key]for key in display}for row in table['rows']]
        allowed_numbers[table['source_id']] |= _numbers(json.dumps(shown,ensure_ascii=False)) | {Decimal(table['row_count']),Decimal(len(table['columns']))}
    if draft.get('title') not in (None,packet['title']):
        raise ValueError('report title must retain the approved title')
    lines = ['# '+_plain(packet['title']).replace('\n',' '), '']
    used_tables = set(); used_refs = set(); used_documents=set(); used_figures=set(); narrative = 0
    for block_index,block in enumerate(draft['blocks']):
        try:
            if not isinstance(block,dict): raise ValueError('report block must be structured')
            kind = block.get('kind')
            if kind == 'narrative':
                text=block.get('text'); ids=resolved_source_ids(packet,block.get('source_ids')); claim=block.get('claim_type')
                if (not isinstance(text,str) or not text.strip() or '```' in text or
                        not isinstance(ids,list) or not ids or any(not isinstance(key,str)for key in ids)
                        or not set(ids)<=set(sources) or
                        claim not in {'measured','derived','hypothesis','external','recommendation',
                                      'limitation','method','result','inference'}):
                    raise ValueError('narrative requires actual source references and claim type')
                # Local paths are not prose citations. Detect abbreviated and
                # relative paths as well as host paths before looking for the
                # English word "literature"; that word may simply be a folder.
                local_path=r'(?<![\w])(?:\.{1,3}/)?(?:[\w.+-]+/)+[\w.+-]+\.(?:pdf|cif|csv|json|dat|lmp|lammps)(?![\w])'
                path_scan=re.sub(r'https?://\S+','',text,flags=re.I)
                if '/home/' in path_scan or 'file://' in text or re.search(local_path,path_scan,re.I):
                    raise ValueError('local path in report prose; remove the path and use the native owned source-document link or a verified citation block')
                if re.search(r'https?://|!?\[[^\]]*\]\(',text):
                    raise ValueError('narrative links must use verified citation or figure blocks; remove URLs and Markdown links from this block. Owned native documents are linked automatically; cite only available reference IDs and use exact excerpt blocks for original text')
                for reference in re.findall(r'\[(R[0-9]+)\]',text):
                    if reference not in references:raise ValueError('unknown literature reference in narrative')
                numbers=_numbers(_mask_source_material_labels(text,sources,ids)); allowed=set().union(*(allowed_numbers[key]for key in ids))
                from .literature_report_bridge import narrative_excerpt_bindings
                excerpt_bindings=narrative_excerpt_bindings(packet,block)
                if any(binding['source_id']not in ids for binding in excerpt_bindings):
                    raise ValueError('excerpt claim source differs from narrative source')
                for binding in excerpt_bindings:allowed|=_numbers(binding['excerpt'])
                rendered,bound=_bind_measurements(packet,block)
                if (packet.get('independent_review_required') is False
                        and re.search(r'\b(?:prove[sd]?|establish(?:es|ed)?|demonstrat(?:es|ed))\b|证明|证实',text,re.I)
                        and not bound and not any(text.strip()==b.get('excerpt','').strip() for b in excerpt_bindings)):
                    raise ValueError('a claim of proof requires a source-bound measurement or an exact source excerpt; describe an unsupported interpretation as unverified')
                if (packet.get('independent_review_required') is False and claim in {'result','measured','derived'}
                        and not bound and not excerpt_bindings):
                    raise ValueError('result statement requires a source-bound measurement or excerpt; unverified explanations belong in inference blocks')
                if claim in {'measured','derived','result'}and numbers:
                    raise ValueError('measured numerical prose requires material/metric source bindings, not a number pool')
                if claim in {'measured','derived','result'}and block.get('data_refs')and not bound:
                    raise ValueError('measurement references were not resolved')
                lowered=text.lower()
                cited_references=block.get('reference_ids')or[]
                if (not isinstance(cited_references,list)or any(not isinstance(key,str)for key in cited_references)
                        or not set(cited_references)<=set(references)):
                    raise ValueError('unknown literature reference in claim')
                comparison=bool(re.search(r'\bliterature\b|\bpublished\b|\bprevious(?:ly| studies?)\b.{0,24}\breported\b|\b(?:prior|previous|earlier|other)\s+(?:studies|research|work|papers|reports)\b|文献|前人|(?:已有|既有|既往|先前|此前|过往|其他|已发表)(?:的)?(?:研究|工作|报道|论文|结果)|\b[a-z][a-z-]{1,40}\s*(?:等人|et\s+al\b)',lowered))
                if references and re.search(r'(?:不含|未包含|没有)(?:任何)?(?:参考)?文献(?:条目|记录|证据)',lowered):
                    raise ValueError('report claims no references although the source packet contains references')
                # Denying comparability does not assert an external result.
                # Claim labels alone never exempt a positive literature claim.
                denied_comparison=claim in {'inference','recommendation','method','limitation'}and _nonquantitative_literature_denial(text)
                if (comparison and not denied_comparison)or claim=='external':
                    if not excerpt_bindings and (not cited_references or any(references[key].get('support_kind')!='supplied_excerpt'
                            for key in cited_references)):
                        raise ValueError('literature comparison lacks actual supporting excerpt evidence; if only denying a comparison, put one complete simple denial in a separate limitation block; ambiguous or positive claims require verified excerpts')
                if claim=='recommendation'and numbers and (not cited_references or
                        any(references[key].get('support_kind')!='supplied_excerpt'for key in cited_references)):
                    raise ValueError('numeric recommendation lacks task-applicable external or approved protocol evidence; separate source-backed observed conditions into a method or limitation block and keep an unsupported suggestion nonnumeric; never invent a numeric recommendation')
                if claim in {'measured','derived','method','result'}and _positive_normality_or_causality(lowered):
                    raise ValueError('measured data do not by themselves establish a normality or causal explanation')
                for key in cited_references:
                    allowed|=_numbers(references[key].get('snippet',''))
                if not numbers<=allowed:
                    missing=[str(value)for value in sorted(numbers-allowed)[:12]]
                    raise ValueError('unsupported numerical claim: values '+json.dumps(missing)
                        +' absent from cited native source slots '+json.dumps(ids)
                        +'; identifiers/resources require a native provenance fact, not scientific numerical prose'
                        +'; rejected text (redacted): '+_rejected_narrative_excerpt(text))
                for source_id in ids:
                    semantics=sources[source_id].get('quantity_contract')or{}
                    if (semantics.get('pore_diameters_are_not_sampling_probe_radius')is True
                            and re.search(r'\bdf\b|\bpld\b|孔窗|pore.limiting',lowered)
                            and re.search(r'probe[- ](?:dependent|sensitive)|取决于.{0,12}探针|探针.{0,12}(?:决定|敏感)',lowered)
                            and not re.search(r'\bnot\b|不(?:是|取决|等同|直接)|非',lowered)):
                        raise ValueError('quantity meaning exceeds the native pore-diameter contract')
                # Naming a table's convergence field is not asserting its
                # value. Keep any actual predicate (including one later in
                # the sentence) under the original strict convergence gate.
                convergence_text=re.sub(r'收敛(?:标记|字段|列|标签|指标|状态)(?!\s*(?:为|是|显示|证明|表明|已经|已))',
                                        '[convergence-field]',lowered)
                positive = re.search(r'all[^.。]{0,50}(?:converged|equilibrated)|(?:已经|均已|全部)[^。]{0,15}收敛',convergence_text)
                if positive and not all(sources[key].get('scientific_convergence') is True for key in ids):
                    raise ValueError('unsupported convergence claim')
                if _positive_mixture_selectivity_claim(text):
                    raise ValueError('unsupported mixture-selectivity interpretation')
                prefix=('Unverified hypothesis / 未验证假设：'if claim=='hypothesis'else
                        'Inference / 推断（待核验）：'if claim=='inference'else
                        'Proposed follow-up / 建议（未验证）：'if claim=='recommendation'else
                        'Derived from cited data / 来源推导：'if claim=='derived'else'')
                # Authored prose may already carry the exact standard label.
                # Preserve it once; the claim classification and all evidence
                # checks remain unchanged.
                if prefix and rendered.lstrip().startswith(prefix):prefix=''
                if excerpt_bindings:
                    prefix+='Source-text interpretation (origin and scientific claims unverified): '
                    rendered+=' '+''.join('['+binding['source_id']+'; '+binding['anchor']+']'for binding in excerpt_bindings)
                if packet.get('independent_review_required') is False:
                    prefix='Model-authored explanation / 模型表述（未独立核验）：'+prefix
                lines.extend([prefix+_plain(rendered.strip())+' '+''.join('['+key+']'for key in ids+cited_references),''])
                narrative += 1
            elif kind == 'measurement':
                text,_,source_id=_measurement_block(packet,block)
                lines.extend([_plain(text)+' ['+source_id+']',''])
            elif kind == 'excerpt':
                from .literature_report_bridge import excerpt_block
                text,binding=excerpt_block(packet,block)
                used_documents.add(binding['source_id'])
                lines.extend([text,''])
            elif kind == 'fact':
                text,_,source_id=_fact_block(packet,block)
                lines.extend([_plain(text)+' ['+source_id+']',''])
            elif kind == 'table':
                table=tables.get(block.get('table_id'))if isinstance(block.get('table_id'),str)else None
                if table is None: raise ValueError('unknown source table')
                available=table.get('display_columns')or table['columns']
                cols=block.get('columns')or available
                if (not isinstance(cols,list) or not cols or any(not isinstance(c,str)for c in cols)
                        or len(set(cols))!=len(cols) or not set(cols)<=set(available)):
                    raise ValueError('presentation columns must be a declared source subset')
                # Present the selected source fields; the full native table is
                # linked below for readers who need other conditions or units.
                used_tables.add(table['id'])
                lines.extend(['| '+' | '.join(_escape(x)for x in cols)+' |',
                              '| '+' | '.join('---'for _ in cols)+' |'])
                for row in table['rows']: lines.append('| '+' | '.join(_escape(row[c])if row[c]is not None else'unavailable (native missing/nonfinite)'for c in cols)+' |')
                lines.extend(['','Data: ['+table['source_id']+']; units and method labels retain the source table.',''])
                parent=Path(packet.get('report_output_path')or'index.md').parent
                source_link=quote(os.path.relpath(table['path'],str(parent)),safe='/.-_')
                suffix = Path(table['path']).suffix.lower()
                source_label = {'.csv': 'CSV', '.json': 'JSON', '.cif': 'CIF',
                                '.log': 'log', '.txt': 'text'}.get(suffix, 'source file')
                if table.get('format') == 'native_thermo' and suffix == '.log':
                    source_label = 'native log'
                elif table.get('format') == 'native_cif_geometry' and suffix == '.cif':
                    source_label = 'native CIF'
                lines.extend(['Complete source data: ['+source_label+']('+source_link+'); '+str(table['row_count'])+
                    ' observed source rows. Integration steps and replicate n are separate; missing uncertainty is not estimated.',''])
                if table.get('format')=='native_thermo':
                    context=table['native_context']
                    lines.extend(['Native context: '+_plain('; '.join(str(k)+'='+str(v)for k,v in context.items()))+'.',
                        'Column units: '+_plain('; '.join(k+'='+str(v if v is not None else'unknown')for k,v in table['column_units'].items()))+'.',
                        'Source header line '+str(table['native_source']['header_source']['line'])+'; row lines '+
                            ', '.join(str(a['line'])for a in table['row_sources'])+'. Original log SHA-256 '+table['sha256']+
                            '; native parser SHA-256 '+table['parser_sha256']+'.',''])
                if table.get('provenance_columns'):
                    lines.extend(['Machine provenance columns remain in the original source table; they are not rewritten into prose.',''])
            elif kind == 'citation':
                ref=references.get(block.get('reference_id'))if isinstance(block.get('reference_id'),str)else None
                if ref is None: raise ValueError('unknown literature reference')
                used_refs.add(ref['id'])
                lines.extend(['['+ref['id']+'] ['+_plain(ref['title'])+']('+quote(ref['url'],safe=':/?#=&%.-_~')+'). Evidence: '+_plain(ref['support_kind'])+'.',''])
            elif kind == 'figure':
                fig=figures.get(block.get('figure_id'))if isinstance(block.get('figure_id'),str)else None
                if fig is None: raise ValueError('unknown figure artifact')
                used_figures.add(fig['id'])
                parent=Path(packet.get('report_output_path')or'index.md').parent
                link=quote(os.path.relpath(fig['path'],str(parent)),safe='/.-_')
                visual_text=('independent visual review verified for these exact PNG bytes; scientific acceptance remains false'
                    if fig.get('visual_review')=='verified'and(fig.get('visual_evidence')or{}).get('status')=='verified'
                    else'visual review remains pending')
                lines.extend(['!['+fig['id']+']('+link+')',
                    'Figure provenance: ['+fig['source_id']+']; '+visual_text+'.',''])
                evidence=fig.get('visual_evidence') or {}
                if evidence.get('status')=='verified':
                    for label,key in (('Figure author limitation','author_limitations'),
                                      ('Visual reviewer limitation','review_limitations')):
                        for note in evidence.get(key) or []:
                            # These are model-declared caveats, not a native
                            # scientific verdict; keep host paths out of prose.
                            note=re.sub(r'(?<!\w)/(?:[^/\s]+/)+[^\s]*','[local path]',note)
                            note=re.sub(r'https?://\S+','[link]',note)
                            lines.append(_plain(label+' (unverified): '+note))
                    lines.append('')
            elif kind == 'heading':
                text=block.get('text')
                if not isinstance(text,str) or not text.strip() or '\n'in text: raise ValueError('invalid heading')
                lines.extend(['## '+_plain(text),''])
            else: raise ValueError('unsupported report block')
        except ValueError as error:
            raise ValueError('block['+str(block_index)+']: '+str(error)) from error
    # Existing documents are optional unless the approved task requires them;
    # any excerpt actually used still undergoes native identity validation.
    minimum_figures=packet.get('minimum_figures',0)
    if type(minimum_figures)is not int or minimum_figures<0:raise ValueError('invalid approved report figure requirement')
    if len(used_figures)<minimum_figures:raise ValueError('approved report requires additional figure blocks')
    minimum_citations=packet.get('minimum_citations',0)
    if type(minimum_citations)is not int or minimum_citations<0:
        raise ValueError('invalid approved report citation requirement')
    if len(used_refs)<minimum_citations:
        raise ValueError('approved report requires additional task-sourced citation blocks')
    # References supplied in a packet are optional context.  Claims that
    # actually compare with literature are still checked above against
    # verified excerpts; unrelated references need not appear in a report.
    lines.extend(['## Evidence limits', ''])
    if packet.get('independent_review_required') is False:
        lines.extend(['Source identities and source-bound data were checked; narrative interpretation was not independently verified and independent model review was not performed.', ''])
    lines.extend([
        'Completion and convergence are separate. Unknown convergence remains unknown.',
        'Different methods and units are retained separately; cross-method ordering is exploratory.',
        'A rendered figure or bibliographic identifier alone does not establish scientific validity.', ''])
    for source in packet['sources']:
        limits=source.get('comparison_limits') or []
        if isinstance(limits,str):limits=[limits]
        for limit in limits: lines.append('- '+_plain(limit))
    lines.extend(['','## Source traceability',''])
    for source in packet['sources']:
        lines.append('['+source['id']+'] '+_plain(source['step_id'])+'; native call '+source['call_id']+'; SHA-256 '+source['source_sha256'])
        document=source.get('native_document')
        if document:
            name=document.get('path');sha=document.get('sha256');path=Path(name or '')
            if (not isinstance(name,str) or not name or path.is_absolute() or '..' in path.parts
                    or not isinstance(sha,str) or not re.fullmatch(r'[0-9a-f]{64}',sha)
                    or not any(f.get('path')==name and f.get('sha256')==sha for f in packet.get('file_facts',[]))):
                raise ValueError('native document link lacks matching owned source evidence')
            report_path=Path(packet.get('report_output_path')or'index.md')
            if report_path.is_absolute() or '..' in report_path.parts:
                raise ValueError('report link parent escapes owned source scope')
            parent=report_path.parent
            link=quote(os.path.relpath(name,str(parent)),safe='/.-_')
            lines.append('['+source['id']+' native source document]('+link+'); original bytes SHA-256 '+sha+'; source origin and scientific claims remain unverified.')
    return '\n'.join(lines).rstrip()+'\n'


class ReportReviewUnresolved(ValueError):
    """Bounded independent review did not accept the report draft."""
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


class ReportContractConflict(ReportReviewUnresolved):
    """A bounded attempt found competing review and source-coverage constraints."""


def _positive_normality_or_causality(text):
    """Allow bounded questions, retain the existing positive-claim gate.

    A question about whether a result is normal does not assert normality.
    Contrast, conjunction, and sentence boundaries start new scopes; a later
    positive assertion must still pass the original evidence gate.
    """
    for clause in re.split(r'[。！？!?；;，,]|但是|然而|而且|并且|但|且|而', text.lower()):
        for match in re.finditer(r'\bnormal\b|\bexpected redistribution\b|正常|符合预期|必然',clause):
            prefix=clause[:match.start()];suffix=clause[match.end():]
            question=re.search(r'(?:能否|是否|会否|可否|能不能|有没有|is\s+it)\s*[^。！？!?；;，,]{0,40}$',prefix)
            # Treat only a narrow, non-assertive question tail as safe.
            # Unknown wording fails closed instead of guessing its polarity.
            question_tail=re.fullmatch(r'(?:\s|、|启动|运行|运转|退出|工作|执行|完成|结束|终止|连接|响应|加载|生成|计算|收敛|处理|配置|发布|请求|并|与|及|或|和|地|中|过程|状态|流程|功能|有待验证|待验证|尚未确认|未知|仍|尚)*',suffix)
            previous_assertion=re.search(r'\bnormal\b|\bexpected redistribution\b|正常|符合预期|必然',prefix)
            if not question or not question_tail or previous_assertion:return True
    return False


def author_report(packet, *, model_call, control_valid=None, max_rounds=2):
    """Author and independently review; no workflow status or publication writes.

    The server binds model_call to real isolated communicator/reviewer Sessions
    with durable call IDs. Tests use controlled callbacks, never scientific data
    fabrication in the live application.
    """
    if not callable(model_call) or type(max_rounds)is not int or not 1<=max_rounds<=3:
        raise ValueError('bounded native author capability required')
    problems=[]
    previous_draft=None
    review_blockers=[]
    coverage_failure=False
    content_gap=None
    for attempt in range(max_rounds):
        content_gap=None
        coverage_failure=False
        _control(control_valid)
        if not sources_unchanged(packet['root'],packet):raise ValueError('report source changed before authoring')
        schema=author_stage_schema(packet)
        authored=model_call('author',{'packet':packet,'previous_issues':problems,
            'previous_draft':previous_draft,
            'output_schema':schema,
            'required_output':'Return draft.blocks containing scientific narrative, native tables and relevant verified figures. Do not add unrelated literature comparisons or an unnecessary no-literature-comparison statement. A reference list may retain relevant literature actually obtained in this task; unused bibliography is optional. Do not search for references solely to fill a list when the user did not request them. If the user explicitly requests literature, request task-scoped lookup if needed: the title and DOI/link already obtained in this task suffice for a citation, while factual comparisons and numerical claims require matching source content. Do not invent or imply verification. For native_document sources select an excerpt block with source_id, exact page/block anchor and start/end character offsets; native code inserts the original text. Never treat source_url_declared as verified publisher identity. Preserve truncation and unverified origin/claim limits. Select the source tables and columns relevant to the approved task; preserve the requested results, methods, units, quality and any relevant uncertainty. Do not add unrelated tables. Quantitative prose is optional: use a measurement block selecting table_id, an exact row_selector and columns; the native renderer supplies all material names, values and units. Write interpretation, methods and limitations in narrative blocks without copying measured numbers, file identifiers, paths, hashes or template macros. The native renderer retains the approved report title and complete SourceData links. If previous_draft is present, correct the exact block identified by previous_issues while retaining supported facts and the required source coverage.' + (' In lightweight mode preserve requested sections and source-bound explanations. Mark interpretations that have not been independently verified and do not invent facts.' if packet.get('independent_review_required') is False else '')})
        _control(control_valid)
        if not isinstance(authored,dict) or authored.get('actor')!='communicator' or not authored.get('call_id'):
            raise ValueError('author did not produce a native communicator call receipt')
        original_draft=authored.get('draft')
        strict=packet.get('independent_review_required',True)
        if type(strict)is not bool:raise ValueError('invalid independent review policy')
        draft=original_draft;projection={'omitted_narrative':0,'omitted_headings':0,'kept_source_narrative':0,'missing_sections':[]}
        if not strict:
            blocks=draft.get('blocks') if isinstance(draft,dict) else None
            if isinstance(blocks,list):
                headings={b.get('text') for b in blocks if isinstance(b,dict) and b.get('kind')=='heading'}
                missing=[v for v in packet.get('section_titles',[]) if v not in headings]
                narratives=[b for b in blocks if isinstance(b,dict) and b.get('kind')=='narrative']
                if missing or packet.get('explanatory_content_required') is True and not narratives:
                    content_gap={'omitted_narrative':0,'missing_sections':missing}
                    problems=['Requested sections or explanation are missing; add source-bound content or explicitly report the gap.']
                    previous_draft=copy.deepcopy(original_draft);continue
                if not narratives:
                    sources=packet.get('sources') or []
                    if sources:
                        draft=copy.deepcopy(draft)
                        draft['blocks'].insert(0,{'kind':'narrative','claim_type':'limitation','source_ids':[sources[0]['id']],'text':'本报告展示可追溯的来源数据。'})
        try:markdown=render_validated_draft(packet,draft)
        except ValueError as error:
            problems=[str(error)]
            if str(error) == 'full source table coverage is required' and review_blockers:
                coverage_failure=True
                problems += ['Earlier independent review issues still apply; include every source table using permitted source columns, retaining required context. If constraints cannot be reconciled, request a plan or user decision; do not invent data or omit source coverage.']
                problems += review_blockers
            previous_draft=copy.deepcopy(original_draft)
            continue
        if not sources_unchanged(packet['root'],packet):raise ValueError('report source changed during authoring')
        draft_sha=_digest(draft)
        if not strict:
            proof={'schema_version':1,'mode':'source_numeric','source_packet_sha256':packet['packet_sha256'],
                'draft_sha256':draft_sha,'original_draft_sha256':_digest(original_draft),
                'report_sha256':hashlib.sha256(markdown.encode()).hexdigest(),
                'author_call_id':authored['call_id'],'independent_review_performed':False}
            proof['proof_sha256']=_digest(proof)
            return {'markdown':markdown,'draft':draft,'draft_sha256':draft_sha,'verified':True,
                'author_call_id':authored['call_id'],'review_call_id':None,'review_mode':'source_numeric',
                'original_draft_sha256':_digest(original_draft),'omitted_unverified_blocks':projection['omitted_narrative']+projection['omitted_headings'],
                'independent_review_performed':False,'validation_receipt':proof,
                'packet_sha256':packet['packet_sha256'],'scientific_acceptance':False,
                'visual_review':'pending unless separately established by a real image-input review'}
        review_schema=copy.deepcopy(REVIEW_SCHEMA)
        review_fields=review_schema['properties']['review']['properties']
        review_fields['draft_sha256']['const']=draft_sha
        review_fields['checked_blocks']['const']=list(range(len(draft['blocks'])))
        reviewed=model_call('review',{'packet':packet,'draft':draft,'rendered_markdown':markdown,'draft_sha256':draft_sha,
            'output_schema':review_schema,
            'review_scope':'Check approved report instructions and source relevance. Literature is optional unless the user explicitly requests it; a reference list may retain relevant task-obtained sources. Do not demand unrelated literature comparison or an unnecessary no-comparison statement. For requested bibliography, use titles and DOI/links already obtained in this task without requiring another lookup or PDF; source content is needed for claims about findings, numerical comparisons or agreement. If required evidence is missing, block publication and identify the gap. Classify every block as measured/derived/hypothesis/external/recommendation/limitation. Native excerpt blocks are exact extraction, not validation of scientific claims or publisher identity. Check that interpretation does not exceed the selected excerpt and that truncation and source-origin limits remain explicit. Check numerical fidelity AND quantity meaning, units, causal support, method/convergence scope, citations and actual figure identity. A short MD temperature change cannot establish normality or thermal redistribution; a fixed universal trajectory/cycle minimum cannot be invented. No uncited literature numeric comparisons. Hypotheses must remain explicitly unverified, and inapplicable or missing support must be removed or marked as a gap.'})
        _control(control_valid)
        review=reviewed.get('review')if isinstance(reviewed,dict)else None
        blockers=(review.get('blocking_issues',review.get('issues'))if isinstance(review,dict)else None)
        if isinstance(review,dict)and 'blocking_issues'in review and 'issues'in review:
            # Legacy ambiguous issues retain fail-closed semantics; only the
            # explicit notes field can carry non-blocking observations.
            if isinstance(blockers,list)and isinstance(review['issues'],list):blockers=blockers+review['issues']
            else:blockers=None
        if (not isinstance(review,dict) or reviewed.get('actor')!='report-reviewer' or not reviewed.get('call_id')
                or review.get('draft_sha256')!=draft_sha or review.get('passed')is not True
                or not isinstance(review.get('checked_blocks'),list)
                or any(type(index)is not int for index in review['checked_blocks'])
                or set(review['checked_blocks'])!=set(range(len(draft['blocks'])))
                or not isinstance(blockers,list)or any(not isinstance(item,str)for item in blockers)or blockers):
            problems=['independent report review did not verify the exact complete draft']
            if isinstance(blockers,list):
                reported=[str(item)[:1500] for item in blockers[:16]]
                problems.extend(reported)
                # An untrusted or malformed stage cannot become an
                # authenticated independent-review observation.
                if (isinstance(review,dict) and reviewed.get('actor')=='report-reviewer'
                        and reviewed.get('call_id') and review.get('draft_sha256')==draft_sha
                        and isinstance(review.get('checked_blocks'),list)
                        and all(type(index)is int for index in review['checked_blocks'])
                        and set(review['checked_blocks'])==set(range(len(draft['blocks'])))
                        and all(isinstance(item,str) for item in blockers)):
                    review_blockers=reported
            previous_draft=copy.deepcopy(draft)
            continue
        if not sources_unchanged(packet['root'],packet):raise ValueError('report source changed during review')
        return {'markdown':markdown,'draft':draft,'draft_sha256':draft_sha,'verified':True,
            'author_call_id':authored['call_id'],'review_call_id':reviewed['call_id'],
            'review_mode':'independent_model','independent_review_performed':True,
            'packet_sha256':packet['packet_sha256'],'scientific_acceptance':False,
            'visual_review':'pending unless separately established by a real image-input review'}
    message='report author/review budget exhausted: '+'; '.join(problems)
    if content_gap is not None and not review_blockers:
        raise ReportCapabilityGap('requested narrative or section content lacks source-bound support; report remains incomplete',
            required_action='provide_source_bound_explanation_or_request_independent_review',
            omitted_narrative_count=content_gap['omitted_narrative'],missing_sections=content_gap['missing_sections'])
    if coverage_failure and review_blockers:
        blocks=previous_draft.get('blocks',[]) if isinstance(previous_draft,dict) else []
        used={b.get('table_id') for b in blocks if isinstance(b,dict) and b.get('kind')=='table' and isinstance(b.get('table_id'),str)}
        tables=[{'id':t.get('id'),'available_columns':list(t.get('display_columns') or t.get('columns') or [])} for t in packet.get('tables',[]) if isinstance(t,dict)]
        missing=[t['id'] for t in tables if t['id'] not in used]
        message=('Possible report instruction/evidence conflict: required source tables missing: '+', '.join(missing)+
            '. Preserve source and independent review; choose permitted columns retaining required context, or request an authorized plan revision or user decision. '+message)
        raise ReportContractConflict(message, {'contract_code':'possible_report_constraint_conflict',
            'missing_table_ids':missing,
            'source_tables':tables,'independent_review_issues':review_blockers,
            'resolution':'Retain source evidence and independent review. Consider source-backed column selection that preserves required context; if the instructions and mandatory evidence cannot both be met, ask for an authorized plan revision or user decision. Do not fabricate, omit required coverage, or assert that a revision is approved.'})
    if review_blockers:
        tables=[{'id':t.get('id'),'available_columns':list(t.get('display_columns') or t.get('columns') or [])} for t in packet.get('tables',[]) if isinstance(t,dict)]
        raise ReportReviewUnresolved('Independent report review remains unresolved after bounded attempts. Preserve evidence and seek a supported revision or user decision. '+message,
            {'contract_code':'report_review_unresolved','source_tables':tables,
             'independent_review_issues':review_blockers,
             'resolution':'Review the cited source and the precise critique. The reviewer is not assumed correct. Preserve independent review and user-approved scientific constraints; seek an authorized plan revision or user decision if needed. Do not fabricate or claim acceptance.'})
    raise ValueError(message)


def publish_report(root, output_path, packet, authored, *, control_valid=None):
    """New immutable result only. The native caller validates current goal/attempt.

    Source SHA is checked before and immediately before publication. This
    function does not claim a cross-file transaction against arbitrary writers.
    """
    _control(control_valid)
    if authored.get('verified')is not True or not authored.get('author_call_id'):
        raise ValueError('authored report lacks verified model call receipts')
    text=authored.get('markdown')
    draft=authored.get('draft')
    if (not isinstance(draft,dict) or authored.get('draft_sha256')!=_digest(draft)
            or authored.get('packet_sha256')!=packet.get('packet_sha256')
            or render_validated_draft(packet,draft)!=text):
        raise ValueError('publication does not match the exact reviewed draft')
    if not isinstance(text,str) or not text.strip() or len(text.encode())>MAX_DRAFT:
        raise ValueError('invalid report body')
    from .report_validation_receipt import report_validation_satisfied
    validation={'status':'authored_verified','author_call_id':authored.get('author_call_id'),
        'review_call_id':authored.get('review_call_id'),'review_mode':authored.get('review_mode'),
        'independent_review_performed':authored.get('independent_review_performed'),
        'validation_receipt':authored.get('validation_receipt'),'source_packet_sha256':packet['packet_sha256'],
        'original_draft_sha256':authored.get('original_draft_sha256'),'draft_sha256':authored.get('draft_sha256'),'report_sha256':hashlib.sha256(text.encode()).hexdigest()}
    if not report_validation_satisfied(validation,strict_required=packet.get('independent_review_required',True)):
        raise ValueError('authored report lacks the approved native validation receipt')
    if not sources_unchanged(root,packet):raise ValueError('report source changed before publication')
    rel=_relative(root,output_path);parent,name=_open_parent(root,rel,create=True)
    if packet.get('report_output_path') and str(rel)!=packet['report_output_path']:
        os.close(parent)
        raise ValueError('publication path differs from approved report contract')
    temporary='.report-'+uuid.uuid4().hex
    created=False
    try:
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent);created=True
        with os.fdopen(fd,'wb')as stream:
            stream.write(text.encode());stream.flush();os.fsync(stream.fileno())
        _control(control_valid)
        if not sources_unchanged(root,packet):raise ValueError('report source changed during publication')
        os.link(temporary,name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
        os.fsync(parent)
    finally:
        if created:os.unlink(temporary,dir_fd=parent)
        os.close(parent)
    data_paths={item['path']for item in packet['tables']+packet['figures']}
    for source in packet['sources']:
        data_paths.update(item['path']for item in source.get('structured_data',[]))
        data_paths.update(item['path']for item in source.get('text_documents',[]))
        if source.get('native_document'):data_paths.add(source['native_document']['path'])
    receipt={'status':'authored_verified','report_file':str(Path(root)/rel),'output_path':str(Path(root)/rel),
        'output_files':[str(Path(root)/rel)],'source_steps':[s['step_id']for s in packet['sources']],
        'report_sha256':hashlib.sha256(text.encode()).hexdigest(),'source_packet_sha256':packet['packet_sha256'],
        'report_preview':text[:24000],
        'source_receipts':source_receipts(packet),
        'native_table_bindings':[{k:t[k]for k in('id','path','sha256','parser_sha256','native_identity_sha256','native_source','native_context','row_sources')}
            for t in packet['tables']if t.get('format')=='native_thermo'],
        'figure_bindings':[{k:f[k]for k in('path','sha256','visual_review','visual_evidence')}for f in packet['figures']],
        'author_call_id':authored['author_call_id'],'review_call_id':authored.get('review_call_id'),
        'review_mode':authored.get('review_mode','independent_model'),
        'independent_review_performed':authored.get('independent_review_performed',True),
        'draft_sha256':authored['draft_sha256'],'original_draft_sha256':authored.get('original_draft_sha256'),
        'omitted_unverified_blocks':authored.get('omitted_unverified_blocks',0),
        'validation_receipt':authored.get('validation_receipt'),
        'data_sources':[{k:fact[k]for k in ('path','size','sha256')}for fact in packet['file_facts']
            if fact['path']in data_paths],
        'reference_records':[{k:v for k,v in ref.items()if k not in{'id','source_id'}}for ref in packet['references']],
        'measurement_bindings':[binding for block in draft['blocks']
            if block.get('kind')in{'narrative','measurement'}
            for binding in (_measurement_block(packet,block)[1]if block['kind']=='measurement'
                            else _bind_measurements(packet,block)[1])],
        'excerpt_bindings':[__import__('agents.literature_report_bridge',fromlist=['excerpt_block']).excerpt_block(packet,block)[1]
            for block in draft['blocks']if block.get('kind')=='excerpt']+
            [binding for block in draft['blocks']if block.get('kind')=='narrative'
             for binding in __import__('agents.literature_report_bridge',fromlist=['narrative_excerpt_bindings']).narrative_excerpt_bindings(packet,block)],
        'fact_bindings':[_fact_block(packet,block)[1]for block in draft['blocks']if block.get('kind')=='fact'],
        'scientific_acceptance':False,'workflow_complete':False,'no_existing_results_overwritten':True}
    if packet.get('report_mode')=='fragment'and len(packet['sources'])==1:
        source=packet['sources'][0]
        receipt.update(source_step=source['step_id'],source_evidence_call_id=source['call_id'],
            source_attempt_fingerprint=source.get('execution_fingerprint'),
            evidence_fingerprint=source['source_sha256'],fragment_fingerprint=receipt['report_sha256'])
    return receipt


def _nonquantitative_literature_denial(text):
    """Only explicit nonnumeric denial clauses; never attest external results.

    Positive or mixed clauses still require excerpts. Independent scientific
    review and all numerical/measurement checks remain separate obligations.
    """
    topic=r'\bliterature\b|\bpublished\b|\bprevious(?:ly| studies?)\b.{0,24}\breported\b|文献|前人|(?:已有|既有|既往|先前|此前|过往|其他|已发表)(?:的)?(?:研究|工作|报道|论文|结果)'
    clauses=_scientific_claim_clauses(text)
    mentions=[clause for clause in clauses if re.search(topic,clause)]
    if not mentions:return False
    # A literature denial cannot authorize an independent, unresolved
    # statement elsewhere in the same model-authored block. Only a complete
    # simple absence/provenance clause may accompany it; unknown text must be
    # split or supplied with evidence.
    harmless=(r'(?:无|没有|未提供|未发现)(?:相应|相关|额外)?(?:证据|资料|数据|结果)'
              r'|(?:必须|应|只能)由(?:对应的)?(?:原生产出步骤|原生来源|原始来源)提供(?:证据|证据块)'
              r'|(?:任何)?与(?:实验等温线|其他力场结果|已发表数据)(?:(?:、|或|和|及)(?:实验等温线|其他力场结果|已发表数据)){0,12}的比较(?:均)?超出(?:本片段|本报告|当前)(?:的)?证据范围'
              r'|(?:此处|本片段|本报告|本文)(?:亦|也)?不(?:作|做)(?:此类)?(?:比较|对比|对照)'
              r'|(?:也|并)?不(?:将|把)(?:本来源表|本表|来源数据|本结果)与(?:外部数据集|其他数据集|外部数据|其他数据)并列')
    if any(not re.search(topic,clause) and not re.fullmatch(harmless,clause.strip()) for clause in clauses):return False
    positive=r'一致|吻合|相符|\bagree(?:s|ment)?\b|\bconsistent with\b|\bmatch(?:es)?\b'
    substitution=r'(?:而)?不是由(?:外部)?(?:文献|前人)(?:值|数值|数据|结果)替代'
    def bounded_context_denial(clause):
        return bool(re.fullmatch(r'(?:上述读数|这些读数|本次观察|单次观察|当前结果)(?:不足以|不能|无法)(?:支撑|支持)(?:任何)?跨材料(?:比较|对比)或与文献(?:值|结果)的一致性判断',clause.strip()))
    for clause in mentions:
        if bounded_context_denial(clause):continue
        # Multiple comparison predicates are ambiguous without a full parse;
        # do not let one negation cover an independent later assertion.
        if len(list(re.finditer(r'比较|对比|对照|比对',clause)))>1:return False
        # Reversal of a negation is not an explicit denial. Apply before
        # any existing denial alternative can match its leading operator.
        if len(clause)>512 or len(re.findall(r'没有|没|不|未|非|无',clause))>1:return False
        comparisons=list(re.finditer(r'比较|对比|对照|比对',clause))
        # Any following statement can introduce a second claim outside the
        # negated predicate; unknown syntax fails closed.
        if comparisons and clause[comparisons[0].end():].strip():return False
        if re.search(r'(?:不含|未包含|没有)(?:任何)?(?:参考)?文献(?:条目|记录|证据)|(?:不|没有|未)做.{0,24}(?:文献|前人|比对|比较|对照)',clause) and re.search(positive,clause):
            return False
        if re.search(substitution,clause):
            remainder=re.sub(substitution,'',clause)
            if re.search(topic,remainder) or re.search(positive,remainder):return False
    neutral=r'(?:(?:本次|本任务|本片段|本研究|本文|本报告|该研究|证据包中|来源中|单次观察|此处|这里|我们|也|亦|并|尚|则|仍|而)\s*)*'
    def whole_denial(clause):
        # Accept only complete simple grammar. Unknown entities, extra verbs,
        # mixed predicates and long ambiguous prose remain evidence-required.
        entity=(r'(?:已发表文献|其他材料|已有研究|既往研究|先前研究|相关研究|先前报道|此前报道|实验数据|模拟数据|文献|前人|材料|实验|模拟|数据|结果)'
                r'(?:的?(?:数值|数据|结果))?')
        objects=r'(?:这些|上述|相关)?'+entity+r'(?:(?:、|和|与|及|或|同)'+entity+r'){0,12}'
        modifier=r'(?:直接|定量|任何|相关|相应|额外|任何额外){0,3}'
        comparison=r'(?:比较|对比|对照|比对)'
        negative=r'(?:没有|不能|无法|未|不)'
        if bounded_context_denial(clause):return True
        grammar=(
            negative+r'(?:将|把|拿)(?:其|结果|输出|数据|该结果|本结果)(?:与|同|和)'+objects+r'(?:作|进行|做)?'+modifier+comparison,
            negative+r'(?:与|同|对)'+objects+r'(?:作|进行|做)?'+modifier+comparison,
            negative+r'(?:做|进行|作|开展)'+modifier+r'(?:文献|前人)(?:层面)?(?:的)?'+comparison,
            negative+r'(?:包含)(?:任何)?(?:与|同)'+objects+r'的'+modifier+comparison,
            negative+r'(?:作|做)(?:肯定或否定的)(?:文献|前人)判断',
            r'(?:不含|未包含|没有)(?:任何)?(?:参考)?文献(?:条目|记录|证据)',
            negative+r'(?:引入|使用|采用)(?:任何|额外|任何额外)?(?:模拟或)?(?:外部)?(?:文献|前人)(?:数值|数据|结果)(?:或外部规范名称)?',
            r'(?:而)?不是由(?:外部)?(?:文献|前人)(?:值|数值|数据|结果)替代',
            r'\b(?:no|without)\s+(?:(?:direct|quantitative|valid|supported)\s+)?(?:literature comparison|comparison with (?:the )?literature|claim of agreement)(?: is made)?\b',
            r'(?:the observations|the evidence|these observations|these results) cannot support agreement with (?:published|literature) (?:values|results)',
            r'(?:future work|further work) needs literature (?:excerpt )?evidence before comparison',
            r'(?:未来|后续)(?:需要|应当|必须)(?:补充|提供|获取)文献(?:摘录|证据|资料)(?:后|之后)才能(?:比较|对比)',
        )
        return any(re.fullmatch(neutral+'(?:'+pattern+')',clause.strip()) for pattern in grammar)

    return all(not _numbers(clause) and not re.search(r'\d',clause)
               and whole_denial(clause) for clause in mentions)
