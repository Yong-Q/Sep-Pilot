"""Source-bound native extraction, never model-written full text or URL proof."""
import re
import math

EXCERPT_SCHEMA={'type':'object','additionalProperties':False,
    'required':['kind','source_id','anchor','start','end'],
    'description':'Quote an exact bounded excerpt of an attested native document page/block. Offsets select existing text; no model transcription. This does not verify publisher identity or scientific claims.',
    'properties':{'kind':{'const':'excerpt'},'source_id':{'type':'string'},
        'anchor':{'type':'string'},'start':{'type':'integer','minimum':0},
        'end':{'type':'integer','minimum':1}}}


def attach_native_document(root, result, summary, budget, file_facts, read):
    if result.get('ok') is not True or result.get('status')!='text_extracted':
        raise ValueError('reader has no successful native extracted text')
    original=result.get('source_document')or{}
    from .document_source_facts import stream_document_fact
    prefix,fact=stream_document_fact(root,original.get('path'),budget)
    if (fact['sha256']!=original.get('sha256')or fact['size']!=original.get('bytes')
            or result.get('embedded_actions_executed')is not False):
        raise ValueError('native extracted text source integrity mismatch')
    kind=result.get('document_type')
    if kind=='pdf':
        if b'%PDF-'not in prefix:raise ValueError('native document is not PDF')
        blocks=result.get('pages');pattern=r'page-[1-9][0-9]*';number='page'
    elif kind=='html':
        blocks=result.get('blocks');pattern=r'html-block-[1-9][0-9]*';number='block'
    else:raise ValueError('unsupported native extracted document')
    if not isinstance(blocks,list)or not 1<=len(blocks)<=100:
        raise ValueError('native reader page/block bounds invalid')
    seen=set();documents=[];characters=0
    for block in blocks:
        anchor=block.get('anchor');text=block.get('text')
        if (not isinstance(text,str)or not isinstance(anchor,str)or not re.fullmatch(pattern,anchor)
                or anchor in seen or type(block.get(number))is not int
                or anchor.rsplit('-',1)[-1]!=str(block[number])):
            raise ValueError('native reader anchor invalid')
        seen.add(anchor);characters+=len(text)
        documents.append({'anchor':anchor,number:block[number],'untrusted_content':text})
    if characters==0 or characters>200000 or characters!=result.get('extractable_characters'):
        raise ValueError('native extracted character bounds invalid')
    file_facts.append(fact)
    summary['native_document']={'path':fact['path'],'sha256':fact['sha256'],'format':kind,
        'blocks':documents,'truncated':result.get('truncated')is not False,
        'source_origin_verified':False,'scientific_claims_verified':False,
        'parser_implementation_sha256':result.get('parser_implementation_sha256'),
        'next_page':result.get('next_page'),'next_block':result.get('next_block')}
    summary.setdefault('comparison_limits',[]).append(
        'Native extracted excerpts support only the displayed page/block text. Publisher identity, source URL and scientific claims remain unverified; extraction may be incomplete.')
    return 1



def excerpt_block(packet, block):
    if block.get('kind')!='excerpt' or set(block)!={'kind','source_id','anchor','start','end'}:raise ValueError('invalid excerpt selection')
    from .report_authoring import resolved_source_ids, _plain
    source_id=resolved_source_ids(packet,[block['source_id']])[0]
    source=next(item for item in packet['sources']if item['id']==source_id)
    document=source.get('native_document')or{}
    page=next((item for item in document.get('blocks',[])if item['anchor']==block['anchor']),None)
    def exact_offset(value):
        # JSON Schema integer accepts 1120.0 as an integer. Normalize only
        # finite integral JSON numbers; bools and fractional values fail.
        if type(value) is int: return value
        if type(value) is float and math.isfinite(value) and value.is_integer(): return int(value)
        return None
    start,end=exact_offset(block['start']),exact_offset(block['end'])
    if (page is None or type(start)is not int or type(end)is not int
            or not 0<=start<end<=len(page['untrusted_content'])or end-start>2000):
        raise ValueError('excerpt must select bounded original native page text')
    text=page['untrusted_content'][start:end]
    binding={'source_id':source_id,'path':document['path'],'sha256':document['sha256'],
        'anchor':block['anchor'],'start':start,'end':end,'excerpt':text,
        'source_origin_verified':False,'scientific_claims_verified':False}
    return ('> '+_plain(text).replace('\n','\n> ')+'\n\n['+source_id+'; '+block['anchor']+
        '] Native extracted excerpt; source origin and scientific claims unverified.'),binding


def attach_reader_ancestor(root,state,result,summary,budget,file_facts,read):
    """Called only after recursive native source-receipt validation succeeded."""
    pending=list(result.get('source_receipts')or[]);readers={};seen=set()
    while pending:
        receipt=pending.pop();step=receipt.get('step_id')
        if step in seen:continue
        seen.add(step)
        if len(seen)>64:raise ValueError('document ancestry exceeds bounded limit')
        node=state.get('nodes',{}).get(step)or{}
        tool=(node.get('contract')or{}).get('tool')
        from .report_authoring import _result
        original=_result(node.get('result'))
        if tool=='read_scientific_document':readers[step]=original
        elif tool in{'generate_scientific_report','generate_result_figures'}:
            pending.extend(original.get('source_receipts')or[])
    if not readers:return 0
    if len(readers)>1:
        raise ValueError('document assembly needs one reader per source fragment; split document branches')
    step,reader=next(iter(readers.items()))
    attached=attach_native_document(root,reader,summary,budget,file_facts,read)
    summary['native_document']['reader_step_id']=step
    return attached


def narrative_excerpt_bindings(packet,block):
    """A claim may point at exact text without inventing publisher verification."""
    values=block.get('excerpt_refs')or[]
    if not isinstance(values,list)or len(values)>8:raise ValueError('bounded excerpt claim references required')
    bindings=[]
    for value in values:
        if not isinstance(value,dict):raise ValueError('invalid excerpt claim reference')
        _,binding=excerpt_block(packet,{'kind':'excerpt',**value})
        bindings.append(binding)
    return bindings
