"""Keep only source-bound report content without inventing qualitative facts."""
import copy


_NEUTRAL_HEADINGS={'方法','研究方法','计算方法','条件','计算条件','结果','计算结果','数据','讨论','局限','限制','来源','参考文献',
                   'method','methods','conditions','results','data','discussion','limitations','sources','references'}


def project_light_draft(packet,draft):
    if not isinstance(draft,dict) or not isinstance(draft.get('blocks'),list):
        raise ValueError('invalid bounded report draft')
    sources=packet.get('sources')or[]
    if not sources or not isinstance(sources[0],dict) or not sources[0].get('id'):
        raise ValueError('light report requires an owned source')
    requested=packet.get('section_titles')or[];keep=[];omitted_narrative=0;omitted_headings=0;kept_source_narrative=0;seen_sections=set()
    for block in draft['blocks']:
        if not isinstance(block,dict):keep.append(copy.deepcopy(block));continue
        kind=block.get('kind')
        if kind=='heading':
            title=block.get('text')
            if isinstance(title,str) and title.strip().lower() in _NEUTRAL_HEADINGS:
                keep.append(copy.deepcopy(block));seen_sections.add(title)
            elif isinstance(title,str) and title in requested:
                # Preserve the explicit approved section name without implying
                # that its wording is a verified scientific conclusion.
                item=copy.deepcopy(block);item['text']=title+'（用户指定，未核实）';keep.append(item);seen_sections.add(title)
            else:omitted_headings+=1
        elif kind=='narrative':
            from .literature_report_bridge import narrative_excerpt_bindings
            try:bindings=narrative_excerpt_bindings(packet,block)
            except (ValueError,PermissionError,KeyError,TypeError):bindings=[]
            text=block.get('text');ids=block.get('source_ids')or[]
            exact=(isinstance(text,str) and any(isinstance(b,dict) and b.get('source_id') in ids
                and isinstance(b.get('excerpt'),str) and text.strip()==b['excerpt'].strip() for b in bindings))
            if exact:keep.append(copy.deepcopy(block));kept_source_narrative+=1
            else:omitted_narrative+=1
        else:keep.append(copy.deepcopy(block))
    if not any(isinstance(block,dict) and block.get('kind') in {'table','fact','measurement','figure','citation','excerpt'} for block in keep) and not kept_source_narrative:
        raise ValueError('light report needs at least one source-bound table, fact, measurement, figure, citation or excerpt')
    notice={'kind':'narrative','claim_type':'limitation','source_ids':[sources[0]['id']],
        'text':'本报告展示可追溯的来源数据。' + (' 未逐句核实的模型解释已省略。' if omitted_narrative else '')}
    projected={key:copy.deepcopy(value) for key,value in draft.items() if key!='blocks'}
    projected['blocks']=[notice]+keep
    return projected,{'omitted_narrative':omitted_narrative,'omitted_headings':omitted_headings,
                      'kept_source_narrative':kept_source_narrative,'missing_sections':[x for x in requested if x not in seen_sections]}
