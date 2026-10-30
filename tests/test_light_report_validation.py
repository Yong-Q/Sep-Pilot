import copy
import pytest
from agents import report_authoring as report
from agents.report_validation_receipt import report_validation_satisfied
from tests.test_report_native_bridge import fixture


def make_packet(tmp_path,requirements=None):
    root,_,_,state,attest,_,_,add=fixture(tmp_path);add('report',['figure'])
    if requirements is not None:state['nodes']['report']['contract']['arguments']['report_requirements']=requirements
    packet=report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)
    return root,state,packet


def model_for(packet,seen,*,review_ok=True):
    blocks=([{'kind':'narrative','text':'The source evidence has limited scope.','source_ids':['S1'],'claim_type':'limitation'}]
        if packet.get('independent_review_required',True) else [{'kind':'heading','text':'方法'}])
    blocks += [{'kind':'table','table_id':table['id']} for table in packet['tables']]
    if packet.get('minimum_figures'):blocks += [{'kind':'figure','figure_id':figure['id']}for figure in packet['figures'][:packet['minimum_figures']]]
    draft={'blocks':blocks}
    def model(phase,payload):
        seen.append(phase)
        if phase=='author':return {'actor':'communicator','call_id':'author-real','draft':draft}
        return {'actor':'report-reviewer','call_id':'review-real','review':{'passed':review_ok,'draft_sha256':payload['draft_sha256'],
            'checked_blocks':list(range(len(draft['blocks']))),'blocking_issues':[] if review_ok else ['review rejected']}}
    return model


def test_default_authors_once_binds_real_sources_and_marks_no_model_review(tmp_path):
    root,state,packet=make_packet(tmp_path);assert packet['independent_review_required'] is False;seen=[]
    authored=report.author_report(packet,model_call=model_for(packet,seen));assert seen==['author']
    receipt=report.publish_report(root,state['nodes']['report']['contract']['arguments']['output_path'],packet,authored)
    assert receipt['review_call_id'] is None and receipt['independent_review_performed'] is False
    assert receipt['review_mode']=='source_numeric' and report_validation_satisfied(receipt)
    assert receipt['scientific_acceptance'] is False and 'Unknown convergence remains unknown' in authored['markdown']
    assert not report_validation_satisfied(receipt,strict_required=True)


def test_explicit_strict_review_calls_real_reviewer_and_rejects_forgery(tmp_path):
    root,state,packet=make_packet(tmp_path,{'independent_review':True});seen=[]
    authored=report.author_report(packet,model_call=model_for(packet,seen));assert seen==['author','review']
    receipt=report.publish_report(root,state['nodes']['report']['contract']['arguments']['output_path'],packet,authored)
    assert receipt['independent_review_performed'] is True and receipt['review_call_id']=='review-real'
    assert report_validation_satisfied(receipt,strict_required=True)
    forged=copy.deepcopy(receipt);forged['review_call_id']=None;forged['review_mode']='source_numeric';forged['independent_review_performed']=False
    assert not report_validation_satisfied(forged,strict_required=True)


def test_explicit_figure_must_appear_but_default_does_not_require_figure(tmp_path):
    _,_,packet=make_packet(tmp_path,{'minimum_figures':1});assert packet['minimum_figures']==1
    seen=[];model=model_for(packet,seen);authored=report.author_report(packet,model_call=model)
    assert '![F1]' in authored['markdown']
    draft=copy.deepcopy(authored['draft']);draft['blocks']=[b for b in draft['blocks'] if b.get('kind')!='figure']
    with pytest.raises(ValueError,match='requires additional figure blocks'):report.render_validated_draft(packet,draft)


def test_light_receipt_tampering_is_rejected(tmp_path):
    root,state,packet=make_packet(tmp_path);authored=report.author_report(packet,model_call=model_for(packet,[]))
    receipt=report.publish_report(root,state['nodes']['report']['contract']['arguments']['output_path'],packet,authored)
    for field in ('source_packet_sha256','draft_sha256','report_sha256','author_call_id'):
        changed=copy.deepcopy(receipt);changed[field]='f'*64;assert not report_validation_satisfied(changed)


def test_source_change_between_author_and_publish_never_writes_output(tmp_path):
    root,state,packet=make_packet(tmp_path);authored=report.author_report(packet,model_call=model_for(packet,[]))
    (root/'solver.log').write_text('tampered')
    output=state['nodes']['report']['contract']['arguments']['output_path']
    with pytest.raises(ValueError,match='source changed'):report.publish_report(root,output,packet,authored)
    assert not __import__('os').path.lexists(output)


def test_strict_review_rejects_conflicting_or_malformed_receipt():
    base={'status':'authored_verified','author_call_id':'author','review_call_id':'review'}
    assert report_validation_satisfied(base,strict_required=True)  # Historical reviewed receipt.
    assert not report_validation_satisfied({**base,'review_mode':'independent_model','independent_review_performed':False},strict_required=True)
    assert not report_validation_satisfied({**base,'review_mode':'unknown','independent_review_performed':True},strict_required=True)
    assert not report_validation_satisfied({**base,'review_call_id':'../../bad','review_mode':'independent_model','independent_review_performed':True},strict_required=True)
    assert report_validation_satisfied({**base,'review_mode':'independent_model','independent_review_performed':True},strict_required=True)


@pytest.mark.parametrize('text',[
 '对报告严格地进行审核', '请独立仔细审查报告', '请做严格的科学与同行评审',
 'Please review independently and rigorously', '不要独立审核而是严格审核',
 '不用独立审核并要求严格审核', '不要进行独立审核，请严格审核', '不要省略独立审核',
])
def test_explicit_strict_user_intent_is_preserved(text):
    from agents.goal_contract import GoalContract
    from agents.report_review_intent import explicit_strict_review
    assert explicit_strict_review(GoalContract.from_user_message(text).to_dict())


def test_explicit_strict_goal_cannot_be_overridden_by_false_field(tmp_path):
    from agents.goal_contract import GoalContract
    root,_,_,state,attest,_,_,add=fixture(tmp_path);add('report',['figure'])
    goal=GoalContract.from_user_message('请独立仔细审查报告');state['goal_contract']=goal.to_dict()
    state['nodes']['report']['contract']['arguments']['report_requirements']={'independent_review':False}
    packet=report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)
    assert packet['independent_review_required'] is True


@pytest.mark.parametrize('claim',['该材料存在氢键网络','该结构具有显著的双峰孔径分布'])
def test_unsourced_narrative_never_becomes_completed_light_report(tmp_path,claim):
    root,state,packet=make_packet(tmp_path);draft={'blocks':[{'kind':'narrative','text':claim,'source_ids':['S1'],'claim_type':'result'},
        *[{'kind':'table','table_id':t['id']}for t in packet['tables']]]}
    def author(phase,payload):assert phase=='author';return {'actor':'communicator','call_id':'author-real','draft':draft}
    with pytest.raises(ValueError,match='result statement requires a source-bound measurement or excerpt'):report.author_report(packet,model_call=author)
    assert not __import__('os').path.lexists(state['nodes']['report']['contract']['arguments']['output_path'])


@pytest.mark.parametrize('goal',['请详细解释吸附行为并写成报告','请给出中文分析与科学讨论','解释这些现象并形成报告',
 '请分析分子的扩散','Please explain adsorption and make a report','请写一份带讨论的报告'])
def test_explicit_explanation_without_source_bound_prose_is_incomplete(tmp_path,goal):
    root,_,_,state,attest,_,_,add=fixture(tmp_path);add('report',['figure']);state['goal_contract']['original_goal']=goal
    # No reliance on the model to mark a structured requirement.
    p=report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)
    assert p.get('explanatory_content_required') is True
    with pytest.raises(report.ReportCapabilityGap,match='report remains incomplete'):report.author_report(p,model_call=model_for(p,[]))


def test_neutral_and_approved_sections_are_preserved(tmp_path):
    root,_,_,state,attest,_,_,add=fixture(tmp_path);add('report',['figure'])
    state['nodes']['report']['contract']['arguments']['report_requirements']={'section_titles':['实验条件']}
    p=report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)
    draft={'blocks':[{'kind':'heading','text':'实验条件'}, *[{'kind':'table','table_id':t['id']} for t in p['tables']]]}
    def author(phase,payload):return {'actor':'communicator','call_id':'author-real','draft':draft}
    result=report.author_report(p,model_call=author);assert '## 实验条件' in result['markdown']
    draft['blocks'].pop(0)
    with pytest.raises(report.ReportCapabilityGap,match='report remains incomplete'):report.author_report(p,model_call=author)


def test_exact_local_excerpt_is_preserved_as_source_bound_explanation(monkeypatch):
    from tests.test_native_document_links import packet as document_packet
    p=document_packet();p.update(root='/unused',independent_review_required=False,explanatory_content_required=True)
    p['packet_sha256']=report._digest(p);monkeypatch.setattr(report,'sources_unchanged',lambda *a:True)
    text='Original evidence text.';draft={'blocks':[{'kind':'heading','text':'方法'},
        {'kind':'narrative','claim_type':'limitation','source_ids':['S1'],'text':text,
         'excerpt_refs':[{'source_id':'S1','anchor':'page-1','start':0,'end':len(text)}]}]}
    def author(phase,payload):return {'actor':'communicator','call_id':'author-real','draft':draft}
    result=report.author_report(p,model_call=author)
    assert text in result['markdown'] and '## 方法' in result['markdown'] and result['omitted_unverified_blocks']==0
    draft['blocks'][1]['text']='The evidence proves an invented mechanism.'
    with pytest.raises(ValueError,match='report author/review budget exhausted'):report.author_report(p,model_call=author)


def test_source_only_omission_requires_explicit_user_intent(tmp_path):
    root,_,_,state,attest,_,_,add=fixture(tmp_path);add('report',['figure'])
    state['goal_contract']['original_goal']='只输出数据，不需要解释'
    state['nodes']['report']['contract']['arguments']['report_requirements']={'source_only':True}
    p=report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)
    draft={'blocks':[{'kind':'narrative','text':'该材料存在氢键网络','claim_type':'result','source_ids':['S1']},
                     *[{'kind':'table','table_id':t['id']}for t in p['tables']]]}
    with pytest.raises(ValueError,match='result statement requires a source-bound measurement or excerpt'):
        report.author_report(p,model_call=lambda phase,payload:{'actor':'communicator','call_id':'author-real','draft':draft})
    draft['blocks'].pop(0)
    result=report.author_report(p,model_call=lambda phase,payload:{'actor':'communicator','call_id':'author-real','draft':draft})
    assert '氢键网络' not in result['markdown'] and result['omitted_unverified_blocks']==0
    state['goal_contract']['original_goal']='请写一份报告'
    with pytest.raises(ValueError,match='requires explicit user request'):
        report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)


def test_explicit_review_cancellation_and_later_reaffirmation():
    from agents.report_review_intent import explicit_strict_review
    assert not explicit_strict_review({'original_goal':'不需要独立审核','directives':[]})
    assert explicit_strict_review({'original_goal':'不需要独立审核','directives':['但请严格审核']})
    assert not explicit_strict_review({'original_goal':'请独立审核','directives':['无需独立审核']})


def test_later_explanation_request_cannot_be_overridden_by_source_only(monkeypatch):
    from tests.test_native_document_links import packet as document_packet
    p=document_packet();p.update(root='/unused',independent_review_required=False,explanatory_content_required=True,source_only_requested=True)
    p['packet_sha256']=report._digest(p);monkeypatch.setattr(report,'sources_unchanged',lambda *a:True)
    text='Original evidence text.';draft={'blocks':[{'kind':'narrative','claim_type':'limitation','source_ids':['S1'],'text':text,
        'excerpt_refs':[{'source_id':'S1','anchor':'page-1','start':0,'end':len(text)}]},
        {'kind':'narrative','claim_type':'result','source_ids':['S1'],'text':'An unsupported additional conclusion.'}]}
    def author(phase,payload):return {'actor':'communicator','call_id':'author-real','draft':draft}
    with pytest.raises(ValueError,match='result statement requires a source-bound measurement or excerpt'):report.author_report(p,model_call=author)


@pytest.mark.parametrize('text',['不要只输出数据','不能仅提供数据','Do not only output data','不要只输出数据而是解释结果', '只输出数据，但写完整报告', '只输出数据。不过仍要完整报告', '只输出数据。撤销先前限制', '不要把报告限制为只输出数据','不是为了只输出数据','禁止仅提供数据','不应该只输出数据','并非真的只输出数据'])
def test_negated_source_only_is_never_treated_as_permission(text):
    from agents.report_review_intent import explicit_source_only
    assert not explicit_source_only({'original_goal':text,'directives':[]})


def test_source_only_latest_explicit_user_change_is_respected():
    from agents.report_review_intent import explicit_source_only
    assert explicit_source_only({'original_goal':'请写报告','directives':['只输出数据']})
    assert not explicit_source_only({'original_goal':'只输出数据','directives':['不要只输出数据']})
    for later in ['改为完整报告','不要了，写完整报告','还是给我完整版','撤销先前限制']:
        assert not explicit_source_only({'original_goal':'只输出数据','directives':[later]})


def test_report_continuation_accepts_new_contract_fields_without_weakening_types():
    from agents.presentation_continuation import valid_report_requirements
    assert valid_report_requirements({'minimum_citations':1,'independent_review':True,'section_titles':['方法']})
    assert valid_report_requirements({'source_only':False,'require_explanatory_narrative':True,'minimum_figures':0})
    assert valid_report_requirements({'minimum_figures':17})
    for bad in ({'minimum_citations':True},{'independent_review':1},{'source_only':'true'},
                {'minimum_figures':-1},{'section_titles':['a','a']},{'unknown':True}):
        assert not valid_report_requirements(bad)
