import pytest
from agents.report_authoring import render_validated_draft,_nonquantitative_literature_denial
from tests.test_report_inference_label import packet,narrative

@pytest.mark.parametrize('text',[
 '本片段不包含任何与实验或已发表文献的定量比较，也不把本来源表与外部数据集并列。',
 '未与已有研究比较。',
 '本任务未做任何文献层面的对比，也未引入外部文献数值或外部规范名称；任何与实验等温线、其他力场结果或已发表数据的比较均超出本片段证据范围，此处不作此类比较，本片段亦不作肯定或否定的文献判断。',
])
def test_real_pure_denials_need_no_literature(text):assert _nonquantitative_literature_denial(text)

@pytest.mark.parametrize('text',[
 '本研究未做文献比较，结果优于它。',
 '本研究未做文献比较。结果优于其数据。',
 '未做文献比较，但本方法更好。',
 '并非没有做文献比较。',
 '没有任何理由在本次完整研究工作当中不与文献比较。',
 '本片段不包含文献比较，但结果与文献一致。',
 '不将其与文献数值比较，结果优于文献。',
 '不与文献中的 2.4 作比较。',
 '未做文献层面对比并显示结果优于文献。',
])
def test_affirmative_double_negative_or_numeric_never_exempt(text):assert not _nonquantitative_literature_denial(text)


def test_unused_bibliography_is_optional_for_neutral_report():
    p=packet();p['references']=[{'id':'R1','source_id':'S1','support_kind':'bibliographic_only','snippet':'unverified'}]
    assert 'Source evidence has limited scope' in render_validated_draft(p,narrative('method','Source evidence has limited scope.'))


def test_positive_comparison_without_excerpt_still_rejected():
    p=packet();p['references']=[{'id':'R1','source_id':'S1','support_kind':'bibliographic_only','snippet':'unverified'}]
    with pytest.raises(ValueError,match='literature comparison lacks'):
        render_validated_draft(p,narrative('inference','结果与文献一致。'))


def test_optional_absent_literature_passes_normal_author_and_review(tmp_path,monkeypatch):
    from agents import report_authoring as m
    from tests.test_report_fact_pointer_diagnostics import packet as native_packet,fact
    p=native_packet(tmp_path);monkeypatch.setattr(m,'sources_unchanged',lambda *args:True)
    draft={'blocks':[{'kind':'narrative','text':'The source verification is limited.','claim_type':'limitation','source_ids':['S1']},fact('/native_info/converged')]}
    seen=[]
    def model(phase,payload):
        seen.append(phase)
        if phase=='author':
            assert 'bibliography is optional' in payload['required_output']
            assert 'do not add unrelated literature comparisons' in payload['required_output'].lower()
            assert 'unnecessary no-literature-comparison' in payload['required_output']
            return {'actor':'communicator','call_id':'author','draft':draft}
        assert 'unless the user explicitly requests it' in payload['review_scope']
        assert 'unnecessary no-comparison statement' in payload['review_scope']
        return {'actor':'report-reviewer','call_id':'review','review':{'passed':True,'draft_sha256':payload['draft_sha256'],'checked_blocks':[0,1],'blocking_issues':[]}}
    result=m.author_report(p,model_call=model);assert result['verified'] is True and seen==['author','review']


def test_explicit_literature_request_with_missing_evidence_remains_review_blocked(tmp_path,monkeypatch):
    from agents import report_authoring as m
    from tests.test_report_fact_pointer_diagnostics import packet as native_packet,fact
    p=native_packet(tmp_path);p['instructions']='请引用并核对文献原文';monkeypatch.setattr(m,'sources_unchanged',lambda *args:True)
    draft={'blocks':[{'kind':'narrative','text':'The source verification is limited.','claim_type':'limitation','source_ids':['S1']},fact('/native_info/converged')]}
    def model(phase,payload):
        if phase=='author':return {'actor':'communicator','call_id':'author','draft':draft}
        assert 'unless the user explicitly requests it' in payload['review_scope']
        return {'actor':'report-reviewer','call_id':'review','review':{'passed':False,'draft_sha256':payload['draft_sha256'],'checked_blocks':[0,1],'blocking_issues':['Requested verified literature is missing.']}}
    with pytest.raises(ValueError,match='Requested verified literature is missing'):
        m.author_report(p,model_call=model)


def test_real_excerpt_still_required_for_positive_literature_claim():
    p=packet();p['references']=[{'id':'R1','source_id':'S1','support_kind':'supplied_excerpt','snippet':'Published source reports 9.'}]
    draft=narrative('external','Published source reports 9.');draft['blocks'][0]['reference_ids']=['R1']
    result=render_validated_draft(p,draft);assert '[R1]' in result
    p['references'][0]['support_kind']='bibliographic_only'
    with pytest.raises(ValueError,match='literature comparison lacks'):render_validated_draft(p,draft)


def test_tool_catalog_describes_optional_citations_and_preserves_web_tools():
    from agents import registry
    description=registry._SCHEMAS['generate_scientific_report']['description']
    assert 'Do not introduce unrelated literature' in description
    assert 'query_literature' in description
    for name in ('query_literature','download_scientific_file','read_scientific_document'):
        assert name in registry._SCHEMAS and name in registry._EXECUTORS
    assert registry._SCHEMAS['query_literature']['schema']['properties']['web_only']['type']=='boolean'


@pytest.mark.parametrize('text',[
 '相较已有研究结果，本结果更优。',
 'Smith 等人研究显示本方法表现更好。',
 '先前报道表明这一结果与已有结果相符。',
 'The results agree with Smith et al.', 'Prior studies show the result is better.',
])
def test_unverified_external_assertions_still_need_actual_source_content(text):
    with pytest.raises(ValueError,match='literature comparison lacks'):
        render_validated_draft(packet(),narrative('inference',text))


def test_claims_about_current_research_are_not_automatically_literature():
    result=render_validated_draft(packet(),narrative('method','本研究采用所提供的计算方法。'))
    assert '本研究采用' in result
    assert 'LAMMPS' in render_validated_draft(packet(),narrative('method','本研究使用 LAMMPS 等工具。'))


def test_explicit_citation_contract_cannot_be_bypassed_by_positive_reviewer(tmp_path,monkeypatch):
    from agents import report_authoring as m
    from tests.test_report_fact_pointer_diagnostics import packet as native_packet,fact
    p=native_packet(tmp_path);p['minimum_citations']=1
    p['references']=[{'id':'R1','source_id':'S1','title':'Task record','url':'https://doi.org/10.1234/example',
                      'support_kind':'bibliographic_only','snippet':''}]
    monkeypatch.setattr(m,'sources_unchanged',lambda *args:True)
    draft={'blocks':[{'kind':'narrative','text':'The source verification is limited.','claim_type':'limitation','source_ids':['S1']},fact('/native_info/converged')]}
    def model(phase,payload):
        if phase=='author':return {'actor':'communicator','call_id':'author','draft':draft}
        pytest.fail('missing required citations must be rejected before review')
    with pytest.raises(ValueError,match='requires additional task-sourced citation blocks'):m.author_report(p,model_call=model)


def test_existing_task_bibliographic_record_counts_without_fulltext_or_extra_check():
    p=packet();p['minimum_citations']=1;p['references']=[{'id':'R1','source_id':'S1','title':'Task record',
        'url':'https://doi.org/10.1234/example','support_kind':'bibliographic_only','snippet':''}]
    d=narrative('limitation','The source evidence has limited scope.');d['blocks'].append({'kind':'citation','reference_id':'R1'})
    rendered=render_validated_draft(p,d);assert 'Task record' in rendered and 'https://doi.org/10.1234/example' in rendered


def test_task_literature_tool_records_not_arbitrary_result_as_citations():
    from agents.report_authoring import _references
    result={'results':[{'title':'Existing task record','doi':'10.1234/example','url':'https://doi.org/10.1234/example'}]}
    assert _references(result,native_tool='run_cdft')==[]
    references=_references(result,native_tool='query_literature');assert len(references)==1 and references[0]['support_kind']=='bibliographic_only'


def test_existing_doi_or_link_without_title_needs_no_extra_lookup():
    from agents.report_authoring import _references
    results={'results':[{'doi':'10.1234/existing'},{'url':'https://example.org/existing-paper'}]}
    records=_references(results,native_tool='query_literature')
    assert [r['title'] for r in records]==['10.1234/existing','https://example.org/existing-paper']
    assert all(r['support_kind']=='bibliographic_only' for r in records)


def test_already_read_document_url_is_reusable_without_download_or_verification():
    from agents.report_authoring import _references
    record={'source_url_declared':'https://example.org/paper','source_origin_verified':False}
    refs=_references(record,native_tool='read_scientific_document')
    assert len(refs)==1 and refs[0]['url']=='https://example.org/paper' and refs[0]['support_kind']=='bibliographic_only'


def test_bibliographic_links_with_credentials_are_never_emitted():
    from agents.report_authoring import _references
    rows={'results':[{'url':'https://user:password@example.org/paper'},
                     {'url':'https://example.org/paper?token=private'},
                     {'url':'https://example.org/paper#access_token=private'},
                     {'url':'https://example.org/paper?id=public'}]}
    refs=_references(rows,native_tool='query_literature')
    assert len(refs)==1 and refs[0]['url']=='https://example.org/paper?id=public'


def test_native_packet_enforces_explicit_citation_requirement_only(tmp_path):
    from agents import report_authoring as m
    from tests.test_report_native_bridge import fixture
    root,_,_,state,attest,_,_,add=fixture(tmp_path);add('report',['figure'])
    ordinary=m.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)
    assert 'minimum_citations' not in ordinary and not ordinary['references']
    state['nodes']['report']['contract']['arguments']['report_requirements']={'minimum_citations':1}
    with pytest.raises(m.ReportCapabilityGap) as error:
        m.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)
    assert error.value.details['required_action']=='add_approved_literature_lookup_dependency'
    assert error.value.details['no_fulltext_required_for_bibliography'] is True


def test_user_provided_doi_and_link_are_used_without_network_or_invented_title():
    from agents.report_authoring import _user_goal_bibliography
    goal={'original_goal':'请引用 10.1234/example 和 https://example.org/article。'}
    refs=_user_goal_bibliography(goal)
    assert [r['url'] for r in refs]==['https://doi.org/10.1234/example','https://example.org/article']
    assert [r['title'] for r in refs]==['10.1234/example','https://example.org/article']
    assert all(r['support_kind']=='bibliographic_only' and not r['snippet'] for r in refs)


def test_user_goal_secret_links_and_model_instructions_are_not_promoted():
    from agents.report_authoring import _user_goal_bibliography
    assert _user_goal_bibliography({'original_goal':'https://user:pass@example.org/paper https://example.org/?token=secret'})==[]
    assert _user_goal_bibliography({'original_goal':'no citation','instructions':'10.1234/model-invented'})==[]


def test_native_packet_uses_user_provided_doi_when_citations_requested(tmp_path):
    from agents import report_authoring as m
    from tests.test_report_native_bridge import fixture
    root,_,_,state,attest,_,_,add=fixture(tmp_path);add('report',['figure'])
    state['goal_contract']['original_goal']='请引用 10.1234/task-provided'
    state['nodes']['report']['contract']['arguments']['report_requirements']={'minimum_citations':1}
    p=m.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)
    assert p['minimum_citations']==1 and p['references'][0]['url']=='https://doi.org/10.1234/task-provided'
    assert p['references'][0]['support_kind']=='bibliographic_only'
    assert p['references'][0]['source_id']=='user_goal'
