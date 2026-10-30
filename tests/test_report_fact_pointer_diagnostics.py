"""Exact source binding stays strict; bounded hints contain keys, never values."""
import copy
import json
import pytest
from agents.report_authoring import _fact_block,_digest,author_stage_schema,author_report
from agents.report_fact_diagnostics import scalar_pointer_examples,missing_fact_pointer_diagnostic


def packet(tmp_path,projected=False):
    root=tmp_path/'runs/owner/conv';root.mkdir(parents=True)
    content={'native_info':{'converged':False,'iterations':20},'a/b~':{'value':7},
             'units':'native units','secret_label':'THIS_VALUE_MUST_NOT_APPEAR_IN_HINTS'}
    doc={'id':'J1','path':'source.json','sha256':'a'*64,'untrusted_content':content}
    if projected:doc.update(json_root_pointer='/public',metadata_schema='native_report_source_metadata_v1',role='provenance')
    p={'root':str(root),'owner':{'username':'owner','conv_id':'conv'},'title':'Source limits',
       'report_output_path':'report.md','sources':[{'id':'S1','step_id':'native','call_id':'producer',
       'source_sha256':'b'*64,'structured_data':[doc]}],'tables':[],'references':[],'figures':[],
       'file_facts':[]}
    p['packet_sha256']=_digest(p)
    return p


def fact(pointer):return {'kind':'fact','source_id':'S1','document_id':'J1','pointer':pointer}


@pytest.mark.parametrize('projected',[False,True])
def test_hints_do_not_resolve_missing_field_or_change_native_binding(tmp_path,projected):
    p=packet(tmp_path,projected);before=copy.deepcopy(p)
    with pytest.raises(ValueError,match='does not identify a source value') as error:
        _fact_block(p,fact('/scientific_convergence'))
    hint=str(error.value)
    assert 'available_keys' in hint and 'native_info' in hint
    assert 'THIS_VALUE' not in hint and 'native units' not in hint
    text,binding,source=_fact_block(p,fact('/native_info/converged'))
    assert binding['value'] is False
    assert binding['pointer']==('/public' if projected else '')+'/native_info/converged'
    assert binding['source_sha256']=='a'*64 and source=='S1'
    assert p==before


@pytest.mark.parametrize('pointer',['/public/native_info/converged','/binding/username','/owner/username',
    '/private/job_id','/producer_sha256','/native_info/Converged','/native_info/scientific_convergence'])
def test_no_root_alias_private_envelope_or_spelling_fallback(tmp_path,pointer):
    with pytest.raises(ValueError,match='does not identify a source value'):
        _fact_block(packet(tmp_path,True),fact(pointer))


@pytest.mark.parametrize('pointer',['/a~2b','/native_info/iterations/0','/native_info','/native_info/iterations/'])
def test_existing_pointer_escape_type_and_scalar_guards_remain(tmp_path,pointer):
    with pytest.raises(ValueError):_fact_block(packet(tmp_path),fact(pointer))


def test_escaped_pointer_value_and_original_sha_are_unchanged(tmp_path):
    p=packet(tmp_path,True);_,binding,_=_fact_block(p,fact('/a~1b~0/value'))
    assert binding['pointer']=='/public/a~1b~0/value' and binding['value']==7


def test_unknown_projection_schema_is_rejected_before_key_diagnostics(tmp_path):
    p=packet(tmp_path,True);p['sources'][0]['structured_data'][0]['metadata_schema']='untrusted-schema'
    with pytest.raises(ValueError,match='unknown native fact projection root'):
        _fact_block(p,fact('/missing'))
    with pytest.raises(ValueError,match='unknown native fact projection root'):
        author_stage_schema(p)


def test_schema_exposes_bounded_relative_pointer_names_only(tmp_path):
    p=packet(tmp_path,True);schema=author_stage_schema(p)
    variant=next(v for v in schema['properties']['draft']['properties']['blocks']['items']['oneOf'] if v['properties']['kind']['const']=='fact')
    description=variant['properties']['pointer']['description']
    assert '/native_info/converged' in description and 'Do not prepend json_root_pointer' in description
    assert 'THIS_VALUE_MUST_NOT_APPEAR' not in description and 'native units' not in description
    assert variant['properties']['pointer']['type']=='string'
    assert 'enum' not in variant['properties']['pointer']


def test_large_or_nonfinite_values_do_not_create_unusable_pointer_examples():
    d={'untrusted_content':{**{str(i):'VALUE-'+str(i)for i in range(2000)},'bad':float('nan')}}
    hints=scalar_pointer_examples(d)
    assert len(hints)<=48 and all('VALUE' not in p for p in hints)
    text=missing_fact_pointer_diagnostic(d,d['untrusted_content'])
    assert len(json.loads(text)['available_keys'])<=12 and 'VALUE' not in text
    assert scalar_pointer_examples({'untrusted_content':{'oversized':'x'*2001,'bad':float('inf')}})==[]


def test_bounded_feedback_reaches_next_author_without_auto_rewriting_missing_value(tmp_path):
    p=packet(tmp_path);calls=[]
    narrative={'kind':'narrative','text':'The source verification status has a limited scope.','source_ids':['S1'],'claim_type':'limitation'}
    def model(phase,payload):
        calls.append(phase)
        if phase=='author':
            if not payload['previous_issues']:block=fact('/scientific_convergence')
            else:
                assert 'available_keys' in payload['previous_issues'][0]
                assert 'THIS_VALUE' not in payload['previous_issues'][0]
                block=fact('/native_info/converged')
            return {'actor':'communicator','call_id':'author-'+str(len(calls)),
                    'draft':{'blocks':[copy.deepcopy(narrative),block]}}
        return {'actor':'report-reviewer','call_id':'review','review':{'passed':True,
            'draft_sha256':payload['draft_sha256'],'checked_blocks':[0,1],'blocking_issues':[]}}
    result=author_report(p,model_call=model)
    assert calls==['author','author','review'] and result['verified'] is True
    assert r'native\_info/converged = false' in result['markdown']
    assert 'scientific_convergence =' not in result['markdown']
