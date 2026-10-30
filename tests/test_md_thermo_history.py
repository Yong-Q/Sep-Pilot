"""Native byte/line/block contracts using synthetic fixtures only."""
import copy,hashlib,json
from pathlib import Path
import pytest
from agents.md_result_facts import parse_md_thermo,verify_md_thermo,md_thermo_summary,parse_md_result_facts


def stats(stop='energy tolerance',iterations=5,energy='3'):
    return f'''Minimization stats:
 Stopping criterion = {stop}
 Energy initial, next-to-last, final =
 10 4 {energy}
 Force two-norm initial, final = 2 1
 Force max component initial, final = 3 0.5
 Final line search alpha, max atom move = 0.1 0.2
 Iterations, force evaluations = {iterations} {iterations*2}
MPI task timing breakdown:
'''


def block(kind='run',rows='0 300\n10 290\n',steps=10,headers='Step Temp',unit='real',suffix='',complete=True):
    setup='cg style minimization'if kind=='minimization'else'Verlet run'
    text=f'Setting up {setup} ...\n'
    if unit is not None:text+=f' Unit style : {unit}\n'
    text+=' Current step : 0\n Time step : 0.5\n'+headers+'\n'+rows
    if complete:text+=f'Loop time of 0.01 on 4 procs for {steps} steps with 2 atoms\n'
    if kind=='minimization':text+=stats(iterations=steps)
    return text+suffix


def parse(text):return parse_md_thermo(('LAMMPS engineering fixture\n'+text).encode())


def test_distinct_execution_blocks_preserve_reset_step_order_and_loop_counts():
    raw=('LAMMPS\n'+block(rows='0 300\n5 290\n',steps=17)+block(rows='0 280\n5 270\n',steps=23)).encode()
    data=parse_md_thermo(raw)
    assert data['ok']and [b['native_steps']for b in data['blocks']]==[17,23]
    assert len(data['blocks'])==2 and data['blocks'][0]['tables'][0]['rows'][-1]['values'][0]==5
    assert data['replicate_n']is None and all(b['continuity_to_other_blocks']=='not_established'for b in data['blocks'])
    assert verify_md_thermo(raw,data)


def test_every_minimization_reuses_legacy_facts_without_last_block_overwriting_history():
    text=''.join(block('minimization',steps=n,headers='Step Temp TotEng',rows=f'0 0 10\n{n} 0 3\n')for n in(3,7,2))
    data=parse(text)
    assert data['ok']and [b['native_minimization']['iterations']for b in data['blocks']]==[3,7,2]
    assert [b['native_steps']for b in data['blocks']]==[3,7,2]
    assert all(b['native_minimization']['native_tolerance_stop']is True for b in data['blocks'])
    assert parse_md_result_facts(text)['final_minimization']['iterations']==2
    for b in data['blocks']:
        line=b['stats_field_anchors']['iterations'][0]['line']
        assert data['line_count']>=line>=b['line_start']


def test_run_stages_and_unmarked_zero_step_snapshot_are_distinct():
    text=''
    for stage,n in(('warmup',0),('equilibration',0),('production',20)):
        text+=f'SEPPILOT_STAGE {stage} begin\n'+block(steps=n)+f'SEPPILOT_STAGE {stage} end\n'
    text+=block(steps=0,headers='Temp Press PotEng KinEng',rows='150 22 871 189\n')
    data=parse(text);assert data['ok']
    assert [b['stage']for b in data['blocks']]==['warmup','equilibration','production',None]
    assert [b['native_steps']for b in data['blocks']]==[0,0,20,0]
    assert data['blocks'][-1]['tables'][0]['headers']==['Temp','Press','PotEng','KinEng']
    assert data['convergence']=='unknown'


def test_multiple_tables_in_one_execution_never_double_count_native_steps():
    data=parse(block(rows='0 300\nStep Press\n0 15\n10 16\n',steps=37))
    b=data['blocks'][0];assert b['native_steps']==37 and len(b['tables'])==2
    assert b['tables'][0]['headers']==['Step','Temp']and b['tables'][1]['headers']==['Step','Press']


def test_repeated_stage_markers_keep_distinct_sequence_ids():
    one='SEPPILOT_STAGE production begin\n'+block(steps=5)+'SEPPILOT_STAGE production end\n'
    data=parse(one+one);assert data['ok']
    assert [b['stage_marker_sequence']for b in data['blocks']]==[1,2]
    assert len(data['stage_markers'])==2

@pytest.mark.parametrize('ending',['missing_loop','unclosed_stage','unmatched_end','nested_stage','bad_width','duplicate_headers','error_line','nonfinite','invalid_cell'])
def test_truncation_warning_and_bad_data_are_explicit_never_backfilled(ending):
    text=block()
    if ending=='missing_loop':text=block(complete=False)
    elif ending=='unclosed_stage':text='SEPPILOT_STAGE production begin\n'+text
    elif ending=='unmatched_end':text+='SEPPILOT_STAGE other end\n'
    elif ending=='nested_stage':text='SEPPILOT_STAGE warmup begin\nSEPPILOT_STAGE production begin\n'+text
    elif ending=='bad_width':text=block(rows='0 300\n10 290 5\n')
    elif ending=='duplicate_headers':text=block(headers='Step Step')
    elif ending=='error_line':text=block(rows='0 300\nERROR: source failure\n10 290\n')
    elif ending=='nonfinite':text=block(rows='0 300\n10 nan\n')
    else:text=block(rows='0 300\n10 bad\n')
    data=parse(text);assert data['ok']is False and data['parse_status']=='partial'
    if ending=='missing_loop':assert data['blocks'][0]['native_steps']is None
    if ending in{'nonfinite','invalid_cell'}:assert data['blocks'][0]['tables'][0]['rows'][-1]['values'][1]is None
    if ending=='bad_width':assert data['blocks'][0]['tables'][0]['rejected_rows'][0]['source']['text']=='10 290 5'
    json.dumps(data,allow_nan=False)


def test_warnings_do_not_become_numeric_rows_or_convergence_failures():
    data=parse(block(rows='0 300\nWARNING: Inconsistent image flags\n10 290\n'))
    b=data['blocks'][0];assert data['ok']and len(b['tables'][0]['rows'])==2
    assert b['messages'][0]['kind']=='warning'and b['convergence']=='unknown'


def test_native_line_search_stop_is_not_rewritten_as_tolerance_or_job_failure():
    text=block('minimization',headers='Step Temp TotEng',rows='0 0 10\n10 0 3\n').replace('energy tolerance','linesearch alpha is zero')
    data=parse(text);assert data['ok']and data['blocks'][0]['native_minimization']['native_tolerance_stop']is False
    assert data['scientific_acceptance']is False


def test_same_table_step_reset_is_flagged_without_synthesizing_a_trajectory():
    data=parse(block(rows='10 300\n0 290\n',steps=50));t=data['blocks'][0]['tables'][0]
    assert t['step_monotonic']is False and 'do not infer'in t['continuity_warning']
    assert data['blocks'][0]['native_steps']==50

@pytest.mark.parametrize('unit,energy,temp,press',[('real','kcal/mol','K','atm'),('metal','eV','K','bar'),('si','J','K','Pa'),('lj','reduced','reduced','reduced'),(None,None,None,None),('unrecognized',None,None,None)])
def test_native_units_are_declared_or_unknown_not_inferred(unit,energy,temp,press):
    data=parse(block(headers='Step Temp Press TotEng',rows='0 300 15 10\n',unit=unit));t=data['blocks'][0]['tables'][0]
    assert t['column_units']=={'Step':'count','Temp':temp,'Press':press,'TotEng':energy}
    assert 'normalization'in data and 'unknown'in data['normalization']

@pytest.mark.parametrize('tamper',['value','line','sha','kind','headers','source'])
def test_original_source_and_every_native_fact_are_tamper_evident(tamper):
    raw=('LAMMPS\n'+block()).encode();data=parse_md_thermo(raw);bad=copy.deepcopy(data)
    if tamper=='value':bad['blocks'][0]['tables'][0]['rows'][0]['values'][1]=0
    elif tamper=='line':bad['blocks'][0]['tables'][0]['rows'][0]['source']['line']+=1
    elif tamper=='sha':bad['blocks'][0]['tables'][0]['rows'][0]['source']['line_sha256']='0'*64
    elif tamper=='kind':bad['blocks'][0]['kind']='minimization'
    elif tamper=='headers':bad['blocks'][0]['tables'][0]['headers'][1]='Press'
    else:raw+=b'\nchanged'
    with pytest.raises(ValueError):verify_md_thermo(raw,bad)


def test_crlf_and_nonascii_warning_preserve_original_byte_SHA_and_line_anchors():
    raw=('LAMMPS\n'+block(rows='0 300\nWARNING: 测试警告\n10 290\n')).replace('\n','\r\n').encode()
    data=parse_md_thermo(raw);assert data['source_sha256']==hashlib.sha256(raw).hexdigest()
    for row in data['blocks'][0]['tables'][0]['rows']:
        assert row['source']['line_sha256']==hashlib.sha256(raw.splitlines()[row['source']['line']-1]).hexdigest()
    assert verify_md_thermo(raw,data)

@pytest.mark.parametrize('bound',['bytes','blocks','rows','unicode_separator'])
def test_bounds_and_unrecognized_line_separators_fail_closed(bound):
    raw=('LAMMPS\n'+block()+block()).encode();kw={}
    if bound=='bytes':kw['max_bytes']=10
    elif bound=='blocks':kw['max_blocks']=1
    elif bound=='rows':kw['max_rows']=1
    else:raw=raw.replace(b'LAMMPS',b'LAMMPS\xe2\x80\xa8unexpected')
    with pytest.raises(ValueError):parse_md_thermo(raw,**kw)


def test_native_summary_keeps_block_identity_and_endpoints_without_row_count_as_n():
    data=parse(block());summary=md_thermo_summary(data)
    assert len(summary['blocks'][0]['tables'][0]['rows'])==2
    assert summary['source_sha256']==data['source_sha256']and summary['scientific_acceptance']is False




def test_preview_and_complete_native_json_use_one_row_contract():
    raw=('LAMMPS\n'+block(rows=''.join(str(i)+' '+str(300-i)+'\n'for i in range(100)),steps=99)).encode()
    history=parse_md_thermo(raw);summary=md_thermo_summary(history)
    full=history['blocks'][0]['tables'][0];preview=summary['blocks'][0]['tables'][0]
    assert set(preview)<=set(full)
    assert full['row_count']==preview['row_count']==100
    assert full['rows_complete']is True and full['rows_preview']is False
    assert preview['rows_complete']is False and preview['rows_preview']is True
    assert len(full['rows'])==100 and len(preview['rows'])==2
    for obj in(full,preview):
        assert obj['headers'].index('Temp')==1 and obj['rows'][0]['values'][1]==300
    assert 'endpoint_rows'not in full and 'endpoint_rows'not in preview
