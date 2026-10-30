"""Original-node identity stays bounded by native scope, attempt and owner."""
import copy

import pytest

from agents.goal_contract import GoalContract, scope_preserving_continuation
from agents.recovery import RecoveryGate, input_fingerprint
from agents.state_io import json_transaction
from agents.watch_context import set_context, clear_context
from agents.workflow_execution_identity import bound_retry_identity, metadata_only_goal_revision
from tests.test_parallel_workflow import factory, node


@pytest.fixture
def stopped(factory,tmp_path):
    args={'material_type':'MOF','n_structures':1,'output_dir':str(tmp_path/'structures')}
    contract=node('generate',tool='generate_structure',args=args,outputs=[{'kind':'directory','path':args['output_dir'],'pattern':'*.cif','min_count':1}])
    contract['agent']='harness'
    rt=factory([contract]);rt.start(1)
    actor=rt.main;gate=RecoveryGate(path=rt.root/'recovery_state.json');actor.recovery_gate=gate
    actor._interrupt_requested=lambda:False
    ticket=rt.store.claim(rt.workflow_id,'generate')
    key=gate.key('generate_structure',args,actor.goal_contract.version,'generate',tmp_path)
    attempt,reason=gate.claim(key,'generate_structure',args,input_fingerprint(args,tmp_path),
        goal_version=actor.goal_contract.version,project_root=tmp_path)
    assert attempt and not reason
    gate.outcome(key,attempt,{'failed':True,'error':'controlled installation failure'})
    rt.store.update(rt.workflow_id,'generate',ticket['token'],{
        'status':'failed','recovery_key':key,'attempt_id':attempt,'tool_returned':True,
        'effective_arguments':copy.deepcopy(args)},release=True)
    # Retain an old-parser continuation revision, never mutate actual user state.
    actor.goal_contract.version+=1
    actor.goal_contract.active_goal='继续原任务，保持原先的参数。'
    actor.goal_contract.directives.append(actor.goal_contract.active_goal)
    set_context(rt.username,rt.conv_id,'lead-orchestrator',rt.conv_id)
    yield rt,actor,gate,args,key
    clear_context()


def test_stopped_metadata_reconciliation_changes_no_node_or_ledger(stopped):
    rt,actor,gate,args,key=stopped
    before=copy.deepcopy(rt.snapshot()['nodes']);ledger=gate.path.read_bytes()
    assert bound_retry_identity(actor,'generate_structure',args,'generate') is None
    assert rt._reconcile_native_goal_metadata(rt.snapshot())
    assert rt.snapshot()['nodes']==before and gate.path.read_bytes()==ledger
    identity=bound_retry_identity(actor,'generate_structure',args,'generate')
    assert identity['key']==key and identity['identity_goal_version']==1 and identity['goal_version']==2
    assert not gate.snapshot()[key].get('approved_fingerprint')


@pytest.mark.parametrize('change',['owner','goal_science','node_science','attempt','token','project_root','origin','history','pause'])
def test_stable_retry_rejects_changed_scientific_identity(stopped,change):
    rt,actor,gate,args,key=stopped
    assert rt._reconcile_native_goal_metadata(rt.snapshot())
    if change=='owner':set_context('other',rt.conv_id,'lead-orchestrator',rt.conv_id)
    elif change=='goal_science':actor.goal_contract.parameters['temperature_K']=308
    else:
        with json_transaction(rt.store.path) as data:
            state=data['workflows'][rt.workflow_id];current=state['nodes']['generate']
            if change=='node_science':current['contract']['arguments']['n_structures']=2
            elif change=='attempt':current['attempt_id']='new-attempt'
            elif change=='token':current['token']=None
            elif change=='origin':current['origin_goal_version']=2
            elif change=='history':state['goal_reconciliation_history']=[]
            elif change=='pause':state['user_paused']=True
        if change=='project_root':
            with gate.transaction() as data:data[key]['project_root']='/foreign/project'
    ledger=gate.path.read_bytes()
    assert bound_retry_identity(actor,'generate_structure',args,'generate') is None
    assert gate.path.read_bytes()==ledger


@pytest.mark.parametrize('text',[
    '继续原任务，但力场改为UFF。',
    '继续原任务，保持温度，换成另一个MOF。',
    '继续原任务，保持参数，改为输入 /other/material.cif。',
])
def test_unknown_prose_scope_change_cannot_reconcile_as_metadata(stopped,text):
    rt,actor,gate,args,key=stopped
    actor.goal_contract.active_goal=text;actor.goal_contract.directives[-1]=text
    before=rt.snapshot();ledger=gate.path.read_bytes()
    assert not metadata_only_goal_revision(before['goal_contract'],actor.goal_contract.to_dict())
    assert not rt._reconcile_native_goal_metadata(before)
    assert rt.snapshot()==before and gate.path.read_bytes()==ledger


def test_new_filename_is_not_a_substring_of_existing_path():
    approved=[{'arguments':{'cif_path':'/project/beta.cif'}}]
    assert not scope_preserving_continuation('继续原任务，保持参数，使用 a.cif。',approved)
