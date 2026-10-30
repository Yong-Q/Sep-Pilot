import copy

import pytest

from tests.test_api_review_regressions import bind_real_session


@pytest.mark.parametrize('change', [None, 'goal', 'policy'])
def test_resource_review_never_stamps_old_budget_with_new_authority(tmp_path, monkeypatch, change):
    import api
    conv,config=bind_real_session(tmp_path,monkeypatch)
    conv.session._checkpoint=lambda reason:None
    root=tmp_path/'runs'/conv.username/conv.conv_id
    root.mkdir(parents=True,exist_ok=True)
    policy={'excluded_nodes':['node06']}
    monkeypatch.setattr('agents.node_inventory.load_policy',lambda path:copy.deepcopy(policy))
    request={'kind':'before_submission','review_phase':'dispatch','nodes':[
        {'step_id':'calc','tool':'run_cdft','arguments':{'action':'submit','input_dir':str(root/'inputs')},
         'workload_facts':{'status':'ready','work_items':999},'executable_sha256':'fabricated'}],
        'resource_constraints':[{'source':'user','field':'cpus','value':1}],
        'resource_pins':{'calc':{'nodelist':['fabricated-node']}}}
    def review(observer, observed):
        assert observed['resource_constraints']==[] and observed['resource_pins']=={}
        assert observed['nodes'][0]['workload_facts']['status']!='ready'
        assert 'executable_sha256' not in observed['nodes'][0]
        if change=='goal':
            conv.session.goal_contract.version+=1
            conv.session.goal_contract.directives.append('cDFT只用1核')
        elif change=='policy':policy['excluded_nodes'].append('node25')
        return {'status':'ready','resource_review_id':'native-review','evidence_calls':[]}
    monkeypatch.setattr('agents.resource_review.run_resource_review',review)
    monkeypatch.setattr('agents.resource_review.allocate_reviewed_resources',lambda *args:{
        **args[-1],'resource_allocations':{'calc':{'budget_final':True,'cpus':8}}})
    receipt=api._review_resources(conv,request)
    if change:
        assert receipt['status']=='review_retry' and receipt['resource_allocations']=={}
    else:
        assert receipt['status']=='ready'
        assert receipt['resource_allocations']['calc']['goal_version']==conv.session.goal_contract.version


def test_busy_resource_reviewer_does_not_scan_native_inputs(tmp_path,monkeypatch):
    import api
    conv,config=bind_real_session(tmp_path,monkeypatch)
    class Busy:
        def __init__(self,*args):pass
        def acquire(self,**kwargs):return False
    monkeypatch.setattr('agents.state_io.ConversationLock',Busy)
    monkeypatch.setattr('agents.workload_context.enrich_resource_request',
        lambda *a,**kw:pytest.fail('busy coordination must not perform input hashing'))
    assert api._review_resources(conv,{'kind':'before_submission'})['status']=='review_busy'
