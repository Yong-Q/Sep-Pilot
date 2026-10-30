import copy
from types import SimpleNamespace
import pytest
from agents import job_control


def assessment():
    def node(name, states=None):
        return {'name':name,'states':states or ['idle'],'partitions':['compute'],
                'cpus':{'idle':4,'total':8},'memory':{'verified':True,'available_for_scheduling_mib':65536},'glibc':'2.39'}
    return {'effective_constraints':{'JobState':'PENDING','NumCPUs':'1','ReqTRES':'cpu=1,mem=16G,node=1','TimeLimit':'3-00:00:00','ReqNodeList':'busy','Partition':'compute'},
        'partition_limits':{'compute':{'verified':True,'state':'UP','max_time':'5-08:00:00'}},
        'node_inventory':{'verified':True,'stale':False,'observed_at':100,'policy':{'excluded_nodes':[],'preferred_nodes':[], 'allowed_states':['idle','mix'],'tool_profiles':{}},'nodes':{'planned':node('planned',['mix-']),'good':node('good')}}}


def test_planned_target_is_retryable_with_budget_preserving_alternative():
    feedback=job_control.retarget_compatibility(assessment(),{'tool':'run_cdft'},'planned','compute')
    assert feedback['retryable'] and feedback['requires_user'] is False
    assert 'state_not_allowed' in feedback['reason_codes']
    assert [(x['nodelist'],x['memory_mb'],x['cpus'],x['walltime']) for x in feedback['compatible_targets']]==[('good',16384,1,'3-00:00:00')]


@pytest.mark.parametrize('change', ['node_pin','partition_pin','excluded','gpu','glibc','time','unknown_partition','cpu','ram'])
def test_alternatives_obey_hard_constraints(change):
    a=assessment();p=a['node_inventory']['policy'];good=a['node_inventory']['nodes']['good']
    profile=p['tool_profiles'].setdefault('run_cdft',{})
    if change=='node_pin':profile['allowed_nodes']=['other']
    if change=='partition_pin':profile['allowed_partitions']=['other']
    if change=='excluded':p['excluded_nodes']=['good']
    if change=='gpu':profile['gpus']=1
    if change=='glibc':profile['min_glibc']='2.40'
    if change=='time':a['partition_limits']['compute']['max_time']='2-00:00:00'
    if change=='unknown_partition':a['partition_limits']={}
    if change=='cpu':good['cpus']['idle']=0
    if change=='ram':good['memory']['available_for_scheduling_mib']=1024
    before=copy.deepcopy(a)
    result=job_control.retarget_compatibility(a,{'tool':'run_cdft'},'good','compute')
    assert not result['target_compatible'] and not result['compatible_targets']
    assert result['reason_codes'] and a==before


def test_same_explicit_memory_is_not_a_profile_change(tmp_path,monkeypatch):
    from agents.job_watch import JobWatch
    watch=JobWatch(str(tmp_path/'watch.json'));watch.register('123',username='a',conv_id='c',tool='run_cdft')
    monkeypatch.setattr('agents.job_watch.get_watch',lambda:watch)
    a=assessment();commands=[]
    monkeypatch.setattr('agents.resource_review.assess_job_resources',lambda _:copy.deepcopy(a))
    def run(argv,**kw):
        commands.append(argv);a['effective_constraints'].update(ReqNodeList='good',Partition='compute')
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(job_control.subprocess,'run',run)
    result=job_control.retarget_queued_job('123','good','compute','a','c',tmp_path,memory_mb=16384)
    assert result['ok'] and not result['memory_changed'] and len(commands)==1
    result=job_control.retarget_queued_job('123','good','compute','a','c',tmp_path,memory_mb=8192)
    assert result['blocked'] and not result['target_compatible'] and len(commands)==1


@pytest.mark.parametrize('field,value', [('stale',True),('verified',False)])
def test_unverified_inventory_never_supplies_alternatives(field,value):
    a=assessment();a['node_inventory'][field]=value
    result=job_control.retarget_compatibility(a,{'tool':'run_cdft'},'good','compute')
    assert not result['compatible_targets'] and not result['target_compatible']


def test_native_gpu_demand_and_current_gpu_allocation_cannot_disappear():
    a=assessment()
    assert not job_control.retarget_compatibility(a,{'tool':'run_string_tst'},'good','compute')['target_compatible']
    a['effective_constraints']['ReqTRES']+=',gres/gpu=1'
    assert not job_control.retarget_compatibility(a,{'tool':'run_cdft'},'good','compute')['target_compatible']
    a['node_inventory']['nodes']['good']['gpus']={'verified':True,'available':1,'total':1}
    assert job_control.retarget_compatibility(a,{'tool':'run_cdft'},'good','compute')['target_compatible']


def test_assessment_collects_partition_limits_without_updating_scheduler(tmp_path,monkeypatch):
    from agents.resource_review import assess_job_resources
    a=assessment();calls=[]
    monkeypatch.setattr('agents.job_watch.get_watch',lambda:SimpleNamespace(get=lambda _: {'tool':'run_cdft','username':'a','conv_id':'c','work_dir':str(tmp_path)}))
    monkeypatch.setattr('agents.watch_context.get_context',lambda:{'username':'a','conv_id':'c'})
    monkeypatch.setattr('agents.config.get_config',lambda:SimpleNamespace(project_root=tmp_path))
    monkeypatch.setattr('agents.node_inventory.node_inventory',lambda *a_,**kw:a['node_inventory'])
    monkeypatch.setattr('agents.slurm.check_job_status',lambda *a,**kw:{'state':'PENDING'})
    def run(argv,**kw):
        calls.append(argv)
        if argv[2]=='job':
            output=' '.join(f'{k}={v}' for k,v in a['effective_constraints'].items())
        else: output='PartitionName=compute State=UP MaxTime=5-08:00:00'
        return SimpleNamespace(returncode=0,stdout=output,stderr='')
    monkeypatch.setattr('agents.resource_review.subprocess.run',run)
    result=assess_job_resources('123')
    assert result['read_only'] and result['compatible_targets'][0]['nodelist']=='good'
    assert result['partition_limits']['compute']['max_time']=='5-08:00:00'
    assert calls==[['scontrol','show','job','123','-o'],['scontrol','show','partition','-o']]


def test_changed_memory_alternatives_require_each_nodes_operator_profile():
    a=assessment();p=a['node_inventory']['policy']['tool_profiles']
    p['run_cdft']={'memory_fraction_of_available':0.5}
    a['node_inventory']['nodes']['different']=copy.deepcopy(a['node_inventory']['nodes']['good'])
    a['node_inventory']['nodes']['different'].update(name='different')
    a['node_inventory']['nodes']['different']['memory']['available_for_scheduling_mib']=131072
    result=job_control.retarget_compatibility(a,{'tool':'run_cdft'},'good','compute',32768)
    assert result['target_compatible']
    assert [x['nodelist'] for x in result['compatible_targets']]==['good']


def test_rejected_target_never_mutates_job_and_returns_fresh_alternative(tmp_path,monkeypatch):
    from agents.job_watch import JobWatch
    watch=JobWatch(str(tmp_path/'watch.json'));watch.register('123',username='a',conv_id='c',tool='run_cdft')
    monkeypatch.setattr('agents.job_watch.get_watch',lambda:watch)
    monkeypatch.setattr('agents.resource_review.assess_job_resources',lambda _:assessment())
    monkeypatch.setattr(job_control.subprocess,'run',lambda *a,**k:pytest.fail('rejected target must not update scheduler'))
    result=job_control.retarget_queued_job('123','planned','compute','a','c',tmp_path,memory_mb=16384)
    assert result['status']=='infrastructure_retry' and result['retryable']
    assert result['requires_user'] is False and result['job_id']=='123'
    assert result['compatible_targets'][0]['nodelist']=='good'
    assert not list((tmp_path/'runs/a/c/job_control').glob('123_*.json'))


def test_retarget_cannot_add_missing_native_gpu_reservation_by_node_selection():
    a=assessment()
    a['node_inventory']['nodes']['good']['gpus']={'verified':True,'available':4,'total':4}
    result=job_control.retarget_compatibility(a,{'tool':'run_string_tst'},'good','compute')
    assert not result['target_compatible'] and not result['compatible_targets']
    assert 'existing_gpu_request_below_native_or_operator_requirement' in result['reason_codes']
