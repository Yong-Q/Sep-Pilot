import copy
from agents.parallel_workflow import WorkflowStore
from agents.state_io import write_checkpoint


def contract(step,tool,args,deps=()):return {'step_id':step,'agent':'analyst','tool':tool,'arguments':args,'depends_on':list(deps),'expected_outputs':['out'], 'resources':[]}

def setup(tmp_path,changed=True):
 store=WorkflowStore(tmp_path/'workflow.json');wid=store.identity('u','c')
 prep=contract('prep','run_cdft',{'action':'inputs','temperature':298})
 calc=contract('calc','run_cdft',{'action':'submit','input_dir':'same/path'},['prep'])
 old={'username':'u','conv_id':'c','plan_version':1,'status':'active','goal_contract':{'version':1,'approved_plan_version':1},'nodes':{}}
 for n in (prep,calc):old['nodes'][n['step_id']]={'contract':copy.deepcopy(n),'status':'succeeded','phase':'finish','token':n['step_id']+'-token','job_ids':['123']if n['step_id']=='calc'else[],'result_ref':{'call_id':n['step_id']},'node_verification':{'ok':True}}
 write_checkpoint(store.path,{'workflows':{wid:old},'leases':{}})
 newprep=copy.deepcopy(prep)
 if changed:newprep['arguments']['temperature']=308
 independent=contract('independent','run_pore_analysis',{})
 return store,wid,old,[newprep,calc,independent],{'version':1,'approved_plan_version':2}

def test_changed_upstream_preserves_history_and_independent_work(tmp_path):
 store,wid,old,nodes,goal=setup(tmp_path)
 state=store.activate(wid,nodes,goal,'u','c',{'calc'})
 node=state['nodes']['calc'];assert node['status']=='validation_failed' and node['no_automatic_resubmission'] is True
 assert node['changed_upstream_steps']==['prep'] and node['attempt_history'][-1]['job_ids']==['123']
 assert store.claim(wid,'calc') is None and store.claim(wid,'independent') is not None
 assert state['status']=='active'

def test_unchanged_upstream_retains_completed_job(tmp_path):
 store,wid,old,nodes,goal=setup(tmp_path,False)
 state=store.activate(wid,nodes,goal,'u','c',{'prep','calc'})
 assert state['nodes']['calc']['status']=='succeeded' and state['nodes']['calc']['job_ids']==['123']
 assert state['nodes']['calc']['token']==old['nodes']['calc']['token']

def test_actor_change_cannot_bypass_changed_upstream(tmp_path):
 store,wid,old,nodes,goal=setup(tmp_path)
 nodes[1]['agent']='different-actor'
 state=store.activate(wid,nodes,goal,'u','c',{'calc'})
 assert state['nodes']['calc']['status']=='validation_failed'
 assert state['nodes']['calc']['attempt_history'][-1]['job_ids']==['123']
 assert store.claim(wid,'independent') is not None
