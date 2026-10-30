from types import SimpleNamespace
import pytest
from agents.workflow_metadata_alignment import prove_forward_transition,session_transition_proof

class Goal:
 def __init__(self,version=0):self.version=version
 def to_dict(self):return {'approved_plan_version':self.version}

def runtime(state,version=0):
 return SimpleNamespace(main=SimpleNamespace(goal_contract=Goal(version)),username='u',conv_id='c',snapshot=lambda:state)

def test_initial_dag_can_be_proposed_without_fake_prior_runtime():
 rt=runtime({});nodes=[{'step_id':'read','tool':'read_file'}]
 assert prove_forward_transition(rt,nodes,1,['read'])=={'no_dispatch':True,'covered_terminal_nodes':[],'proofs':{}}
 session=SimpleNamespace(_runtime_snapshot=rt.snapshot,_on_workflow_metadata_transition=lambda nodes,version,affected:prove_forward_transition(rt,nodes,version,affected))
 assert session_transition_proof(session,nodes,['read'],1)['no_dispatch'] is True

@pytest.mark.parametrize('state,version,plan',[
 ({},1,2),({'username':'other','conv_id':'c','nodes':{}},0,1),
 ({'username':'u','conv_id':'different','nodes':{}},0,1),({},0,0),
])
def test_foreign_or_noninitial_state_cannot_claim_empty_initial_proof(state,version,plan):
 with pytest.raises(ValueError):prove_forward_transition(runtime(state,version),[{'step_id':'read'}],plan,['read'])
