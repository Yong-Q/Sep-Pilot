from agents.parallel_workflow import dependency_runtime_arguments, _dependency_binding_updates

def test_container_input_keeps_approved_hierarchy_even_for_one_successful_case(tmp_path):
 root=tmp_path/'gcmc';case=root/'CO2'/'298K'/'material';case.mkdir(parents=True)
 node={'contract':{'tool':'analyze_gcmc_screening','arguments':{'work_dir':str(root),'gases':['CO2']},'depends_on':['producer']}}
 states={'producer':{'status':'succeeded','result':{'work_dir':str(case)}}}
 args,bindings=dependency_runtime_arguments(node,states,tmp_path)
 assert args['work_dir']==str(root) and bindings==[]

def test_real_timestamp_directory_binding_still_works(tmp_path):
 root=tmp_path/'cdft';case=root/'data'/'timestamp';case.mkdir(parents=True)
 node={'contract':{'tool':'run_cdft','arguments':{'action':'collect','job_work_dir':str(root)},'depends_on':['producer']}}
 states={'producer':{'status':'succeeded','result':{'work_dir':str(case)}}}
 args,bindings=dependency_runtime_arguments(node,states,tmp_path)
 assert args['job_work_dir']==str(case) and bindings[0]['source_step']=='producer'

def test_stale_incompatible_binding_is_cleared_for_stopped_node(tmp_path):
 root=tmp_path/'gcmc';case=root/'CO2'/'298K'/'material';case.mkdir(parents=True)
 nodes={'producer':{'status':'succeeded','result':{'work_dir':str(case)}},
  'collect':{'status':'failed','contract':{'tool':'analyze_gcmc_screening','arguments':{'work_dir':str(root)},'depends_on':['producer']},
   'resolved_arguments':{'work_dir':str(case)},'runtime_input_bindings':[{'resolved_path':str(case)}]}}
 updates=_dependency_binding_updates(nodes,tmp_path)
 assert ('collect',None,None,None) in updates

def test_other_input_uses_independent_contract_and_preserves_ownership_bounds(tmp_path):
 root=tmp_path/'models';case=root/'actual';case.mkdir(parents=True)
 node={'contract':{'tool':'ml_predict','arguments':{'model_dir':str(root)},'depends_on':['train']}}
 states={'train':{'status':'succeeded','result':{'model_dir':str(case)}}}
 assert dependency_runtime_arguments(node,states,tmp_path)[0]['model_dir']==str(case)
 outside=tmp_path/'foreign';outside.mkdir();states['train']['result']['model_dir']=str(outside)
 assert dependency_runtime_arguments(node,states,tmp_path)[0]['model_dir']==str(root)
