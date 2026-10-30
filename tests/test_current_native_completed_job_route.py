"""Current same-job repair uses its native authority, not archived restoration."""
import copy
import json
from pathlib import Path
import pytest
from agents.state_io import json_transaction
from agents.workflow_view import bind_result_handoff
from tests.test_parallel_workflow import factory
from tests.test_cdft_partial_recovery import partial_cdft
from tests.test_cdft_workflow_result_inspection import complete_inspected_cdft,recorded_cdft_inspection


@pytest.mark.parametrize('status',['prefinish','validation_failed'])
def test_public_current_cdft_job_id_revalidates_then_inspects_and_finishes_without_resubmit(partial_cdft,status):
 rt,work,calls,_=complete_inspected_cdft(partial_cdft);before=rt.snapshot()['nodes']['cdft']
 rt.store.update(rt.workflow_id,'cdft',before['token'],{'status':status,'jobs_confirmed_terminal':True})
 bind_result_handoff(rt.main,lambda:rt)
 result=rt.main._on_workflow_revalidate('cdft',[str(work/'results.csv')],completed_job_id='42')
 assert result['ok'] and result['no_resubmission']
 current=rt.snapshot()['nodes']['cdft'];assert current['status']=='prefinish'
 assert current['token']==before['token'] and current['attempt_id']==before['attempt_id']
 assert current['job_ids']==['42'] and current['contract']['arguments']==before['contract']['arguments']
 call=recorded_cdft_inspection(rt)
 rt.main._on_workflow_finish('cdft',[call['call_id']],'All native material outputs are complete; solver convergence is not established.')
 saved=rt.snapshot()['nodes']['cdft'];assert saved['status']=='succeeded' and saved['node_verification']['all_materials_valid'] is True
 assert len(calls)==1


@pytest.mark.parametrize('change',['different_id','running','unconfirmed','foreign_owner','foreign_attempt','foreign_key','failed_job','nonterminal_watch','foreign_tool','watch_id','missing_watch'])
def test_current_job_hint_cannot_bypass_stopped_history_or_native_job_guards(partial_cdft,change):
 rt,work,calls,_=complete_inspected_cdft(partial_cdft);before=rt.snapshot()['nodes']['cdft'];job='42'
 if change=='different_id':job='777'
 elif change in ['running','unconfirmed']:
  update={'status':'running'} if change=='running' else {'jobs_confirmed_terminal':False}
  rt.store.update(rt.workflow_id,'cdft',before['token'],update)
 else:
  original=rt.job_watch.get
  values={'foreign_owner':{'username':'foreign'},'foreign_attempt':{'attempt_id':'foreign'},'foreign_key':{'recovery_key':'foreign'},'failed_job':{'failed':True},'nonterminal_watch':{'terminal':False,'state':'RUNNING'},'foreign_tool':{'tool':'run_gcmc_batch'},'watch_id':{'job_id':'777'}}
  rt.job_watch.get=lambda jid: {} if change=='missing_watch' else {**original(jid),**values[change]}
 before_state=rt.store.path.read_bytes();before_ledger=(rt.root/'recovery_state.json').read_bytes()
 with pytest.raises((ValueError,PermissionError)):
  rt.revalidate_outputs('cdft',[str(work/'results.csv')],completed_job_id=job)
 assert rt.store.path.read_bytes()==before_state and (rt.root/'recovery_state.json').read_bytes()==before_ledger
 assert len(calls)==1
