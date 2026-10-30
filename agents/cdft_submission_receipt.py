"""Native cDFT child receipt classification; no submission or retry authority.

A JSON dictionary cannot attest scheduler acceptance. Only the host-created
child result carries the final native adapter receipt across this local boundary.
"""
import copy
import re


class NativeChildResult(dict):
    """Private provenance is an attribute, never accepted from JSON fields."""
    def __init__(self, result, native_receipt=None, *, transport_uncertain=False):
        super().__init__(result)
        self.native_receipt = copy.deepcopy(native_receipt) if isinstance(native_receipt,dict) else None
        self.transport_uncertain = transport_uncertain is True


def _job_id(value):
    if type(value) not in (str,int):return None
    text=str(value)
    return text if re.fullmatch(r'[1-9][0-9]{0,19}',text) else None


def submission_result(result):
    """Return public JSON facts. Unknown dispatch stays unknown, never retryable."""
    answer=dict(result)
    native=(result.native_receipt or {}) if isinstance(result,NativeChildResult) else {}
    status=str(native.get('status')or answer.get('status')or'').lower()
    native_id=_job_id(native.get('job_id'))
    explicit_unknown=(getattr(result,'transport_uncertain',False) is True
        or answer.get('uncertain')is True or status in {'uncertain','unknown','submission_unknown'})
    blocked=(native.get('blocked')is True or answer.get('blocked')is True
        or status in {'blocked','needs_resource_review','ready','dry_run','sbatch_not_found'})
    failed=(native.get('failed')is True or answer.get('failed')is True
        or status in {'failed','submit_failed','rejected','error'}
        or bool(native.get('error')or answer.get('error')))
    accepted=(isinstance(result,NativeChildResult) and native_id is not None
        and (status=='submitted' or native.get('submitted')is True) and native.get('submitted')is not False
        and answer.get('submitted')is True and not(explicit_unknown or blocked or failed))
    if accepted:
        answer.update(submitted=True,status='submitted',submission_state='submitted',job_id=native_id,job_ids=[native_id],
            submission_receipt={'native_adapter_verified':True,'status':'submitted','job_ids':[native_id]})
        answer['note']='cDFT Slurm job '+native_id+' was accepted; native outputs are not yet verified.'
        return answer
    # A known real ID plus a conflicting error is not proof the job never ran.
    unknown=(explicit_unknown or native_id is not None or not(blocked or failed
        or native.get('submitted')is False or answer.get('submitted')is False))
    if unknown:
        diagnostic={key:copy.deepcopy(answer.get(key))for key in
            ('error','failed','ok','success','exit_code','returncode','parse_error','status','job_id')if key in answer}
        for key in ('error','failed','ok','success','exit_code','returncode'):answer.pop(key,None)
        answer.update(submitted=False,status='uncertain',submission_state='uncertain',uncertain=True,job_id=None,job_ids=[],
            automatic_retry=False,requires_reconciliation=True,
            reason='Scheduler acceptance is not proven by a current native Slurm receipt; reconcile before any retry.',
            submission_diagnostics=diagnostic,submission_receipt={
                'native_adapter_verified':False,'status':'uncertain','job_ids':[],
                'reported_native_job_id':native_id})
        answer['note']=answer['reason']
        return answer
    state='blocked' if blocked or not failed else 'failed'
    reason=str(native.get('error')or answer.get('error')or native.get('note')or answer.get('stderr')or
        ('Submission was blocked before native acceptance.'if state=='blocked'else'Native scheduler submission failed.'))[:4000]
    typed_status=answer.get('status') if state=='blocked' and answer.get('error_kind') else None
    answer.update(submitted=False,status=typed_status or state,submission_state=state,job_id=None,job_ids=[],failed=state=='failed',
        submission_receipt={'native_adapter_verified':False,'status':state,'job_ids':[]},note=reason)
    if state=='blocked':answer['blocked']=True
    else:answer['error']=reason
    return answer


def record_submission(result, params, *, input_dir, output_dir, register_jobwatch):
    """One path for submit/pipeline display truth; never rewrite old histories."""
    from .native_timing import timed_call
    answer=submission_result(result);answer['output_dir']=str(output_dir)
    if answer['submitted']is True and answer['job_ids']:
        timed_call('jobwatch_registration',register_jobwatch,answer,tool='run_cdft')
    try:
        from .watch_context import get_context
        from .task_line import get_store
        context=get_context();store=get_store();line=context.get('line_id')or context.get('conv_id','')
        timed_call('taskline_update',store.upsert_step,line,context.get('step_id')or'cdft',
            tool='run_cdft',arguments=params,job_ids=answer['job_ids'],status=answer['submission_state'],
            input_dir=str(input_dir),output_dir=str(output_dir),done=False,note=answer['note'],
            username=context.get('username',''),conv_id=context.get('conv_id',''))
    except Exception as error:
        # Tracking failure cannot undo an actually accepted scheduler receipt.
        answer['taskline_receipt_error']=str(error)
    return answer
