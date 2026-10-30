"""Native read-only delivery check, bound internally to one WorkflowStore.

The shared flock is the existing store's writer lock, not a new authorization
or a cache. The observation linearizes at the final real source SHA under that
lock. The caller must check its Session/UID/goal again after worker completion
and do no further file I/O before selecting the user-visible publication.
"""
import fcntl
import os
from pathlib import Path
import stat
import time


def bind_workflow_reply_guard(store, *, username, conv_id, is_current=None):
    from .parallel_workflow import WorkflowStore
    if not isinstance(store, WorkflowStore):
        raise TypeError('delivery guard requires the native WorkflowStore')
    workflow_id=WorkflowStore.identity(username,conv_id)
    path=Path(store.path).parent.resolve()/Path(store.path).name

    def unavailable():
        return {'guarded':False,'sources_valid':False,'facts':None}

    def inspect(*, goal, evidence_root, references):
        from .answer_delivery import sources_unchanged
        from .workflow_reply_facts import current_task_status
        if is_current is not None and is_current()is not True:return unavailable()
        fd=None;locked=False
        try:
            # Never create/replace state or its lock. Unknown native binding
            # is a limited answer, not a reason to manufacture execution facts.
            if path.is_symlink():return unavailable()
            fd=os.open(str(path)+'.lock',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            if not stat.S_ISREG(os.fstat(fd).st_mode):return unavailable()
            deadline=time.monotonic()+2
            while True:
                try:
                    fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB);locked=True;break
                except BlockingIOError:
                    if time.monotonic()>=deadline:return unavailable()
                    time.sleep(.01)
            if is_current is not None and is_current()is not True:return unavailable()
            info=os.stat(path,follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=64*1024*1024:return unavailable()
            state=store.snapshot(workflow_id)
            facts=current_task_status(state,username=username,conv_id=conv_id,
                goal=goal,evidence_root=evidence_root)
            if facts is None:return unavailable()
            # This SHA is deliberately LAST: neither a stat tuple nor a
            # prior hash survives an additional uncontrolled status read.
            valid=sources_unchanged(references)
            if is_current is not None and is_current()is not True:return unavailable()
            return {'guarded':True,'sources_valid':valid,'facts':facts,
                'workflow_id':workflow_id,'linearized_at':time.time()}
        except (OSError,ValueError,TypeError,KeyError):return unavailable()
        finally:
            if fd is not None:
                try:
                    if locked:fcntl.flock(fd,fcntl.LOCK_UN)
                finally:os.close(fd)
    return inspect
