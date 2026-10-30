"""Native user-control lifetime, distinct from main-chat output turn ownership.

Only API admission/interrupt/clear call advance_user_control. Model-facing tools
have no schema for supplying or changing this epoch. Live getters never cache.
"""
_MISSING=object()


def advance_user_control(conv):
    epoch=getattr(conv,'_workflow_user_control_epoch',0)
    if type(epoch)is not int or epoch<0:raise ValueError('invalid native user control epoch')
    conv._workflow_user_control_epoch=epoch+1


def bind_api_workflow_control(conv,actor):
    owner=(conv.username,conv.conv_id)
    def read():
        return {'epoch':getattr(conv,'_workflow_user_control_epoch',0),
            'owner':(conv.username,conv.conv_id),
            'actor_bound':conv.session is actor and (conv.username,conv.conv_id)==owner,
            'interrupted':bool(getattr(conv,'interrupt_requested',False)
                               or getattr(conv,'interrupt_message',''))}
    actor._native_workflow_control=read


def workflow_control_guard(main,username,conv_id):
    """Capture a native capability once; re-read live user control every check.

    A getter replacement/error, epoch corruption or owner/actor mismatch fails
    closed. Non-API sessions retain the historical UID+interrupt boundary.
    """
    getter=getattr(main,'_native_workflow_control',_MISSING)
    if getter is _MISSING:
        uid=getattr(main,'_api_turn_id',None)
        def legacy():
            try:return (getattr(main,'_native_workflow_control',_MISSING)is _MISSING
                and getattr(main,'_api_turn_id',None)==uid and not main._interrupt_requested())
            except Exception:return False
        return legacy
    expected_owner=(username,conv_id)
    def read():
        if not callable(getter)or getattr(main,'_native_workflow_control',_MISSING)is not getter:
            raise ValueError('native workflow control binding replaced')
        current=getter()
        if (not isinstance(current,dict)or set(current)!={'epoch','owner','actor_bound','interrupted'}
                or type(current['epoch'])is not int or current['epoch']<0
                or current['owner']!=expected_owner or current['actor_bound']is not True
                or current['interrupted']is not False):
            raise ValueError('native user control revoked')
        return current['epoch']
    try:epoch=read()
    except Exception:return lambda:False
    def valid():
        try:return read()==epoch
        except Exception:return False
    return valid


def workflow_control_current(main,username,conv_id):
    return workflow_control_guard(main,username,conv_id)()
