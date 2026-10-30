"""Static native review diagnostics. Never persist provider exception text."""
import asyncio

REASONS = {
    'identity_mismatch': 'The isolated reviewer does not match the native owner, approved contract, or attempt.',
    'goal_changed': 'The approved goal changed while review was in progress.',
    'turn_changed': 'The primary conversation turn changed while review was in progress.',
    'controls_changed': 'The active conversation controls changed or requested interruption.',
    'review_claim_unavailable': 'The review was already claimed, exhausted, or superseded.',
    'attempt_superseded': 'The reviewed native attempt is no longer awaiting acceptance.',
    'evidence_invalid': 'The finish request lacks fresh native evidence bound to this attempt.',
    'sources_changed': 'The reviewed native files or provenance changed before acceptance.',
    'finish_not_committed': 'The native finish transaction did not commit this review.',
    'review_cancelled': 'The independent review was cancelled before acceptance completed.',
    'unexpected_review_failure': 'The review ended before acceptance; native provenance must be inspected.',
}

class ReviewFailure(ValueError):
    def __init__(self, code):
        if code not in REASONS:code='unexpected_review_failure'
        self.code=code
        super().__init__(REASONS[code])


def failure_record(error):
    # Do not stringify unknown errors or their attributes, class names, request,
    # headers, provider bodies, URLs, or chained exceptions.
    code=error.code if isinstance(error,ReviewFailure)and error.code in REASONS else (
        'review_cancelled' if isinstance(error,asyncio.CancelledError) else 'unexpected_review_failure')
    known=(ValueError,PermissionError,RuntimeError,TypeError,KeyError,OSError,TimeoutError,asyncio.CancelledError,ReviewFailure)
    kind=type(error).__name__ if type(error)in known else 'Exception'
    return {'code':code,'reason':REASONS[code],'error_type':kind,'scientific_acceptance':False}
