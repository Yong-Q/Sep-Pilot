"""Factual progress accounting; no classification of user wording or intent."""
import hashlib
import json

from .recovery import failure_reason, result_object


VOLATILE = {'call_id', 'tool_use_id', 'evidence_ref', 'observed_at', 'updated_at', 'reviewed_at',
            'service_heartbeat', 'heartbeat', 'elapsed', 'estimated_wait_seconds', 'resource_review_id', 'time'}


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, (list, tuple)):
        return [stable(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(stable(value), sort_keys=True, default=str).encode()).hexdigest()


class ReflectionProgress:
    def __init__(self):
        self.seen = set()
        self.state = None
        self.stalled = 0
        self.last_problem = None

    def observe(self, tool, arguments, result, execution_state):
        obj = result_object(result)
        failed = bool(failure_reason(result) or obj.get('error') or obj.get('blocked') or obj.get('executed') is False)
        contract_repair = obj.get('error_kind') in {'schema_validation', 'workflow_contract'}
        state = digest(execution_state)
        changed = self.state is not None and self.state != state
        self.state = state
        signature = digest([tool, arguments, obj or result])
        new = signature not in self.seen
        self.seen.add(signature)
        # A changed parameter/contract proposal is real diagnostic progress even
        # when preflight still rejects it: no compute was launched and the next
        # schema response may identify another field.  Repeating the identical
        # invalid call remains stalled and is still bounded.
        progress = changed or new and (not failed or contract_repair)
        self.stalled = 0 if progress else self.stalled + 1
        if failed:
            self.last_problem = {'tool': tool, 'arguments': arguments, 'actual_result': obj or str(result)[:2500]}
        return {'progress': progress, 'new_evidence': new and (not failed or contract_repair), 'execution_changed': changed,
                'stalled_calls': self.stalled, 'last_problem': self.last_problem,
                'instruction': 'Use the actual error to diagnose and choose a different valid recovery operation. Restore existing verified results before retrying. Ask the user only for a genuinely new scientific choice or missing external authority.'}
