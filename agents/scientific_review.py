"""Shared production/SDK scientific-audit policy, separate from resource monitoring."""
import hashlib
import json
from pathlib import Path

SCIENCE_TOOLS = frozenset({
    'discover_forcefield', 'inspect_forcefield', 'validate_framework_charges',
    'convert_physical_units', 'query_literature',
})


def is_source_result(tool, result):
    """A measured mismatch is valid negative evidence, not a failed read."""
    if not isinstance(result, dict): return False
    if tool == 'validate_framework_charges':
        return (result.get('read_only') is True and result.get('status') in {'validated','charge_mismatch'}
                and bool(result.get('sha256')) and 'cell_net_charge' in result)
    return result.get('ok') is True and not result.get('error')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def canonical_source_inputs(items):
    """Repeated reads/order do not change inputs; distinct hashes remain distinct."""
    return [{'path': path, 'sha256': checksum} for path, checksum in
            sorted({(item['path'], item['sha256']) for item in items})]


def knowledge_fingerprint(root):
    root = Path(root).resolve()
    files = {}
    for relative in ('forcefields/towhee/catalog.json', 'forcefields/towhee/knowledge.json', 'env/forcefield_sources.json', 'env/physical_units.json'):
        path = root / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return digest(files)


def review_action(state, receipt):
    """No resubmission grant. A reviewer outage is not a main-task failure."""
    state['last_receipt'] = receipt
    if receipt.get('review_status') == 'verified':
        state['status'] = 'verified'
        return 'verified'
    if receipt.get('review_status') == 'insufficient_evidence':
        if state.get('verification_retries', 0) < 1:
            state['verification_retries'] = state.get('verification_retries', 0) + 1
            state['status'] = 'retrying_supervisor'
            return 'retry_supervisor'
    elif receipt.get('review_status') == 'rejected_content' and state.get('corrections', 0) < 1:
        state['corrections'] = state.get('corrections', 0) + 1
        state['status'] = 'correcting_main'
        return 'correct_main'
    state['status'] = 'needs_user'
    return 'ask_user'
