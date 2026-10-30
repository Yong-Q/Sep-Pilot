import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from agents.figure_authoring import _publish_verified_pngs
from agents.report_authoring import _read, build_result_packet
from agents.output_contract import artifact_fact, artifact_is_valid
from agents.report_recovery import _OwnedFiles
from agents.local_preparation_recovery import checkpoint_call_witness
from tests.test_parallel_workflow import factory
from tests.test_runtime_authoring_visual import setup, advance


def native_figure(factory, tmp_path, monkeypatch):
    def declared(nodes, **kwargs):
        for node in nodes:
            if node['tool'] == 'generate_result_figures':
                node['expected_outputs'] = [{'kind': 'directory',
                    'path': node['arguments']['output_dir'], 'pattern': '*.png', 'min_count': 1}]
        return factory(nodes, **kwargs)
    rt, output, calls = setup(declared, tmp_path, monkeypatch, 'figure')
    done = advance(rt, 'delivery', timeout=35)
    assert done['status'] == 'succeeded', done
    return rt, output, done


def report_packet(rt, done):
    state = copy.deepcopy(rt.snapshot())
    state['nodes']['report'] = {'status': 'running', 'token': 'report-token',
        'contract': {'step_id': 'report', 'tool': 'generate_scientific_report',
            'agent': 'communicator', 'depends_on': ['delivery'],
            'arguments': {'source_steps': ['delivery'], 'title': 'Native reviewed PNG',
                          'output_path': str(rt.root / 'report.md')}}}
    owner = {'username': rt.username, 'conv_id': rt.conv_id}
    def attest(source, call, raw):
        files = _OwnedFiles(rt.root, owner)
        try:
            path = source['result_ref']['checkpoint_path']
            witness = checkpoint_call_witness(files, path, call,
                step_id=source['contract']['step_id'], require_result_hash=True)
            return {'checkpoint_verified': True, 'checkpoint_path': path,
                    'checkpoint_sha256': witness['sha256']}
        finally: files.close()
    return build_result_packet(rt.root, owner, state, 'report', attest=attest)


def test_native_wildcard_contract_and_downstream_packet_use_exact_reviewed_publication(factory, tmp_path, monkeypatch):
    rt, output, done = native_figure(factory, tmp_path, monkeypatch)
    result = done['result']; published = result['artifacts'][0]
    original = result['attempt_artifacts'][0]
    assert Path(published['path']).parent == output
    assert Path(published['path']).name == 'figure-' + published['sha256'] + '.png'
    assert Path(original['path']).parent == output / 'attempt-1'
    assert Path(published['path']).read_bytes() == Path(original['path']).read_bytes()
    fact = artifact_fact(done['contract']['expected_outputs'][0], rt.root)
    assert artifact_is_valid(fact, started_at=done['started_at'])
    assert len(fact['files']) == 1 and fact['files'][0][0] == published['path']
    packet = report_packet(rt, done)
    assert len(packet['figures']) == 1
    figure = packet['figures'][0]
    assert figure['visual_review'] == 'verified'
    proof = figure['visual_evidence']
    assert proof['native_publication_byte_identity_verified'] is True
    assert proof['image_path'] == str(Path(published['path']).relative_to(rt.root))
    assert proof['reviewed_image_path'] == str(Path(original['path']).relative_to(rt.root))
    assert proof['image_sha256'] == original['sha256']
    assert not result['scientific_acceptance']


@pytest.mark.parametrize('change', ['published_bytes', 'attempt_bytes', 'review_checkpoint', 'forged_source'])
def test_published_figure_cannot_inherit_stale_or_forged_review(factory, tmp_path, monkeypatch, change):
    rt, output, done = native_figure(factory, tmp_path, monkeypatch)
    result = done['result']
    if change == 'published_bytes': Path(result['artifacts'][0]['path']).write_bytes(b'changed PNG')
    elif change == 'attempt_bytes': Path(result['attempt_artifacts'][0]['path']).write_bytes(b'changed original')
    elif change == 'review_checkpoint':
        cid = result['visual_review']['call_id']
        evidence = rt.root / 'evidence' / (cid + '.json')
        data = json.loads(evidence.read_text()); data['params']['report_token'] = 'foreign-token'
        evidence.write_text(json.dumps(data))
    else:
        # Change every public producer claim consistently; independent visual
        # CP still names the real original and must reject this second path.
        original = Path(result['attempt_artifacts'][0]['path'])
        other = original.with_name('forged-source.png'); other.write_bytes(original.read_bytes())
        source = dict(result['attempt_artifacts'][0], path=str(other))
        result['artifacts'][0]['review_source']['path'] = str(other.relative_to(rt.root))
        result['attempt_artifacts'] = [source]
        result['sandbox_receipt']['artifacts'] = [source]
        from agents.state_io import json_transaction
        with json_transaction(rt.store.path) as data:
            data['workflows'][rt.workflow_id]['nodes']['delivery']['result'] = result
        call_path = Path(done['result_ref']['evidence_path'])
        call = json.loads(call_path.read_text()); call['result'] = result
        call_path.write_text(json.dumps(call))
        # Independently test the private-review binding even before the outer
        # producer-checkpoint witness rejects the modified public receipt.
        from agents.report_native_inputs import verified_visual_evidence
        _, published_fact = _read(rt.root, result['artifacts'][0]['path'], {'bytes': 0})
        with pytest.raises(ValueError, match='visual verdict'):
            verified_visual_evidence(rt.root, rt.snapshot(), 'delivery', published_fact,
                read=_read, budget={'bytes': 0}, file_facts=[], cache={})
    with pytest.raises((ValueError, OSError, PermissionError)):
        report_packet(rt, done)


def publication_fixture(tmp_path):
    root = tmp_path / 'runs/owner/case'; attempt = root / 'figures/attempt-1'
    attempt.mkdir(parents=True)
    source = root / 'input.csv'; source.write_text('x,y\n1,2\n')
    _, fact = _read(root, source, {'bytes': 0})
    image = attempt / 'result.png'; Image.new('RGB', (20, 20), (60, 90, 120)).save(image)
    raw, png = _read(root, image, {'bytes': 0})
    artifact = {'path': str(image), 'sha256': png['sha256'], 'bytes': png['size'], 'kind': 'PNG'}
    receipt = {'path': png['path'], 'sha256': png['sha256'], 'bytes': png['size']}
    visual = {'verified': True, 'call_id': 'actual-visual-call', 'image_receipts': [receipt]}
    from agents.report_authoring import _digest
    packet = {'file_facts': [fact]}
    packet['packet_sha256'] = _digest(packet)
    return root, root / 'figures', artifact, visual, packet, raw


@pytest.mark.parametrize('fault', ['missing_review', 'wrong_sha', 'wrong_size', 'source_mutation',
                                  'control_changed', 'existing_different', 'existing_symlink'])
def test_publication_requires_exact_review_current_control_and_no_overwrite(tmp_path, fault):
    root, output, artifact, visual, packet, raw = publication_fixture(tmp_path)
    destination = output / ('figure-' + artifact['sha256'] + '.png')
    if fault == 'missing_review': visual['image_receipts'] = []
    elif fault == 'wrong_sha': visual['image_receipts'][0]['sha256'] = '0' * 64
    elif fault == 'wrong_size': visual['image_receipts'][0]['bytes'] += 1
    elif fault == 'existing_different': destination.write_bytes(b'old scientific result')
    elif fault == 'existing_symlink': destination.symlink_to(Path(artifact['path']))
    def control():
        if fault == 'source_mutation': Path(artifact['path']).write_bytes(b'changed original')
        return fault != 'control_changed'
    with pytest.raises((ValueError, OSError, PermissionError)):
        _publish_verified_pngs(root, output, [artifact], visual, packet, control)
    if fault == 'existing_different': assert destination.read_bytes() == b'old scientific result'
    elif fault == 'existing_symlink': assert destination.is_symlink()
    else: assert not destination.exists()
    assert not list(output.glob('.figure-publish-*'))


def test_repeated_same_content_publication_is_exact_and_retains_original(tmp_path):
    root, output, artifact, visual, packet, raw = publication_fixture(tmp_path)
    first = _publish_verified_pngs(root, output, [artifact], visual, packet, lambda: True)
    second = _publish_verified_pngs(root, output, [artifact], visual, packet, lambda: True)
    assert first == second and Path(first[0]['path']).read_bytes() == raw
    assert Path(artifact['path']).read_bytes() == raw
