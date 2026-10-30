"""Runtime metadata digests must leave scientific evidence and originals intact."""
import copy

from agents import model_context


def runtime_fixture():
    return {
        'nodes': {'sample': {
            'status': 'succeeded', 'tool': 'generate_scientific_report',
            'result_ref': {'call_id': 'proof123', 'evidence_path': '/session/evidence/proof123.json'},
            'path_manifest': {
                'calculation_dir': '/session/work', 'result_dir': '/session/result',
                'input_paths': ['/session/inputs/' + 'a' * 100 + str(i) + '.cif' for i in range(50)],
                'output_files': ['/session/result/' + 'b' * 100 + str(i) + '.cif' for i in range(50)],
                'expected_outputs': [{'kind': 'directory', 'path': '/session/result', 'min_count': 50}],
                'evidence_path': '/session/evidence/proof123.json',
            },
            'error': 'preserve error evidence ' * 500,
            'validation_dossier': {'measurements': [{'uptake': 1.234, 'unit': 'mol/kg'}]},
        }},
        'live_reports': [{
            'step_id': 'sample', 'status': 'succeeded', 'role': 'assembly',
            'report_file': '/session/report.md', 'title': 'Measured report',
            'content': 'Report text with full citations. ' * 600,
            'measurements': [{'uptake': 1.234, 'unit': 'mol/kg'}],
        }],
        'validation_dossier': {'actual_results': [1.234, 2.345]},
    }


def test_model_runtime_digest_preserves_paths_provenance_errors_and_measurements():
    original = runtime_fixture()
    before = copy.deepcopy(original)
    view = model_context.compact_runtime_metadata(original)
    assert original == before
    manifest = view['nodes']['sample']['path_manifest']
    assert manifest['calculation_dir'] == '/session/work'
    assert manifest['result_dir'] == '/session/result'
    assert manifest['expected_outputs'] == before['nodes']['sample']['path_manifest']['expected_outputs']
    for name in ['input_paths', 'output_files']:
        assert manifest[name]['count'] == 50
        assert len(manifest[name]['sha256']) == 64
        assert len(manifest[name]['sample']) == 2
        assert manifest[name]['full_metadata_tool'] == 'lifecycle_state'
    assert view['nodes']['sample']['result_ref'] == before['nodes']['sample']['result_ref']
    assert view['nodes']['sample']['error'] == before['nodes']['sample']['error']
    assert view['nodes']['sample']['validation_dossier'] == before['nodes']['sample']['validation_dossier']
    assert view['validation_dossier'] == before['validation_dossier']
    report = view['live_reports'][0]
    assert report['measurements'] == before['live_reports'][0]['measurements']
    assert report['report_file'] == '/session/report.md'
    assert report['content_reference']['result_ref'] == before['nodes']['sample']['result_ref']
    assert report['content_is_preview'] is True
    assert len(report['content']) <= 512


def test_failed_or_unreferenced_report_previews_are_not_removed():
    for failure in ['failed', 'no_ref', 'no_file']:
        original = runtime_fixture()
        if failure == 'failed':original['live_reports'][0]['status'] = 'failed'
        elif failure == 'no_ref':original['nodes']['sample'].pop('result_ref')
        else:original['live_reports'][0]['report_file'] = ''
        view = model_context.compact_runtime_metadata(original)
        assert view['live_reports'] == original['live_reports']


def test_measurement_records_are_not_mistaken_for_file_metadata():
    original = runtime_fixture()
    original['nodes']['sample']['path_manifest']['files'] = [
        {'path': f'/session/result/{i}.csv', 'uptake_mol_kg': 1.234} for i in range(100)]
    view = model_context.compact_runtime_metadata(original)
    assert view['nodes']['sample']['path_manifest']['files'] == original['nodes']['sample']['path_manifest']['files']


def test_small_manifests_remain_unchanged():
    original = {'nodes': {'sample': {'path_manifest': {'output_files': ['/session/result.csv']}}}}
    assert model_context.compact_runtime_metadata(original) == original
