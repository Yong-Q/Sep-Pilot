import importlib.util
from pathlib import Path

import pytest

import agents.report_authoring as module

@pytest.mark.parametrize('filename,label', [('source.json', 'JSON'), ('source.csv', 'CSV'),
                                             ('source.cif', 'CIF'), ('source.unknown', 'source file')])
def test_table_link_identifies_actual_file_format(filename, label):
    packet = {'title': 'Result', 'sources': [{'id': 'S1', 'step_id': 'calc',
        'call_id': 'call', 'source_sha256': 'a' * 64, 'conditions': {},
        'documents': [], 'structured_data': [], 'text_documents': []}],
        'tables': [{'id': 'T1', 'source_id': 'S1', 'path': '/owned/' + filename,
                    'columns': ['value'], 'rows': [{'value': 1}], 'row_count': 1,
                    'sha256': 'b' * 64}], 'references': [], 'figures': [],
        'report_output_path': '/owned/report.md'}
    draft = {'blocks': [{'kind': 'narrative', 'text': 'Observed result.',
                         'source_ids': ['S1'], 'claim_type': 'method'},
                        {'kind': 'table', 'table_id': 'T1'}]}
    text = module.render_validated_draft(packet, draft)
    assert 'Complete source data: [' + label + '](' + filename + ')' in text
