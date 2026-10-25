"""cDFT accepts an explicit CIF or a bounded directory without losing identities."""
import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def generator():
    pytest.importorskip('numpy')
    path = Path(__file__).resolve().parents[1] / 'tools/cdft/cDFT_Initialization/data_input.py'
    spec = importlib.util.spec_from_file_location('cdft_input_sources_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_cif_is_selected_instead_of_searched_as_a_directory(tmp_path, generator):
    source = tmp_path / 'MOF-5.cif'
    source.write_text('data_structure\n')
    assert generator._cif_dataset(source) == [source]


def test_directory_keeps_nested_cifs_and_rejects_duplicate_names(tmp_path, generator):
    (tmp_path / 'nested').mkdir()
    first = tmp_path / 'first.cif'
    second = tmp_path / 'nested/second.cif'
    first.write_text('data_first\n')
    second.write_text('data_second\n')
    assert generator._cif_dataset(tmp_path) == [first, second]
    (tmp_path / 'nested/first.cif').write_text('data_duplicate\n')
    with pytest.raises(ValueError, match='duplicate CIF basename'):
        generator._cif_dataset(tmp_path)


def test_directory_rejects_symlink_escape(tmp_path, generator):
    source = tmp_path / 'inputs'
    source.mkdir()
    outside = tmp_path / 'outside.cif'
    outside.write_text('data_outside\n')
    (source / 'escape.cif').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink escapes'):
        generator._cif_dataset(source)
