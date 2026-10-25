"""Configured deployment libraries reach subprocesses without replacing MPI paths."""
import importlib.util
from pathlib import Path
import sys


def test_md_batch_bootstrap_preserves_module_libraries_and_adds_configured_runtime(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools/md_runtime.py'
    spec = importlib.util.spec_from_file_location('md_library_runtime', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    interface = tmp_path / 'interface'
    interface.write_text('#!' + sys.executable + '\nimport os\n'
                         'print(os.environ["LD_LIBRARY_PATH"], flush=True)\nraise SystemExit(7)\n')
    interface.chmod(0o700)
    source = tmp_path / 'input.cif'; source.write_text('synthetic fixture')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/module/mpi/lib')
    result = module.run(str(source), str(tmp_path / 'work'), executable='/bin/true',
                        interface_executable=str(interface), n_procs=1,
                        library_path='/configured/environment/lib')
    assert result['success'] is False
    assert (tmp_path / 'work/interface.log').read_text().strip() == '/module/mpi/lib:/configured/environment/lib'
