"""Native subprocess behavior across hostile cwd and installation symlinks."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
import pytest

SOURCE = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audited_python_child',SOURCE/'bimem_agent/python_child.py')
child=importlib.util.module_from_spec(spec);spec.loader.exec_module(child)


def test_relative_interpreter_is_project_relative_not_task_relative(tmp_path,monkeypatch):
    sdk=tmp_path/'SDK';task=tmp_path/'task';(sdk/'bin').mkdir(parents=True);(task/'bin').mkdir(parents=True)
    (sdk/'bin/python').symlink_to(sys.executable)
    poison=task/'bin/python';poison.write_text('#!/bin/sh\necho WRONG_TASK_INTERPRETER\n');poison.chmod(0o700)
    monkeypatch.setattr(child,'project_root',lambda:sdk)
    result=child.run_project_python('./bin/python','print("TRUSTED_INTERPRETER")',cwd=task)
    assert result.returncode==0 and result.stdout.strip()=='TRUSTED_INTERPRETER'
    assert result.args[0]==str(sdk/'bin/python')


def test_relative_path_lookup_is_frozen_before_child_chdir(tmp_path,monkeypatch):
    launcher=tmp_path/'launcher';task=tmp_path/'task';(launcher/'bin').mkdir(parents=True);(task/'bin').mkdir(parents=True)
    (launcher/'bin/python-for-native-test').symlink_to(sys.executable)
    poison=task/'bin/python-for-native-test';poison.write_text('#!/bin/sh\necho WRONG_TASK_PATH\n');poison.chmod(0o700)
    monkeypatch.chdir(launcher)
    env=dict(os.environ,PATH='bin')
    result=child.run_project_python('python-for-native-test','print("TRUSTED_PATH")',cwd=task,environment=env)
    assert result.returncode==0 and result.stdout.strip()=='TRUSTED_PATH'
    assert result.args[0]==str(launcher/'bin/python-for-native-test')


def test_virtualenv_python_symlink_is_not_dereferenced(tmp_path):
    root=tmp_path/'venv';venv.EnvBuilder(with_pip=False,symlinks=True).create(root)
    python=root/'bin/python';assert python.is_symlink()
    result=child.run_project_python(python,'import sys;print(sys.prefix)',cwd=tmp_path)
    assert result.returncode==0 and result.stdout.strip()==str(root)
    assert result.args[0]==str(python)


def test_relative_work_directory_uses_project_root(tmp_path,monkeypatch):
    sdk=tmp_path/'sdk';sdk.mkdir();(sdk/'owned').mkdir()
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir();monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(child,'project_root',lambda:sdk)
    result=child.run_project_python(sys.executable,'import os;print(os.getcwd())',cwd='owned',owned_root=sdk)
    assert result.returncode==0 and result.stdout.strip()==str(sdk/'owned')


def test_implicit_child_work_directory_is_absolute_project_root(tmp_path,monkeypatch):
    sdk=tmp_path/'sdk';sdk.mkdir();monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(child,'project_root',lambda:sdk)
    result=child.run_project_python(sys.executable,'import os;print(os.getcwd())')
    assert result.returncode==0 and result.stdout.strip()==str(sdk)


def test_relative_runtime_override_is_stable_across_chdir(tmp_path,monkeypatch):
    sdk=tmp_path/'sdk';(sdk/'bimem_agent').mkdir(parents=True);(sdk/'config').mkdir()
    (sdk/'bimem_agent/paths.py').write_bytes((SOURCE/'bimem_agent/paths.py').read_bytes())
    config={'paths':{'agent_env_python':sys.executable},'marker':'SDK_CONFIG'}
    (sdk/'config/site.json').write_text(json.dumps(config))
    elsewhere=tmp_path/'elsewhere';(elsewhere/'config').mkdir(parents=True)
    (elsewhere/'config/site.json').write_text(json.dumps({**config,'marker':'WRONG_CWD_CONFIG'}))
    monkeypatch.chdir(elsewhere);monkeypatch.setenv('BIMEM_RUNTIME_CONFIG','config/site.json')
    spec=importlib.util.spec_from_file_location('audited_paths',sdk/'bimem_agent/paths.py')
    paths=importlib.util.module_from_spec(spec);spec.loader.exec_module(paths)
    assert paths.RUNTIME_CONFIG_PATH==sdk/'config/site.json'
    assert paths.load_runtime_config()['marker']=='SDK_CONFIG'


def test_relative_configured_python_preserves_virtualenv_identity(tmp_path,monkeypatch):
    sdk=tmp_path/'sdk';(sdk/'bimem_agent').mkdir(parents=True);(sdk/'config').mkdir()
    (sdk/'bimem_agent/paths.py').write_bytes((SOURCE/'bimem_agent/paths.py').read_bytes())
    venv.EnvBuilder(with_pip=False,symlinks=True).create(sdk/'.venv')
    (sdk/'config/site.json').write_text(json.dumps({'paths':{'agent_env_python':'.venv/bin/python'}}))
    monkeypatch.setenv('BIMEM_RUNTIME_CONFIG',str(sdk/'config/site.json'))
    spec=importlib.util.spec_from_file_location('venv_configured_paths',sdk/'bimem_agent/paths.py')
    paths=importlib.util.module_from_spec(spec);spec.loader.exec_module(paths)
    assert paths.AGENT_ENV_PYTHON==sdk/'.venv/bin/python'
    result=child.run_project_python(paths.AGENT_ENV_PYTHON,'import sys;print(sys.prefix)',cwd=tmp_path)
    assert result.returncode==0 and result.stdout.strip()==str(sdk/'.venv')
