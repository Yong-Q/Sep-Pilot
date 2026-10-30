"""Capture source only at the real trusted Python child transport boundary."""
import os
from pathlib import Path
from bimem_agent import python_child


def native_source(command, invocation):
    assert len(command) == 3 and command[2:] == ['--native-inline']
    script = Path(command[1])
    assert script.is_absolute() and script.resolve() == Path(python_child.__file__).resolve()
    assert invocation.get('capture_output') is True and invocation.get('text') is True
    source = invocation.get('input')
    assert isinstance(source, str) and source
    assert source not in command
    environment = invocation.get('env')
    assert isinstance(environment, dict)
    assert environment.get('PYTHONPATH', '').split(os.pathsep)[0] == str(python_child.project_root())
    return source
