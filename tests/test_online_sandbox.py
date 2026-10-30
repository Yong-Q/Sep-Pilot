"""Real sandbox regression using a local fake index, not a public download."""
import hashlib, io, json, os, socket, threading, time, zipfile
from pathlib import Path
import pytest
from agents.plot_sandbox import plot_result_code


def fake_index():
    wheel=io.BytesIO()
    with zipfile.ZipFile(wheel,'w') as z:
        z.writestr('sandboxprobe/__init__.py','VALUE=42\n')
        z.writestr('sandboxprobe-1.0.0.dist-info/METADATA','Metadata-Version: 2.1\nName: sandboxprobe\nVersion: 1.0.0\n')
        z.writestr('sandboxprobe-1.0.0.dist-info/WHEEL','Wheel-Version: 1.0\nGenerator: task-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        z.writestr('sandboxprobe-1.0.0.dist-info/RECORD','')
    name='sandboxprobe-1.0.0-py3-none-any.whl';data=wheel.getvalue()
    server=socket.socket();server.bind(('127.0.0.1',0));server.listen(8);server.settimeout(.2)
    stop=threading.Event()
    def serve():
        while not stop.is_set():
            try:conn,_=server.accept()
            except socket.timeout:continue
            except OSError:break
            with conn:
                request=conn.recv(8192).split(b'\r\n',1)[0]
                if request.startswith(b'CONNECT '):
                    conn.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
                    request=conn.recv(8192).split(b'\r\n',1)[0]
                if b'/simple/sandboxprobe/' in request:
                    body=f'<a href="http://example.invalid/files/{name}">{name}</a>'.encode();kind='text/html'
                elif name.encode() in request:body=data;kind='application/octet-stream'
                else:body=b'not found';kind='text/plain'
                conn.sendall(f'HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n'.encode()+body)
    t=threading.Thread(target=serve,daemon=True);t.start()
    return server,stop,t


def test_real_sandbox_pip_and_persistent_task_workspace(tmp_path):
    server,stop,thread=fake_index()
    root=tmp_path/'runs'/'u'/'c';root.mkdir(parents=True)
    source=root/'source.csv';source.write_text('x\n1\n')
    owner={'username':'u','conv_id':'c'};producer={'step_id':'analysis','runtime_token':'token-1'}
    install="""import subprocess,sys,pathlib
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-cache-dir','--index-url','http://example.invalid/simple/','--trusted-host','example.invalid','sandboxprobe==1.0.0'],check=True)
import sandboxprobe
assert sandboxprobe.VALUE==42
subprocess.run('printf shell-ok > /work/shell-result',shell=True,check=True)
assert pathlib.Path('/work/shell-result').read_text()=='shell-ok'
pathlib.Path('/output/first.txt').write_text('ok')
"""
    again="""import pathlib,sandboxprobe
assert sandboxprobe.VALUE==42
try:pathlib.Path('/inputs/T1.csv').write_text('bad')
except OSError:pass
else:raise AssertionError('input writable')
pathlib.Path('/output/second.txt').write_text('ok')
"""
    try:
        common={'input_files':[{'path':str(source),'input_id':'T1','format':'csv'}],
                'owned_root':root,'timeout_seconds':60,'_mode':'data','_owner':owner,'_producer':producer,
                '_online':True,'_allow_local':True,'_proxy_port':server.getsockname()[1]}
        a=plot_result_code(install,output_dir=str(root/'first'),_declared_outputs=[{'name':'first.txt','format':'txt'}],**common)
        assert a.get('ok') is True, {k:a.get(k) for k in ('error','status','sandbox','exit_code')}
        b=plot_result_code(again,output_dir=str(root/'second'),_declared_outputs=[{'name':'second.txt','format':'txt'}],**common)
        assert b.get('ok') is True, {k:b.get(k) for k in ('error','status','sandbox','exit_code')}
        assert source.read_text()=='x\n1\n'
        assert (root/'second/artifacts/second.txt').read_text()=='ok'
    finally:
        stop.set();server.close();thread.join(timeout=2)


def test_online_mode_requires_native_owner_and_task(tmp_path):
    root=tmp_path/'runs'/'u'/'c';root.mkdir(parents=True)
    with pytest.raises(PermissionError):
        plot_result_code('print(1)',output_dir=str(root/'out'),owned_root=root,_mode='data',
            _owner={'username':'u','conv_id':'c'},_producer={},_online=True,_proxy_port=1234)


def test_registry_routes_network_from_trusted_operator_and_native_context(tmp_path, monkeypatch):
    from agents import registry, workspace, watch_context, plot_sandbox
    root=tmp_path/'runs'/'u'/'c';root.mkdir(parents=True)
    monkeypatch.setattr(workspace,'session_root',lambda:root)
    monkeypatch.setattr(watch_context,'get_context',lambda:{'username':'u','conv_id':'c','step_id':'step','runtime_token':'token'})
    monkeypatch.setenv('HTTPS_PROXY','http://127.0.0.1:12345')
    calls=[]
    monkeypatch.setattr(plot_sandbox,'plot_result_code',lambda **kw:calls.append(kw) or {'ok':True})
    params={'script':'print(1)','input_files':[],'output_dir':str(root/'out'),
            'declared_outputs':[{'name':'out.txt','format':'txt'}]}
    assert registry._exec_run_analysis_sandbox(params)['ok'] is True
    assert calls[-1]['_online'] is True and calls[-1]['_proxy_port']==12345
    assert calls[-1]['_producer']=={'step_id':'step','runtime_token':'token'}
    assert registry._exec_run_analysis_sandbox({**params,'network_access':False})['ok'] is True
    assert '_online' not in calls[-1]
    monkeypatch.setenv('HTTPS_PROXY','http://user:secret@127.0.0.1:12345')
    result=registry._exec_run_analysis_sandbox({**params,'network_access':True})
    assert result['error_kind']=='sandbox_network_unavailable'
    assert 'secret' not in str(result)


def test_background_descendants_stop_before_outputs_are_published(tmp_path):
    server,stop,thread=fake_index()
    root=tmp_path/'runs'/'u'/'c';root.mkdir(parents=True)
    source=root/'source.csv';source.write_text('x\n1\n')
    owner={'username':'u','conv_id':'c'};producer={'step_id':'analysis','runtime_token':'token-2'}
    script="""import pathlib,subprocess,sys
subprocess.Popen([sys.executable,'-c',"import time,pathlib; time.sleep(2); pathlib.Path('/work/late').write_text('bad')"])
pathlib.Path('/output/result.txt').write_text('done')
"""
    try:
        result=plot_result_code(script,input_files=[{'path':str(source),'input_id':'T1','format':'csv'}],
            output_dir=str(root/'output'),owned_root=root,timeout_seconds=30,_mode='data',_owner=owner,
            _producer=producer,_online=True,_proxy_port=server.getsockname()[1],
            _declared_outputs=[{'name':'result.txt','format':'txt'}])
        assert result.get('ok') is True, {k:result.get(k) for k in ('error','status','sandbox','exit_code')}
        key=hashlib.sha256(json.dumps({'owner':owner,'producer':producer},sort_keys=True).encode()).hexdigest()
        time.sleep(3)
        assert not (root/'.sandbox-work'/key/'late').exists()
    finally:
        stop.set();server.close();thread.join(timeout=2)
