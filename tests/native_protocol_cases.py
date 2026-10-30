"""Synthetic native bytes and a no-network transport at the actual interfaces."""
from pathlib import Path
import shutil
import socket
import json


def cdft_batch(root,names=('amber','birch')):
    from tests.test_cdft_result_facts import fixture
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    _,example,run=fixture(root.parent/(root.name+'-native-template'))
    source=root.parent/(root.name+'-original-inputs');source.mkdir()
    (root/'input').mkdir()
    payload=(example/'input/a.dat').read_bytes()
    for name in names:
        (source/(name+'.dat')).write_bytes(payload);(root/'input'/(name+'.dat')).write_bytes(payload)
        dest=root/'input_dir'/name;dest.mkdir(parents=True)
        (dest/'input.dat').write_bytes(payload)
        for filename in ('output.dat','hr.dat'):(dest/filename).write_bytes((run/filename).read_bytes())
    (root/'results.csv').write_text('MOF,N2_density_mol_L,N2_henry_mol_L_atm\n'+''.join(name+',1.2,1.3\n'for name in names))
    return {'action':'submit','gas':'N2','input_dir':str(source),'completion_policy':'strict','min_success_ratio':1.0}


def gcmc_case(root,pressure=1e5,text=None):
    from tests.test_gcmc_result_contract import native_case,DATA
    return native_case(Path(root),format(pressure,'.2e'),DATA if text is None else text)


def public_https_transport(monkeypatch,payload,address='93.184.216.34'):
    """Exercise public DNS, IP pinning and TLS request construction, no sockets."""
    import urllib3
    records=[]
    class Response:
        status=200
        headers={'Content-Length':str(len(payload)),'Content-Type':'chemical/x-cif','ETag':'test-only'}
        def stream(self,chunk_size,decode_content=True):
            assert chunk_size==65536 and decode_content is True
            yield payload[:8];yield payload[8:]
        def close(self):records.append(('response_closed',))
    class Pool:
        def __init__(self,host,port=443,**kwargs):
            records.append(('pool',host,port,kwargs))
            assert host==address and port==443
            assert kwargs['cert_reqs']=='CERT_REQUIRED'
            assert kwargs['server_hostname']==kwargs['assert_hostname']=='example.org'
        def urlopen(self,method,target,**kwargs):
            records.append(('request',method,target,kwargs))
            assert method=='GET'and kwargs['headers']['Host']=='example.org'
            assert kwargs['preload_content']is False and kwargs['redirect']is False and kwargs['retries']is False
            return Response()
        def close(self):records.append(('pool_closed',))
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(socket.AF_INET,socket.SOCK_STREAM,6,'',(address,443))])
    monkeypatch.setattr(urllib3,'HTTPSConnectionPool',Pool)
    return records
