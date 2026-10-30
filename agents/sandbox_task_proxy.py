"""Task-private byte relay between an isolated network namespace and proxy.

This module implements only the transport. Callers must bind the containing
private directory into the sandbox, bring up its loopback interface, and
enforce task identity, timeout, cancellation and resource limits separately.
It never forwards environment variables or credentials from the parent.
"""
from contextlib import contextmanager
import os
import ipaddress
import json
from pathlib import Path
import selectors
import socket
import stat
import threading
from urllib.parse import urlsplit
from urllib.parse import urlencode
from urllib.request import Request, build_opener, ProxyHandler

MAX_HEADER=65536


def _target_host(host, port, resolver):
    if not host or not 1<=port<=65535:raise ValueError('invalid destination')
    try:
        address=ipaddress.ip_address(host)
        addresses=[address]
    except ValueError:
        # Resolve once here and send the validated numeric address to the
        # upstream proxy: the proxy cannot independently re-resolve a hostname
        # to a different, private target after our check.
        answers=resolver(host,port,type=socket.SOCK_STREAM)
        addresses=list({ipaddress.ip_address(item[4][0]) for item in answers})
    if not addresses or any(not item.is_global for item in addresses):
        raise PermissionError('destination is not a public network address')
    selected=sorted(addresses,key=lambda value:(value.version,str(value)))[0]
    return ('['+str(selected)+']' if selected.version==6 else str(selected))+':'+str(port)


def _read_head(conn):
    data=bytearray();conn.settimeout(10)
    while b'\r\n\r\n' not in data:
        chunk=conn.recv(min(8192,MAX_HEADER+1-len(data)))
        if not chunk:raise ValueError('incomplete proxy request')
        data.extend(chunk)
        if len(data)>MAX_HEADER:raise ValueError('proxy request headers too large')
    head,rest=bytes(data).split(b'\r\n\r\n',1)
    return head,rest


def _prepare_proxy_request(head,resolver):
    lines=head.split(b'\r\n')
    if not lines:raise ValueError('empty proxy request')
    try:method,target,version=lines[0].decode('ascii').split(' ')
    except (ValueError,UnicodeError):raise ValueError('invalid proxy request') from None
    if version not in ('HTTP/1.0','HTTP/1.1') or not method.isalpha():
        raise ValueError('invalid proxy request line')
    if method=='CONNECT':
        parsed=urlsplit('//'+target)
        if parsed.username is not None or parsed.password is not None or parsed.path:
            raise ValueError('invalid CONNECT target')
        port=parsed.port or 443
        origin=None
    else:
        parsed=urlsplit(target)
        if parsed.scheme!='http' or parsed.username is not None or parsed.password is not None or not parsed.hostname:
            raise ValueError('unsupported HTTP target')
        port=parsed.port or 80
        origin=(parsed.path or '/')+('?' + parsed.query if parsed.query else '')
    destination=_target_host(parsed.hostname,port,resolver)
    tunnel=('CONNECT '+destination+' HTTP/1.1\r\nHost: '+destination+'\r\n\r\n').encode('ascii')
    if method=='CONNECT':return tunnel,None
    # The tunnel is pinned to a single validated public IP.  Forward an
    # origin-form HTTP request, retaining the original Host header.
    clean=[line for line in lines[1:] if not line.lower().startswith((b'proxy-authorization:',b'proxy-connection:'))]
    return tunnel,(method+' '+origin+' '+version).encode('ascii')+b'\r\n'+b'\r\n'.join(clean)+b'\r\n\r\n'


def _relay(a, b, stop):
    sel = selectors.DefaultSelector()
    try:
        a.setblocking(False); b.setblocking(False)
        sel.register(a, selectors.EVENT_READ, b)
        sel.register(b, selectors.EVENT_READ, a)
        while not stop.is_set():
            for key, _ in sel.select(timeout=0.25):
                try:
                    data = key.fileobj.recv(65536)
                    if not data: return
                    key.data.setblocking(True)
                    key.data.settimeout(10)
                    key.data.sendall(data)
                    key.data.setblocking(False)
                except (OSError, TimeoutError): return
    finally:
        sel.close()
        for conn in (a, b):
            try: conn.close()
            except OSError: pass


class _Bridge:
    def __init__(self, listener, connect, max_connections, prepare=None, direct=False):
        self.listener, self.connect = listener, connect
        self.prepare=prepare
        self.direct=direct
        self.stop = threading.Event()
        self.limit = threading.BoundedSemaphore(max_connections)
        self.threads = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _client(self, conn):
        try:
            if self.prepare is not None:
                head,rest=_read_head(conn)
                tunnel,origin=self.prepare(head)
            else:tunnel=origin=None;rest=b''
            upstream = self.connect(tunnel) if self.direct else self.connect()
            if tunnel is not None:
                if self.direct:response=b'HTTP/1.1 200 Connection established';response_rest=b''
                else:
                    upstream.settimeout(10);upstream.sendall(tunnel)
                    response,response_rest=_read_head(upstream)
                    if not response.split(b'\r\n',1)[0].startswith((b'HTTP/1.1 200 ',b'HTTP/1.0 200 ')):
                        conn.sendall(b'HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n');upstream.close();return
                if origin is None:conn.sendall(response+b'\r\n\r\n'+response_rest)
                else:upstream.sendall(origin+rest);rest=b''
                if rest:upstream.sendall(rest)
            _relay(conn, upstream, self.stop)
        except (OSError,ValueError,PermissionError,UnicodeError):
            try:conn.sendall(b'HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n')
            except OSError:pass
            conn.close()
        finally:
            self.limit.release()

    def _run(self):
        self.listener.settimeout(0.25)
        while not self.stop.is_set():
            try: conn, _ = self.listener.accept()
            except socket.timeout: continue
            except OSError: break
            if not self.limit.acquire(blocking=False):
                conn.close(); continue
            t = threading.Thread(target=self._client, args=(conn,), daemon=True)
            self.threads.append(t); t.start()
            self.threads[:] = [item for item in self.threads if item.is_alive()]

    def close(self):
        self.stop.set()
        self.listener.close()
        self.thread.join(timeout=2)
        for thread in self.threads: thread.join(timeout=2)


def resolve_public_doh(host, port, **_):
    """Resolve an internet name using authenticated DNS-over-HTTPS.

    It is used only when the operator chooses direct egress. The caller still
    validates and pins every returned IP before connecting. No task headers,
    credentials or response bodies are sent to the DNS provider.
    """
    name=host.encode('idna').decode('ascii')
    if len(name)>253 or any(not part or len(part)>63 for part in name.rstrip('.').split('.')):
        raise ValueError('invalid DNS name')
    opener=build_opener(ProxyHandler({}))
    for kind in ('A','AAAA'):
        url='https://cloudflare-dns.com/dns-query?'+urlencode({'name':name,'type':kind})
        with opener.open(Request(url,headers={'Accept':'application/dns-json'}),timeout=5) as response:
            raw=response.read(65537)
        if len(raw)>65536:raise ValueError('DNS response too large')
        data=json.loads(raw)
        if data.get('Status')!=0:raise ValueError('DNS name could not be resolved')
        answers=[]
        for item in data.get('Answer',[]):
            if item.get('type') not in (1,28):continue
            address=ipaddress.ip_address(item['data'])
            answers.append((socket.AF_INET6 if address.version==6 else socket.AF_INET,
                            socket.SOCK_STREAM,0,'',(str(address),port)))
        if answers:return answers
    raise ValueError('DNS name has no address')


def _direct_connect(tunnel):
    # Destination has already been converted to a validated public numeric IP.
    authority=tunnel.split(b' ',2)[1].decode('ascii')
    parsed=urlsplit('//'+authority)
    address=ipaddress.ip_address(parsed.hostname)
    if not address.is_global:raise PermissionError('non-public direct destination')
    return socket.create_connection((str(address),parsed.port),timeout=10)


@contextmanager
def host_proxy_socket(task_dir, upstream_port, *, socket_name='network-proxy.sock', max_connections=16,
                      resolver=None, direct=False):
    """Expose only the configured local proxy through an owned private socket."""
    owned_fd = None
    if type(task_dir) is int:
        owned_fd = os.dup(task_dir)
        root = Path('/proc/self/fd') / str(owned_fd)
        info = os.fstat(owned_fd)
    else:
        root = Path(task_dir)
        info = root.lstat()
    if (not root.is_absolute() or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
        if owned_fd is not None:os.close(owned_fd)
        raise PermissionError('task proxy directory must be owned and private')
    if not direct and (type(upstream_port) is not int or not 1 <= upstream_port <= 65535):
        if owned_fd is not None:os.close(owned_fd)
        raise ValueError('invalid upstream proxy port')
    if (not isinstance(socket_name,str) or not socket_name.startswith('network-proxy')
            or '/' in socket_name or len(socket_name)>70):
        if owned_fd is not None:os.close(owned_fd)
        raise ValueError('invalid private proxy socket name')
    path = root / socket_name
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if resolver is None:resolver=resolve_public_doh if direct else socket.getaddrinfo
    try:
        listener.bind(str(path))  # fails if a file already exists; no overwrite
        os.chmod(path, 0o600)
        identity = path.lstat()
        listener.listen(max_connections)
        connect=_direct_connect if direct else lambda:socket.create_connection(('127.0.0.1',upstream_port),timeout=10)
        bridge = _Bridge(listener,connect,max_connections,prepare=lambda head:_prepare_proxy_request(head,resolver),direct=direct)
        try: yield path
        finally:
            bridge.close()
            # Only unlink the exact socket this context created.
            try:
                current = path.lstat()
                if stat.S_ISSOCK(current.st_mode) and (current.st_dev,current.st_ino)==(identity.st_dev,identity.st_ino):
                    path.unlink()
            except FileNotFoundError: pass
    finally:
        listener.close()
        if owned_fd is not None:os.close(owned_fd)


@contextmanager
def namespace_proxy_socket(unix_path, *, max_connections=16):
    """Bridge a private namespace's loopback to its mounted task socket."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    listener.listen(max_connections)
    def connect():
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(10); conn.connect(str(unix_path)); return conn
    bridge = _Bridge(listener, connect, max_connections)
    try: yield listener.getsockname()[1]
    finally: bridge.close()
