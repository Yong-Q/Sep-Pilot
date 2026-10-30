"""Public-IP-pinned HTTPS retrieval, optionally over operator CONNECT proxy.

TLS authenticates the original hostname; CONNECT targets the validated public
IP. Neither model arguments nor ambient proxy variables select a relay.
"""
import ipaddress
import json
import socket
import time
from urllib.parse import urlsplit, urlencode


def public_ip(value):
    address=ipaddress.ip_address(value)
    if not address.is_global or address.is_multicast or address.is_unspecified:
        raise ValueError('URL resolves/connects to a forbidden network address')
    return str(address)


def parse_public_url(url):
    parsed=urlsplit(url)
    if (parsed.scheme!='https'or not parsed.hostname or parsed.username or parsed.password
            or parsed.fragment or any(ord(char)<33 for char in url)):
        raise ValueError('credential-free public HTTPS URL without fragment required')
    try:port=parsed.port or 443
    except ValueError:raise ValueError('invalid HTTPS port')from None
    host=parsed.hostname.encode('idna').decode('ascii')
    return parsed,host,port


def operator_network(project_root):
    from bimem_agent import paths
    paths.load_runtime_config.cache_clear()
    config=paths.get_runtime_section('scientific_download')or{}
    if not config:return {}
    from pathlib import Path
    if not isinstance(config.get('scope_root'),str)or Path(config['scope_root']).resolve()!=Path(project_root).resolve():
        return {}
    result={}
    proxy=config.get('proxy_url')
    if proxy:
        parsed=urlsplit(proxy)
        if(parsed.scheme not in {'http','https'}or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in{'','/'}or parsed.query or parsed.fragment or not parsed.port):
            raise ValueError('operator scientific proxy must be an explicit credential-free CONNECT endpoint')
        result['proxy_url']=proxy
    resolver=config.get('dns_https')
    if resolver:
        parsed,host,port=parse_public_url(resolver.get('url',''))
        if port!=443 or parsed.query:raise ValueError('operator DNS resolver needs one HTTPS endpoint')
        result['dns_https']={'url':resolver['url'],'ip':public_ip(resolver.get('ip',''))}
    return result


def _remaining(deadline):
    value=deadline-time.monotonic()
    if value<=0:raise TimeoutError('scientific retrieval total time budget exhausted')
    return value


def open_pinned(url,address,network,deadline):
    import certifi
    import urllib3
    parsed,host,port=parse_public_url(url)
    address=public_ip(address)
    kwargs={'cert_reqs':'CERT_REQUIRED','ca_certs':certifi.where(),
            'assert_hostname':host,'server_hostname':host}
    if network.get('proxy_url'):
        manager=urllib3.ProxyManager(network['proxy_url'],num_pools=1,
            cert_reqs='CERT_REQUIRED',ca_certs=certifi.where())
        pool=manager.connection_from_host(address,port,'https',pool_kwargs=kwargs)
    else:
        manager=None
        pool=urllib3.HTTPSConnectionPool(address,port=port,**kwargs)
    host_header=('['+host+']'if':'in host else host)+((':'+str(port))if port!=443 else'')
    target=(parsed.path or'/')+('?'+parsed.query if parsed.query else'')
    try:
        response=pool.urlopen('GET',target,headers={'Host':host_header,
            'User-Agent':'SEP-Pilot-scientific-reader/1.0','Accept-Encoding':'identity'},
            preload_content=False,redirect=False,retries=False,
            timeout=urllib3.Timeout(connect=min(10,_remaining(deadline)),read=min(10,_remaining(deadline))))
        return response,manager or pool
    except BaseException:
        if manager:manager.clear()
        else:pool.close()
        raise


def close_response(response,manager):
    response.close()
    clear=getattr(manager,'clear',None)
    if callable(clear):clear()
    else:manager.close()


def resolve_public(host,network,deadline):
    try:records=socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)
    except socket.gaierror:records=[]
    addresses={item[4][0]for item in records}
    if not addresses:
        resolver=network.get('dns_https')
        if not resolver:raise ValueError('origin DNS unavailable; operator may configure a pinned public HTTPS resolver')
        query=resolver['url']+'?'+urlencode({'name':host,'type':'A'})
        response,manager=open_pinned(query,resolver['ip'],network,deadline)
        try:
            if response.status!=200:raise ValueError('operator HTTPS DNS resolver did not return HTTP 200')
            raw=response.read(65537)
            if len(raw)>65536:raise ValueError('HTTPS DNS response exceeds bound')
            data=json.loads(raw)
            if data.get('Status')!=0:raise ValueError('operator HTTPS DNS lookup has no valid result')
            addresses={item['data']for item in data.get('Answer',[])if item.get('type')in{1,28}}
        finally:close_response(response,manager)
    if not addresses:raise ValueError('origin DNS returned no addresses')
    # Reject mixed public/private answers, rather than silently picking a safe
    # answer while another endpoint might be used by a subsequent resolver.
    values=[public_ip(address)for address in addresses]
    return sorted(values,key=lambda value:(ipaddress.ip_address(value).version,value))


def open_public(url,network,deadline):
    parsed,host,port=parse_public_url(url)
    try:addresses=[public_ip(host)]
    except ValueError:addresses=resolve_public(host,network,deadline)
    # Connection targets this public IP exactly. TLS Host/SNI stay the URL host.
    response,manager=open_pinned(url,addresses[0],network,deadline)
    return response,manager,{'hostname':host,'port':port,'connect_target_public_ip':addresses[0],
        'origin_tls_verified':True,'proxy_mode':'operator_connect'if network.get('proxy_url')else'direct'}
