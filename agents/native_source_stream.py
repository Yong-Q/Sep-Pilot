"""Bounded NOFOLLOW streaming hashes for native source verification.

Callers choose fixed host-side bounds. This does not load original bytes into
LLM/sandbox inputs or grant permissions outside the supplied owned root.
"""
import hashlib,os,stat

def stream_owned_source(root,path,budget,*,max_file_bytes,max_total_bytes,max_files=2048,verification):
    from .report_authoring import _relative,_open_parent
    relative=_relative(root,path)
    if budget.get('source_files',0)>=max_files:
        raise ValueError('original document verification count exceeds bound')
    parent,name=_open_parent(root,relative);fd=None
    try:
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        before=os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode)or before.st_uid!=os.getuid()
                or not 0<before.st_size<=max_file_bytes):
            raise ValueError('original document is not bounded owned regular data')
        if budget.get('source_bytes',0)+before.st_size>max_total_bytes:
            raise ValueError('original document verification byte budget exceeded')
        digest=hashlib.sha256();size=0;prefix=b''
        while True:
            block=os.read(fd,min(65536,max_file_bytes+1-size))
            if not block:break
            size+=len(block)
            if size>max_file_bytes:raise ValueError('original document grew beyond byte limit')
            if len(prefix)<1024:prefix+=block[:1024-len(prefix)]
            digest.update(block)
        after=os.fstat(fd);linked=os.stat(name,dir_fd=parent,follow_symlinks=False)
        identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode)
        if size!=before.st_size or identity(before)!=identity(after)or identity(after)!=identity(linked):
            raise ValueError('original document changed while hashing')
        fact={'path':str(relative),'size':size,'sha256':digest.hexdigest(),
            'device':before.st_dev,'inode':before.st_ino,'mtime_ns':before.st_mtime_ns,
            'ctime_ns':before.st_ctime_ns,'verification':verification}
        budget['source_files']=budget.get('source_files',0)+1
        budget['source_bytes']=budget.get('source_bytes',0)+size
        return prefix,fact
    finally:
        if fd is not None:os.close(fd)
        os.close(parent)
