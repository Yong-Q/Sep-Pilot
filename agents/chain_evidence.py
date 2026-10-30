"""Immutable, owner-scoped chain payloads; current state contains references."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid

MAX_RECORD_BYTES = 512 * 1024 * 1024


def _directory(root):
    directory = Path(root).resolve() / 'chain_evidence'
    if directory.is_symlink():
        raise PermissionError('chain evidence directory cannot be a symlink')
    return directory


def _open_directory(root, create=False):
    directory = _directory(root)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def _read_bytes(directory_fd, name, size):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ValueError('chain evidence size/type mismatch')
        return stream.read(size + 1)


def read_reference(root, owner, reference):
    digest = reference.get('sha256', '')
    if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
        raise ValueError('invalid chain evidence digest')
    if reference.get('path') != 'chain_evidence/' + digest + '.json':
        raise PermissionError('chain evidence path is outside the owned directory')
    size = reference.get('bytes')
    if type(size) is not int or not 0 <= size <= MAX_RECORD_BYTES:
        raise ValueError('invalid chain evidence size')
    directory_fd = _open_directory(root)
    try:
        raw = _read_bytes(directory_fd, digest + '.json', size)
    finally:
        os.close(directory_fd)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('chain evidence digest mismatch')
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get('owner') != owner:
        raise PermissionError('chain evidence belongs to another user/session')
    kind = reference.get('record_type')
    if kind == 'legacy_chain':
        if 'record_type' in data:
            raise ValueError('legacy chain cannot be an evidence envelope')
        return data
    if data.get('record_type') != kind or 'value' not in data:
        raise ValueError('chain evidence record type mismatch')
    return data['value']


def publish(root, owner, kind, value=None, *, raw=None):
    if raw is None:
        raw = json.dumps({'schema_version': 1, 'owner': owner,
                          'record_type': kind, 'value': value},
                         ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError('chain evidence exceeds maximum record size')
    digest = hashlib.sha256(raw).hexdigest()
    reference = {'path': 'chain_evidence/' + digest + '.json', 'sha256': digest,
                 'bytes': len(raw), 'record_type': kind}
    directory_fd = _open_directory(root, create=True)
    target = digest + '.json'
    name = '.publish-' + uuid.uuid4().hex
    temporary_created = False
    try:
        try:
            existing = _read_bytes(directory_fd, target, len(raw))
        except FileNotFoundError:
            pass
        else:
            if existing != raw:
                raise ValueError('chain evidence content mismatch')
            return reference
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory_fd)
        temporary_created = True
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, target, src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                    follow_symlinks=False)
        except FileExistsError:
            if _read_bytes(directory_fd, target, len(raw)) != raw:
                raise ValueError('chain evidence content mismatch')
        os.fsync(directory_fd)
    finally:
        if temporary_created:
            os.unlink(name, dir_fd=directory_fd)
        os.close(directory_fd)
    return reference
