"""Fail closed on unsupported native cDFT grids; no physics or file changes.

The current solver maps integer FFT/displacement indices using N/2. Its odd-N
mapping is not the principal signed-frequency convention. This guard certifies
only the even-axis support restriction, not convergence or physical validity.
"""
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import stat


class UnsupportedNativeGrid(ValueError):
    def __init__(self, axes):
        self.grid_axes = tuple(axes)
        super().__init__('CDFT_GRID_UNSUPPORTED: current native cDFT solver requires even effective axes; actual grid '
            +str(self.grid_axes)+' contains an odd axis with an unverified FFT/displacement mapping. No grid was changed; revise inputs explicitly or use a validated solver.')


def native_grid_axes(data, expected_mesh=None):
    """Read the actual native header, including six-decimal generated values."""
    try:
        if not isinstance(data, bytes) or not data or len(data)>16*1024*1024:
            raise ValueError('bounded native bytes required')
        tokens=data[:16384].decode('ascii').split()
        if len(tokens)<23 or tokens[3:6]!=['Nmaxx','Nmaxy','Nmaxz'] or tokens[9:16]!=['Lxa','Lyb','Lzc','dx','angle_a','angle_b','angle_c']:
            raise ValueError('unrecognized native grid header')
        if not all(math.isfinite(float(v)) for v in tokens[1:3]):
            raise ValueError('nonfinite native header')
        if any(not re.fullmatch(r'[+]?\d+',v) for v in tokens[6:9]):
            raise ValueError('invalid native grid caps')
        caps=list(map(int,tokens[6:9]));cell=list(map(float,tokens[16:23]))
        if (any(not 1<=v<=2**31-1 for v in caps) or not all(math.isfinite(v)for v in cell)
                or any(v<=0 for v in cell[:4])):
            raise ValueError('invalid native grid geometry')
        mesh=expected_mesh or {}
        if ('ngrid'in mesh and any(v!=mesh['ngrid']for v in caps)
                or 'dl'in mesh and cell[3]!=float(f"{mesh['dl']:.6f}")):
            raise ValueError('declared mesh differs from actual prepared native inputs')
        ratios=[v/cell[3]for v in cell[:3]]
        if any(not math.isfinite(v)or v>2**31-2 for v in ratios):
            raise ValueError('native grid count conversion exceeds safe integer bound')
        axes=tuple(min(cap,int(ratio)+1)for cap,ratio in zip(caps,ratios))
        if math.prod(axes)>2**31-1:raise ValueError('native grid exceeds safe integer bound')
    except (ValueError,TypeError,OverflowError,UnicodeError)as error:
        raise ValueError('CDFT_GRID_UNVERIFIED: cannot verify actual native grid: '+str(error))from error
    if any(axis%2 for axis in axes):
        raise UnsupportedNativeGrid(axes)
    return axes


def _reader():
    # Also callable from standalone public adapters, without importing agents.
    path=Path(__file__).with_name('workload_budget.py')
    spec=importlib.util.spec_from_file_location('_cdft_grid_workload_reader',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def inspect_grid_directory(directory, expected_mesh=None):
    """Check every exact bounded regular .dat; return observed content identities."""
    reader=_reader();path=Path(directory).absolute();fd=reader._owned_fd(Path('/'),path,directory=True)
    try:
        def names_now():
            names=[]
            with os.scandir(fd)as entries:
                for index,entry in enumerate(entries):
                    if index>=reader.MAX_ENTRIES:raise ValueError('native input directory exceeds entry bound')
                    if entry.name.endswith('.dat'):names.append(entry.name)
            return names
        before=os.fstat(fd);names=names_now()
        if not 1<=len(names)<=reader.MAX_ITEMS:raise ValueError('empty or oversized native input batch')
        items=[];total=0
        for name in sorted(names):
            fact=reader._fact(fd,name)
            total+=fact['bytes']
            if total>reader.MAX_TOTAL_BYTES:raise ValueError('native input batch exceeds aggregate byte bound')
            child=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
            try:
                initial=os.fstat(child)
                if not stat.S_ISREG(initial.st_mode):raise ValueError('regular native input required')
                chunks=[];count=0
                while True:
                    block=os.read(child,min(1024*1024,reader.MAX_FILE_BYTES+1-count))
                    if not block:break
                    chunks.append(block);count+=len(block)
                    if count>reader.MAX_FILE_BYTES:raise ValueError('native input exceeds byte bound')
                final=os.fstat(child);linked=os.stat(name,dir_fd=fd,follow_symlinks=False)
                identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                raw=b''.join(chunks)
                if (identity(initial)!=identity(final)or identity(final)!=identity(linked)
                        or len(raw)!=fact['bytes']or hashlib.sha256(raw).hexdigest()!=fact['sha256']):
                    raise ValueError('native input changed during grid verification')
            finally:os.close(child)
            items.append({**fact,'grid_axes':list(native_grid_axes(raw,expected_mesh))})
        after_names=set(names_now())
        if after_names!=set(names)or before.st_mtime_ns!=os.fstat(fd).st_mtime_ns:
            raise ValueError('native input directory changed during grid verification')
        return {'schema_version':1,'status':'supported_even_grid','n_inputs':len(items),'items':items}
    finally:os.close(fd)
