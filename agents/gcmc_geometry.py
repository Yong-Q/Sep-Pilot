"""Source-bound triclinic minimum-image geometry; no scientific model edits.

Automatic multiplication is the component-wise minimum satisfying 2*cutoff to
explicit roundoff tolerance. Explicit insufficient triples are retained and
flagged for owner review, not silently changed or called scientific failure.
"""
import io
import math
import os
import re
from pathlib import Path
import warnings

from .raspa_assets import _read

_KEYS=('_cell_length_a','_cell_length_b','_cell_length_c',
       '_cell_angle_alpha','_cell_angle_beta','_cell_angle_gamma')
MAX_REPLICATION=1_000_000


def _cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def _norm(a):return math.hypot(*a)


def _triple(value):
    if isinstance(value,str):
        parts=value.split()
        if len(parts)!=3 or any(not p.isascii() or not p.isdigit() for p in parts):raise ValueError('three positive integer multipliers required')
        result=[int(p) for p in parts]
    elif isinstance(value,(list,tuple)) and len(value)==3 and all(type(p)is int for p in value):result=list(value)
    else:raise ValueError('three positive integer multipliers required')
    if any(not 1<=n<=MAX_REPLICATION for n in result):raise ValueError('multipliers outside bounded supported range')
    return result


def derive_gcmc_geometry(cif_path,cutoff,unit_cells=None,*,owned_root=None):
    """Return JSON facts with explicit units, source SHA and no execution grant.

    Input CIF bytes are bounded, regular, stable and NOFOLLOW (all ancestors).
    ASE reads CIF syntax/uncertainties, but does not expand atoms/change angles.
    Every one of six cell scalars is required, with no default right angles.
    """
    base={'schema_version':1,'kind':'gcmc_geometry','ok':False,'resource_authorized':False,'scientific_acceptance':False,
          'length_unit':'angstrom','angle_unit':'degree','warning_codes':[]}
    if type(cutoff)not in (float,int) or not math.isfinite(cutoff) or not 0<cutoff<=1_000_000:
        return {**base,'reason':'invalid_cutoff'}
    mode='auto' if unit_cells is None or isinstance(unit_cells,str) and unit_cells=='auto' else 'explicit'
    try:explicit=None if mode=='auto' else _triple(unit_cells)
    except (ValueError,OverflowError):return {**base,'reason':'invalid_unit_cells'}
    try:
        from ase.io.cif import parse_cif
        path=Path(os.path.abspath(cif_path))
        if owned_root is not None:
            root=Path(os.path.abspath(owned_root))
            if not path.is_relative_to(root):raise ValueError('source outside owned scope')
        raw,source=_read(path)
        # ASE's scalar reader overwrites duplicate tags silently. Count native
        # cell declarations outside CIF semicolon text before trusting values.
        counts={k:0 for k in _KEYS};in_text=False
        for line in raw.decode('utf-8').splitlines():
            if line.startswith(';'):in_text=not in_text;continue
            if in_text:continue
            parts=line.strip().split(None,1)
            if parts and parts[0].lower() in counts:
                counts[parts[0].lower()]+=1
        if any(n>1 for n in counts.values()):raise ValueError('duplicate cell scalar')
        with warnings.catch_warnings(record=True)as caught:
            warnings.simplefilter('always')
            blocks=list(parse_cif(io.BytesIO(raw)))
        if any('duplicat' in str(w.message).lower() for w in caught):raise ValueError('duplicate CIF definitions')
        cells=[block for block in blocks if any(block.get(k)is not None for k in _KEYS)]
        if len(cells)!=1:raise ValueError('exactly one cell block required')
        values=[cells[0].get(k) for k in _KEYS]
        number=re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?(?:\([0-9]+\))?\Z')
        values=[float(v.split('(',1)[0]) if isinstance(v,str) and number.fullmatch(v) else v for v in values]
        if any(type(v)not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('six finite cell scalars required')
        a,b,c,alpha,beta,gamma=map(float,values)
        if not all(0<x<=1_000_000 for x in (a,b,c)) or not all(0<x<180 for x in (alpha,beta,gamma)):raise ValueError('invalid cell lengths or angles')
        ca,cb,cg=(math.cos(math.radians(x)) for x in (alpha,beta,gamma))
        sg=math.sin(math.radians(gamma))
        gram=1-ca*ca-cb*cb-cg*cg+2*ca*cb*cg
        if not math.isfinite(gram) or gram<=1e-12 or abs(sg)<=1e-12:raise ValueError('degenerate or numerically singular cell')
        # Standard right-handed triclinic basis: rows are lattice vectors.
        lattice=[[a,0.,0.],[b*cg,b*sg,0.],
                 [c*cb,c*(ca-cb*cg)/sg,c*math.sqrt(gram)/sg]]
        va,vb,vc=lattice
        volume=abs(sum(x*y for x,y in zip(va,_cross(vb,vc))))
        faces=[_norm(_cross(vb,vc)),_norm(_cross(va,vc)),_norm(_cross(va,vb))]
        if not math.isfinite(volume) or volume<=0 or not all(math.isfinite(x) and x>0 for x in faces):raise ValueError('nonfinite or degenerate lattice')
        heights=[volume/x for x in faces]
        target=2*float(cutoff);tolerance=max(1e-12,target*2e-12)
        minimum=[]
        for height in heights:
            n=max(1,math.ceil((target-tolerance)/height))
            while n*height+tolerance<target:n+=1
            while n>1 and (n-1)*height+tolerance>=target:n-=1
            if n>MAX_REPLICATION:raise ValueError('replication count exceeds supported range')
            minimum.append(n)
        chosen=minimum if explicit is None else explicit
        expanded=[height*n for height,n in zip(heights,chosen)]
        sufficient=all(x+tolerance>=target for x in expanded)
        native_warning=any(target>x for x in expanded)
        at_boundary=any(abs(x-target)<=tolerance for x in expanded)
        warnings_out=[]
        if not sufficient:warnings_out.append('insufficient_minimum_image')
        if math.prod(chosen)>64:warnings_out.append('large_replication_requires_resource_review')
        if any(abs(x-90)>1e-10 for x in (alpha,beta,gamma)):warnings_out.append('nonorthogonal_cell_supported')
        return {**base,'ok':True,'mode':mode,'source':source,'cif_sha256':source['sha256'],
                'source_sha256':source['sha256'],'lengths':[a,b,c],'angles':[alpha,beta,gamma],
                'lattice':lattice,'volume':volume,'volume_unit':'angstrom^3','face_heights':heights,
                'cutoff':float(cutoff),'minimum_face_height':target,'minimum_unit_cells':minimum,
                'unit_cells':chosen,'unit_cells_string':' '.join(map(str,chosen)),
                'expanded_face_heights':expanded,'geometry_sufficiency':sufficient,
                'boundary_tolerance_angstrom':tolerance,'warning_codes':warnings_out,
                'native_warning_predicted':native_warning,'at_float_boundary':at_boundary,
                'strict_example_clearance':all(x>target for x in expanded),
                'boundary_policy':'minimum >= 2*cutoff within declared floating-point tolerance; native strict warning and stricter example clearance reported separately',
                'requires_scientific_review':not sufficient,
                'geometry_scope':'lattice minimum-image geometry only; not atom-overlap, force-field or convergence verification'}
    except (ValueError,TypeError,OSError,UnicodeError,AssertionError,IndexError,ZeroDivisionError,OverflowError):
        return {**base,'reason':'invalid_cif_geometry','mode':mode}


def verify_geometry_proof(proof,cif_path,cutoff):
    """Freshly verify source bytes and every derived fact; no execution authority.

    staged_source/staged_sources are transport metadata verified separately by the native output
    validator. It cannot alter the original-source derivation or selected mode.
    Unknown fields are rejected rather than interpreted as authority.
    """
    try:
        if not isinstance(proof,dict) or proof.get('ok') is not True:return False
        if proof.get('mode')not in {'auto','explicit'}:return False
        selected=None if proof['mode']=='auto' else proof['unit_cells']
        current=derive_gcmc_geometry(cif_path,cutoff,selected)
        observed={k:v for k,v in proof.items() if k not in {'staged_source','staged_sources'}}
        import json
        return current.get('ok') is True and json.dumps(current,sort_keys=True,allow_nan=False)==json.dumps(observed,sort_keys=True,allow_nan=False)
    except (OSError,ValueError,TypeError,KeyError):return False
