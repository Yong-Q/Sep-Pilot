"""Bounded, hash-verified on-demand user-provided figure/writing guidance."""
import hashlib
import json
from pathlib import Path
from .raspa_assets import _read


def bundle_root():return Path(__file__).resolve().parents[1]/'.claude/skills'


def read_result_guidance(paths=None):
    root=bundle_root()
    raw,_=_read(root/'result-guidance-manifest.json')
    manifest=json.loads(raw)
    if not paths:
        return {'ok':True,'body_loaded':False,'default_backend':'python',
                'routers':['nature-figure/SKILL.md','nature-writing/SKILL.md'],
                'manifests':['nature-figure/manifest.yaml','nature-writing/manifest.yaml'],
                'guidance_file_count':len(manifest['files']),
                'license_status':manifest['license_status'],'execution_authority':False}
    if not isinstance(paths,list) or not 1<=len(paths)<=8 or len(set(paths))!=len(paths):raise ValueError('request 1..8 unique guidance paths')
    result=[];total=0
    for name in paths:
        if not isinstance(name,str) or Path(name).is_absolute() or '..'in Path(name).parts or name not in manifest['files']:raise ValueError('guidance path is not in the project bundle')
        raw,fact=_read(root/name);digest=hashlib.sha256(raw).hexdigest();total+=len(raw)
        expected=manifest['files'][name]
        if len(raw)>65536 or total>98304 or digest!=expected['sha256'] or len(raw)!=expected['bytes']:raise ValueError('guidance changed or exceeds bounded load')
        result.append({'path':name,'sha256':digest,'bytes':len(raw),'text':raw.decode('utf-8')})
    return {'ok':True,'body_loaded':True,'files':result,'execution_authority':False}
