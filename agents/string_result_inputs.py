"""Expose only native producer-declared String outputs as original text bytes."""
import json
import math
from pathlib import Path
import re

PROTOCOL='tutrast_string_seven_columns_v1'
COLUMNS=['fractional_a','fractional_b','fractional_c','orientation_alpha',
         'orientation_beta','orientation_gamma','potential']


def attach_string_inputs(root, source_contract, native_result, candidates, sealed_inputs,
                         budget, file_facts, read):
    """Validate native metadata, SHA and finite seven-column files, without units inference.

    read is the packet's owned NOFOLLOW, regular-file, 2MiB budgeted reader.
    Metadata labels describe the adapter protocol; they do not establish physical
    units, source semantics, scientific convergence or binary/source binding.
    """
    if source_contract.get('tool')!='run_string_tst':return {}
    manifest=native_result.get('result_json_path')
    if not isinstance(manifest,str)or manifest not in candidates:
        raise ValueError('String result manifest is not a native declared artifact')
    output_dir=Path((source_contract.get('arguments')or{}).get('output_dir',''))
    if not output_dir.is_absolute()or not output_dir.is_relative_to(root):
        raise ValueError('String output branch must be owned')
    if not Path(manifest).is_relative_to(output_dir):raise ValueError('String manifest outside output branch')
    raw,fact=read(root,manifest,budget);file_facts.append(fact)
    value=json.loads(raw);protocol=value.get('output_protocol')or{}
    if (value.get('schema_version')!=2 or value.get('success')is not True
            or protocol.get('id')!=PROTOCOL or protocol.get('column_count')!=7
            or protocol.get('columns')!=COLUMNS):
        raise ValueError('unsupported native String output protocol')
    results=value.get('results')
    if not isinstance(results,list)or not 1<=len(results)<=16:
        raise ValueError('native String result count invalid')
    result={}
    for row in results:
        if not isinstance(row,dict):raise ValueError('native String row invalid')
        path=row.get('output_path');expected=row.get('output_sha256')
        if (not isinstance(path,str)or not Path(path).is_absolute()
                or not Path(path).is_relative_to(output_dir)
                or Path(path).name!='string_path.dat'
                or not re.fullmatch('[a-f0-9]{64}',str(expected))):
            raise ValueError('invalid declared native String path')
        count=row.get('point_count')
        if type(count)is not int or not 2<=count<=2000 or row.get('expected_point_count')!=count:
            raise ValueError('native String point count mismatch')
        data,source_fact=read(root,path,budget);file_facts.append(source_fact)
        if source_fact['sha256']!=expected:raise ValueError('native String path hash changed')
        text=data.decode('utf-8-sig')
        if '\x00'in text:raise ValueError('native String path contains NUL')
        lines=[line.split()for line in text.splitlines()if line.strip()]
        if len(lines)!=count or any(len(line)!=7 for line in lines):
            raise ValueError('native String path shape differs from declared protocol')
        if any(not math.isfinite(float(cell))for line in lines for cell in line):
            raise ValueError('native String path contains nonfinite values')
        if path in result:raise ValueError('duplicate native String output')
        candidates.append(path);sealed_inputs[path]=expected
        result[path]={'format':PROTOCOL,'columns':COLUMNS,'point_count':count,
            'source_sha256':expected,'manifest_sha256':fact['sha256'],
            'energy_unit':protocol.get('energy_unit'),
            'protocol_provenance':protocol.get('provenance')or{},
            'scientific_convergence':'unknown','scientific_verification':False,
            'scope':'Original finite seven-column text. Protocol labels are adapter-declared; missing units remain unknown. Shape and hash checks do not prove physical semantics or convergence.'}
    return result
