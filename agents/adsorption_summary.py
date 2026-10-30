"""Owned native GCMC summaries; cDFT references remain separate methods/units.

These are observations, not scheduler, convergence, retry or finish authority.
"""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import uuid

from .workload_budget import _owned_fd


def _identity(s):
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def _read(root,path,facts,budget,limit):
    path=Path(os.path.abspath(path));fd=_owned_fd(root,path)
    try:
        a=os.fstat(fd)
        if not stat.S_ISREG(a.st_mode) or not 0<a.st_size<=min(limit,budget[0]):
            raise ValueError('bounded nonempty native evidence required')
        raw=bytearray()
        while len(raw)<=a.st_size:
            chunk=os.read(fd,min(65536,a.st_size+1-len(raw)))
            if not chunk:break
            raw.extend(chunk)
        b=os.fstat(fd);c=os.stat(path,follow_symlinks=False)
        if len(raw)!=a.st_size or _identity(a)!=_identity(b) or _identity(b)!=_identity(c):
            raise ValueError('native evidence changed during reading')
        digest=hashlib.sha256(raw).hexdigest();old=facts.get(str(path))
        if old and old['sha256']!=digest:raise ValueError('native evidence changed')
        facts[str(path)]={'sha256':digest,'bytes':len(raw)};budget[0]-=len(raw)
        return bytes(raw).decode('utf-8')
    finally:os.close(fd)


def _entries(root,path,directories=False):
    fd=_owned_fd(root,path,directory=True)
    try:
        values=[]
        with os.scandir(fd)as scan:
            for i,entry in enumerate(scan):
                if i>=4096:raise ValueError('native directory entry limit exceeded')
                if entry.is_symlink():raise ValueError('symlink is not native evidence')
                if entry.is_dir(follow_symlinks=False)==directories:
                    values.append(Path(path)/entry.name)
        return sorted(values)
    finally:os.close(fd)


def _native_case_paths(root, case):
    """Select one owned native layout without flattening pressure directories.

    Legacy direct cases and the native single-pressure child layout are
    supported. Multiple pressure points require an explicit isotherm analysis;
    silently choosing one would alter the scientific comparison.
    """
    files = {path.name for path in _entries(root, case)}
    directories = _entries(root, case, True)
    names = {path.name for path in directories}
    pressures = [path for path in directories if path.name.startswith('P_')]
    direct = 'simulation.input' in files or 'Output' in names
    if direct:
        if pressures or 'simulation.input' not in files or 'Output' not in names:
            raise ValueError('ambiguous or incomplete direct native case layout')
        native = case
    else:
        if len(pressures) != 1:
            raise ValueError('exactly one native pressure directory required per material/gas')
        native = pressures[0]
        if not re.fullmatch(r'P_[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?', native.name):
            raise ValueError('noncanonical native pressure directory')
        _positive(native.name[2:])
        child_files = {path.name for path in _entries(root, native)}
        child_directories = {path.name for path in _entries(root, native, True)}
        if 'simulation.input' not in child_files or 'Output' not in child_directories:
            raise ValueError('incomplete native pressure case layout')
        if any(name.startswith('P_') for name in child_directories):
            raise ValueError('nested pressure directories are not a single native case')
    output = native / 'Output' / 'System_0'
    data = [path for path in _entries(root, output) if path.suffix == '.data']
    if len(data) != 1:
        raise ValueError('exactly one native .data per case required')
    return native / 'simulation.input', data[0]


def _field(text,key):
    values=re.findall(r'^\s*'+re.escape(key)+r'\s+([^\r\n#]+)',text,re.M)
    if len(values)!=1:raise ValueError('native input must declare exactly one '+key)
    return values[0].strip()


def _positive(value):
    x=float(value)
    if not math.isfinite(x)or x<=0:raise ValueError('invalid native positive quantity')
    return x


def _safe_parent(root,parent):
    if not parent.is_relative_to(root):raise ValueError('summary output outside session')
    fd=_owned_fd(root,root,directory=True)
    try:
        for part in parent.relative_to(root).parts:
            try:os.mkdir(part,mode=0o700,dir_fd=fd)
            except FileExistsError:pass
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child
        result=fd;fd=None;return result
    finally:
        if fd is not None:os.close(fd)


def _publish(path,raw,root):
    path=Path(os.path.abspath(path))
    if not path.is_relative_to(root):raise ValueError('summary output outside session')
    parent=_safe_parent(root,path.parent)
    try:
        try:
            fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        except FileNotFoundError:fd=None
        if fd is not None:
            try:
                info=os.fstat(fd)
                if not stat.S_ISREG(info.st_mode)or info.st_size!=len(raw)or os.read(fd,len(raw)+1)!=raw:
                    raise ValueError('existing different summary preserved; choose a new path')
                return
            finally:os.close(fd)
        temporary='.adsorption-summary-'+uuid.uuid4().hex
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
        try:
            with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            os.link(temporary,path.name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
            os.fsync(parent)
        finally:os.unlink(temporary,dir_fd=parent)
    finally:os.close(parent)


def collect_adsorption_summary(work_dir,gases,output_csv,output_markdown,*,owned_root,reference_cdft_csvs=None,ranking_policy="separate_methods",cif_dir=None,top_n=None,top_output_dir=None):
    if ranking_policy not in {'separate_methods','within_method_ratio_combined'}:
        raise ValueError('unsupported explicit ranking policy')
    root=Path(os.path.abspath(owned_root));work=Path(os.path.abspath(work_dir));facts={};budget=[256*1024*1024]
    if Path(os.path.abspath(output_csv))==Path(os.path.abspath(output_markdown)):
        raise ValueError('CSV and Markdown require distinct output paths')
    if root.resolve(strict=True)!=root or not work.is_relative_to(root):raise ValueError('unowned result tree')
    if (not isinstance(gases,list)or not 1<=len(gases)<=16 or len(set(gases))!=len(gases)
            or any(not isinstance(g,str)or not re.fullmatch(r'[A-Za-z0-9_+-]{1,32}',g)for g in gases)):
        raise ValueError('one or more unique gas identities required')
    values={};native={};materials=set()
    from .slurm import _gas_preset
    for gas in gases:
        count=0
        for temperature_dir in _entries(root,work/gas,True):
            if not re.fullmatch(r'[0-9.eE+-]+K',temperature_dir.name):raise ValueError('noncanonical temperature directory')
            for case in _entries(root,temperature_dir,True):
                count+=1
                if count>512:raise ValueError('native case inventory exceeds bound')
                material=case.name
                if not 0<len(material)<=256 or any(ord(c)<32 for c in material):raise ValueError('invalid material identity')
                key=(material,gas)
                if key in values:raise ValueError('ambiguous duplicate material/gas result')
                native_input,native_data=_native_case_paths(root,case)
                text=_read(root,native_input,facts,budget,1024*1024)
                molecule=re.findall(r'^\s*Component\s+0\s+MoleculeName\s+(\S+)',text,re.M)
                if molecule!=[_gas_preset(gas)['molecule_name']]or _field(text,'FrameworkName')!=material:
                    raise ValueError('native gas/framework identity mismatch')
                cycles=_field(text,'NumberOfCycles');init=_field(text,'NumberOfInitializationCycles')
                if not cycles.isdigit()or not init.isdigit()or int(cycles)<1:raise ValueError('invalid native cycle counts')
                meta={'production_cycles':int(cycles),'initialization_cycles':int(init),
                    'temperature_K':_positive(_field(text,'ExternalTemperature')),
                    'pressure_Pa':_positive(_field(text,'ExternalPressure')),
                    'force_field':_field(text,'Forcefield'),'guest_definition':_field(text,'MoleculeDefinition')}
                text=_read(root,native_data,facts,budget,128*1024*1024);lower=text.lower()
                if not any(s in lower for s in ['simulation finished','simulation completed','normal termination']):
                    raise ValueError('native result is not complete')
                if any(s in lower for s in ['simulation failed','fatal error','convergence error']):raise ValueError('native result records failure')
                matches=re.findall(r'Average loading absolute\s+\[mol/kg framework\]\s+(\S+)\s+\+/-\s+(\S+)',text,re.I)
                if len(matches)!=1:raise ValueError('exactly one absolute mol/kg loading required')
                loading,sd=map(float,matches[0])
                if not all(math.isfinite(x)and x>=0 for x in [loading,sd]):raise ValueError('nonfinite/negative native loading or uncertainty')
                meta.update(data_path=str(native_data),data_sha256=facts[str(native_data)]['sha256'])
                values[key]=(loading,sd);native[key]=meta;materials.add(material)
        if count==0:raise ValueError('missing native gas cases')
    if any((m,g)not in values for m in materials for g in gases):raise ValueError('missing gas/material result')
    if len(gases)>=2:
        for material in materials:
            conditions={(native[(material,g)]['temperature_K'],native[(material,g)]['pressure_Pa'])for g in gases}
            if len(conditions)!=1:raise ValueError('uptake ratio requires same native temperature and pressure')
    gcmc_materials=set(materials)
    reference={};reference_gases=[];reference_inventory={};reference_sources={}
    if reference_cdft_csvs:
        if not isinstance(reference_cdft_csvs,dict)or len(reference_cdft_csvs)>16:raise ValueError('invalid cDFT reference mapping')
        for gas,path in reference_cdft_csvs.items():
            if not re.fullmatch(r'[A-Za-z0-9_+-]{1,32}',gas):raise ValueError('invalid reference gas')
            text=_read(root,path,facts,budget,16*1024*1024);reader=csv.DictReader(io.StringIO(text));column=gas+'_density_mol_L'
            if (not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames)
                    or 'MOF'not in reader.fieldnames or column not in reader.fieldnames):raise ValueError('canonical cDFT material/density columns required')
            reference_gases.append(gas);seen=set()
            reference_sources[gas]={'path':str(Path(os.path.abspath(path))),'sha256':facts[str(Path(os.path.abspath(path)))]['sha256']}
            for i,row in enumerate(reader):
                if i>=10000:raise ValueError('reference row bound exceeded')
                material=row.get('MOF','')
                if (not material or material in seen or len(material)>256 or Path(material).name!=material
                        or material in {'.','..'} or any(ord(c)<32 for c in material)):
                    raise ValueError('duplicate/missing/unsafe reference material')
                seen.add(material);value=row.get(column)
                value=None if value in (None,'')else float(value)
                if value is not None and(not math.isfinite(value)or value<0):raise ValueError('invalid cDFT density')
                reference[(material,gas)]=value;materials.add(material)
            reference_inventory[gas]=seen
    if ranking_policy=='within_method_ratio_combined':
        if len(gases)!=2 or set(reference_gases)!=set(gases):
            raise ValueError('combined ranking requires two gases and their canonical cDFT references')
        inventories=list(reference_inventory.values())
        if not inventories or any(v!=inventories[0]for v in inventories) or not gcmc_materials.issubset(inventories[0]):
            raise ValueError('combined ranking requires exact reference material inventories and a GCMC subset')
    rows=[]
    for material in sorted(materials):
        has_gcmc=all((material,g)in values for g in gases)
        row={'material':material,'method':'GCMC'if has_gcmc else'cDFT_reference',
             'loading_unit':'mol/kg_framework'if has_gcmc else None,'scientific_convergence':'unknown'}
        if reference_gases:
            row.update(reference_method='cDFT',cross_method_comparability='different_models_and_units_not_interchangeable')
        for gas in gases:
            pair=values.get((material,gas));meta=native.get((material,gas),{})
            row.update({gas+'_loading_mol_kg':pair[0]if pair else None,gas+'_std_mol_kg':pair[1]if pair else None,
                gas+'_statistical_convergence':'unknown'if pair else'not_available'})
            for key in ['production_cycles','initialization_cycles','temperature_K','pressure_Pa','force_field','guest_definition','data_path','data_sha256']:
                row[gas+'_'+key]=meta.get(key)
        if len(gases)>=2:
            num,den=gases[:2];a=values.get((material,num));b=values.get((material,den));ratio=None if not a or not b or b[0]<=0 else a[0]/b[0]
            row.update({num+'_'+den+'_uptake_ratio':ratio,'ratio_definition':f'single-component uptake ratio {num}/{den}; not IAST selectivity','zero_or_negative_denominator':bool(b and b[0]<=0)})
        for gas in reference_gases:
            value=reference.get((material,gas));row['cdft_'+gas+'_density_mol_L']=value
            row['cdft_'+gas+'_density_status']='missing'if value is None else'available_unverified_convergence'
            row['cdft_'+gas+'_source_csv']=reference_sources[gas]['path']
            row['cdft_'+gas+'_source_sha256']=reference_sources[gas]['sha256']
        rows.append(row)
    if ranking_policy=='within_method_ratio_combined':
        num,den=gases
        for row in rows:
            material=row['material']
            if all((material,g)in values for g in gases):
                numerator,denominator=values[(material,num)][0],values[(material,den)][0]
                method='GCMC';pair={g:{'path':native[(material,g)]['data_path'],'sha256':native[(material,g)]['data_sha256']}for g in gases}
            else:
                numerator,denominator=reference.get((material,num)),reference.get((material,den))
                if numerator is None or denominator is None:
                    raise ValueError('material has no complete within-method gas pair for combined ranking')
                method='cDFT';pair={g:reference_sources[g]for g in gases}
            ratio=None if denominator<=0 else numerator/denominator
            if ratio is not None and not math.isfinite(ratio):raise ValueError('nonfinite combined uptake ratio')
            core={'material':material,'method':method,'gases':gases,'sources':pair}
            row.update(joint_uptake_ratio=ratio,ranking_method=method,
                ratio_definition=f'single-component uptake ratio {num}/{den}; not IAST selectivity',
                ratio_comparison_limit='different Hamiltonians; method-labelled exploratory comparison',
                source_pair_sha256=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest(),
                source_pair_json=json.dumps(core,sort_keys=True),
                ranking_status='unrankable_zero_denominator'if ratio is None else'ranked',joint_rank=None)
        rows.sort(key=lambda row:(row['joint_uptake_ratio']is not None,row['joint_uptake_ratio']if row['joint_uptake_ratio']is not None else float('-inf')),reverse=True)
        for rank,row in enumerate(rows,1):
            if row['joint_uptake_ratio']is not None:row['joint_rank']=rank
    if not reference_gases:
        if len(gases)>=2:
            key=gases[0]+'_'+gases[1]+'_uptake_ratio';rows.sort(key=lambda row:(row[key]is not None,row[key]or float('-inf')),reverse=True)
        else:rows.sort(key=lambda row:row[gases[0]+'_loading_mol_kg'],reverse=True)
    staged=[];top_files=[];top_manifest_path=None
    if any(v is not None for v in (cif_dir,top_n,top_output_dir)):
        if not all(v is not None for v in (cif_dir,top_n,top_output_dir)) or type(top_n)is not int or not 1<=top_n<=512:
            raise ValueError('TOP staging requires cif_dir, positive top_n, and top_output_dir together')
        if reference_gases and ranking_policy!='within_method_ratio_combined':
            raise ValueError('reference-based TOP staging requires explicit combined ranking')
        source_dir=Path(os.path.abspath(cif_dir));top_dir=Path(os.path.abspath(top_output_dir))
        if (not top_dir.is_relative_to(root) or top_dir.is_relative_to(source_dir)
                or source_dir.is_relative_to(top_dir) or top_dir.is_relative_to(work)
                or any(Path(os.path.abspath(v)).is_relative_to(top_dir)for v in (output_csv,output_markdown))):
            raise ValueError('TOP directory must be new, separate and session-owned')
        ranked=[row for row in rows if (row.get('joint_rank')is not None if ranking_policy=='within_method_ratio_combined'
                else len(gases)==1 or row.get(gases[0]+'_'+gases[1]+'_uptake_ratio')is not None)]
        if len(ranked)<top_n:raise ValueError('insufficient rankable materials for declared TOP subset')
        for rank,row in enumerate(ranked[:top_n],1):
            source=source_dir/(row['material']+'.cif');raw=_read(root,source,facts,budget,16*1024*1024).encode()
            name=source.name;staged.append((top_dir/name,raw))
            top_files.append({'name':name,'source':str(source),'sha256':hashlib.sha256(raw).hexdigest(),
                'rank':rank,'ranking_method':row.get('ranking_method','GCMC'),
                'uptake_ratio':row.get('joint_uptake_ratio',row.get(gases[0]+'_'+gases[1]+'_uptake_ratio')if len(gases)>=2 else None)})
        top_manifest_path=top_dir/'subset_manifest.json'
        if top_dir.exists():
            existing={p.name for p in _entries(root,top_dir)}
            expected={p.name for p,_ in staged}|{'subset_manifest.json'}
            if existing!=expected or _entries(root,top_dir,True):
                raise ValueError('existing TOP directory has different or partial contents; choose a new path')
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);csv_raw=buffer.getvalue().encode()
    lines=['# Native adsorption observations','',
        'GCMC absolute loading: mol/kg framework. Statistical convergence: unknown.',
        'Production and initialization cycles are recorded separately; short runs do not establish equilibration.',
        'cDFT mol/L and GCMC mol/kg are not interchangeable; no mixed-method ranking or automatic unit conversion is performed.']
    if len(gases)>=2:lines+=['Single-component uptake ratio; not IAST selectivity.']
    if ranking_policy=='within_method_ratio_combined':
        lines+=['Explicit exploratory combined ranking uses only each material within its own method gas pair. cDFT and GCMC use different Hamiltonians; this is not a matched-model validation or a mixture selectivity. Original densities and loadings remain separate, without filling missing cDFT values.']
    lines+=['','| Material | Method | Native observations |','|---|---|---|']
    for row in rows:
        observed='; '.join(f'{g}: {row[g+"_loading_mol_kg"]} mol/kg; production={row[g+"_production_cycles"]}; initialization={row[g+"_initialization_cycles"]}'for g in gases if row[g+'_loading_mol_kg']is not None)
        if reference_gases:
            observed+='; '+'; '.join(f'cDFT {gas}: {row["cdft_"+gas+"_density_mol_L"]} mol/L ({row["cdft_"+gas+"_density_status"]})' for gas in reference_gases)
        if ranking_policy=='within_method_ratio_combined':
            observed+='; joint rank='+str(row['joint_rank'])+'; within-'+row['ranking_method']+' ratio='+str(row['joint_uptake_ratio'])
        lines.append('| '+row['material'].replace('|','\\|')+' | '+row['method']+' | '+observed+' |')
    md_raw=('\n'.join(lines)+'\n').encode()
    if top_manifest_path is not None:
        manifest={'kind':'ranked_adsorption_subset','selection':'highest_within_method_ratio'if ranking_policy=='within_method_ratio_combined'else'GCMC_ranking',
            'files':top_files,'ranking_policy':ranking_policy,'source_summary_sha256':hashlib.sha256(csv_raw).hexdigest(),
            'source_snapshot_sha256':hashlib.sha256(json.dumps(facts,sort_keys=True).encode()).hexdigest(),
            'comparison_limit':'different method Hamiltonians; exploratory within-method ratios, not mixture selectivity'}
        staged.append((top_manifest_path,(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()))
    # Rehash sources after rendering. No stat-only authorization or cached content.
    fresh={};fresh_budget=[256*1024*1024]
    for path,old in facts.items():
        _read(root,path,fresh,fresh_budget,128*1024*1024)
        if fresh[path]!=old:raise ValueError('source changed before summary publication')
    publications=[(output_csv,csv_raw),(output_markdown,md_raw),*staged]
    for path,raw in publications:
        target=Path(os.path.abspath(path))
        if not target.is_relative_to(root):raise ValueError('summary output outside session')
        if str(target)in facts:raise ValueError('summary cannot replace a source')
        if target.exists()or target.is_symlink():
            prior={};text=_read(root,target,prior,[16*1024*1024],16*1024*1024)
            if text.encode()!=raw:raise ValueError('existing different summary preserved; choose new output paths')
    for path,raw in publications:_publish(path,raw,root)
    output_facts={};output_budget=[32*1024*1024]
    for path,raw in publications:
        if _read(root,path,output_facts,output_budget,16*1024*1024).encode()!=raw:
            raise ValueError('summary changed during publication')
    declared_gases=set(gases)|set(reference_gases)
    coverage=all((m,g)in values or reference.get((m,g))is not None for m in materials for g in declared_gases)
    return {'ok':True,'status':'success','rows':rows,'row_count':len(rows),'output_csv':str(output_csv),
        'output_markdown':str(output_markdown),'top_cif_dir':str(top_manifest_path.parent)if top_manifest_path else None,
        'output_dir':str(top_manifest_path.parent)if top_manifest_path else None,
        'top_subset_manifest':str(top_manifest_path)if top_manifest_path else None,'top_materials':top_files,'source_files':sorted(facts),'source_facts':facts,'output_facts':output_facts,
        'source_snapshot_sha256':hashlib.sha256(json.dumps(facts,sort_keys=True).encode()).hexdigest(),
        'synthetic_data_used':False,'mixed_method_ranking_performed':ranking_policy=='within_method_ratio_combined',
        'ranking_policy':ranking_policy,'ranked_material_count':sum(row.get('joint_uptake_ratio')is not None for row in rows)if ranking_policy=='within_method_ratio_combined'else None,
        'comparison_limits':['Different method Hamiltonians and unknown convergence; not matched-model validation or mixture/IAST selectivity.'],
        'material_gas_observation_coverage_complete':coverage,
        'observation_coverage_is_not_convergence':True,
        'scientific_convergence':None,'execution_authority':False}


def summary_dependency_identity(arguments,project_root):
    """Hash all selected native input/result members and reference CSV bytes.

    Missing, linked or over-bound sources return no cache proof at the caller.
    This identity does not establish native completion or execution authority.
    """
    project=Path(os.path.abspath(project_root))
    def absolute(value):
        path=Path(value).expanduser()
        return Path(os.path.abspath(path if path.is_absolute()else project/path))
    work=absolute(arguments['work_dir']);relative=work.relative_to(project/'runs')
    if len(relative.parts)<3:raise ValueError('analysis cache needs one owned conversation result subtree')
    root=project/'runs'/relative.parts[0]/relative.parts[1]
    if root.resolve(strict=True)!=root:raise ValueError('analysis cache root aliases another path')
    gases=arguments.get('gases')
    if not isinstance(gases,list)or not 1<=len(gases)<=16 or any(not isinstance(g,str)or not re.fullmatch(r'[A-Za-z0-9_+-]{1,32}',g)for g in gases):raise ValueError('invalid cache gas inventory')
    facts={};budget=[256*1024*1024];count=0
    for gas in gases:
        for temperature in _entries(root,work/gas,True):
            for case in _entries(root,temperature,True):
                count+=1
                if count>512:raise ValueError('analysis cache inventory exceeds bound')
                native_input,native_data=_native_case_paths(root,case)
                _read(root,native_input,facts,budget,1024*1024)
                _read(root,native_data,facts,budget,128*1024*1024)
    if not count:raise ValueError('analysis cache has no native cases')
    references=arguments.get('reference_cdft_csvs')or {}
    if not isinstance(references,dict):raise ValueError('invalid cache reference mapping')
    for path in references.values():_read(root,absolute(path),facts,budget,16*1024*1024)
    if arguments.get('cif_dir'):
        source=absolute(arguments['cif_dir']);cifs=[p for p in _entries(root,source)if p.suffix=='.cif']
        if not cifs or len(cifs)>512:raise ValueError('analysis cache CIF inventory exceeds bound')
        for path in cifs:_read(root,path,facts,budget,16*1024*1024)
    from .slurm import _gas_preset
    return {'owner_root':str(root),'sources':facts,'parameters':arguments,
            'gas_aliases':{gas:_gas_preset(gas)for gas in gases}}
