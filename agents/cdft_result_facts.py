"""Bounded read-only cDFT facts; scheduler success is not scientific convergence.

The caller authenticates the owned run/job receipt. This module parses files,
never submits, repairs, changes inputs, accepts results or certifies a build.
"""
import hashlib
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import stat

from .workload_context import _read_regular
from .workload_budget import _owned_fd

MAX_TOTAL_BYTES = 128 * 1024 * 1024


def _finite(value):
    value=float(value)
    if not math.isfinite(value): raise ValueError('nonfinite native number')
    return value


def _input_facts(data):
    lines=[' '.join(line.split()) for line in data.decode('ascii').splitlines()]
    def after(label): return lines[lines.index(label)+1].split()
    caps=list(map(int,after('Nmaxx Nmaxy Nmaxz')))
    cell=list(map(_finite,after('Lxa Lyb Lzc dx angle_a angle_b angle_c')))
    if len(caps)!=3 or len(cell)!=7 or any(x<=0 for x in caps+cell[:4]):raise ValueError('invalid native grid')
    lengths=cell[:3];angles=cell[4:];requested=cell[3]
    if any(not 0<a<180 for a in angles):raise ValueError('invalid cell angles')
    a,b,c=[math.radians(v) for v in angles]
    determinant=1+2*math.cos(a)*math.cos(b)*math.cos(c)-math.cos(a)**2-math.cos(b)**2-math.cos(c)**2
    if determinant<=0:raise ValueError('degenerate native cell')
    volume=math.prod(lengths)*math.sqrt(determinant)
    heights=[volume/(lengths[1]*lengths[2]*math.sin(a)),volume/(lengths[0]*lengths[2]*math.sin(b)),volume/(lengths[0]*lengths[1]*math.sin(c))]
    required=[int(length/requested)+1 for length in lengths]
    grid=[min(cap,n) for cap,n in zip(caps,required)]
    temperature,kapa,delta,torr=map(_finite,after('Temperature(K) Kapa Delta_Max Torr'))
    gases=int(after('Kind_of_gas')[0]);index=lines.index('Name Epsilon(K) Sigma(A) Bulk_Density(molec/A^3)')+1
    if not 1<=gases<=64:raise ValueError('invalid gas count')
    bulk=[list(map(_finite,lines[index+i].split())) for i in range(gases)]
    controls=list(map(_finite,after('Read_Vext EOS Cutoff deos N3_cutoff Diele_con Cutoff_inside cut_coul')))
    if len(controls)!=8:raise ValueError('invalid native solver controls')
    site_types=[]
    for i,line in enumerate(lines):
        if line=='Kind_of_solvent_atoms':
            count=int(lines[i+1])
            if not 1<=count<=4096:raise ValueError('invalid guest site count')
            site_types.append([list(map(_finite,lines[i+3+j].split())) for j in range(count)])
    if len(site_types)!=gases or any(len(v)!=4 for rows in site_types for v in rows):raise ValueError('invalid native guest model')
    molecular_sites=[]
    for i,line in enumerate(lines):
        if re.fullmatch(r'Number_of_solvent_atoms\d*',line):
            count=int(lines[i+1])
            if not 1<=count<=4096:raise ValueError('invalid molecular site count')
            molecular_sites.append([list(map(_finite,lines[i+3+j].split())) for j in range(count)])
    if len(molecular_sites)!=gases or any(len(v)!=4 for rows in molecular_sites for v in rows):raise ValueError('invalid molecular geometry')
    if temperature<=0 or any(len(row)!=4 or row[3]<0 for row in bulk):raise ValueError('invalid temperature or bulk density')
    warnings=[]
    capped=any(n>cap for n,cap in zip(required,caps))
    if capped:warnings.append('grid_cap_coarsens_requested_spacing')
    if min(heights)<=2*max(controls[2],controls[7]):warnings.append('cutoff_cell_height')
    return {'temperature_K':temperature,'kapa':kapa,'delta_max':delta,'torr':torr,
        'native_grid':grid,'requested_spacing_A':requested,'actual_axis_spacing_A':[x/n for x,n in zip(lengths,grid)],
        'grid_capped':capped,'lengths_A':lengths,'angles_deg':angles,'perpendicular_heights_A':heights,
        'volume_A3':volume,'read_vext':controls[0],'EOS':controls[1],'cutoff_A':controls[2],'cut_coul_A':controls[7],
        'gas_count':gases,'bulk_density_molecule_A3':[row[3] for row in bulk],
        'guest_site_types_id_sigma_A_epsilon_K_charge_e':site_types,
        'molecular_sites_id_xyz_A':molecular_sites,'warnings':warnings,
        'model_comparison_note':'Compare the complete actual guest/framework models, mixing rule and electrostatics; gas names do not establish the same Hamiltonian.'}


def _outputs(data,gases):
    lines=data.decode('utf-8',errors='replace').splitlines();values={};active=False
    for line in lines:
        if 'Molecule' in line and 'Average_density' in line:active=True;continue
        if not active:continue
        match=re.match(r'^\s*(\d+)\s+(\S+)\s+\((\S+)%\)',line)
        if match:
            index=int(match[1])
            if index in values:raise ValueError('duplicate native density')
            density,percentage=_finite(match[2]),_finite(match[3])
            if density<0 or not 0<=percentage<=100:raise ValueError('invalid native density or percentage')
            values[index]={'density_mol_L':density,'percentage':percentage}
    expected=set(range(len(gases)));offset=1 if set(values)==set(range(1,len(gases)+1)) else 0
    if set(values)!={i+offset for i in expected}:raise ValueError('missing or ambiguous native gas densities')
    text='\n'.join(lines);pressure=re.findall(r'Pressure of bulk:\s*(\S+)\s+atm',text)
    return {'components':{gas:values[i+offset] for i,gas in enumerate(gases)},
        'pressure_atm':_finite(pressure[-1]) if len(pressure)==1 else None,
        'density_basis':'native simulation-cell average, mol/L; not mol/kg-framework'}


def _henry(data,gases,temperature):
    records={};seen_temperature=None
    for raw in data.decode('utf-8',errors='strict').splitlines():
        if not raw.strip():continue
        if 'unit:' in raw:
            parts=raw.split()
            if len(parts)!=2 or parts[1]!='unit:mol/(L*atm)':raise ValueError('unknown Henry unit')
            seen_temperature=_finite(parts[0])
            if temperature is None or abs(seen_temperature-temperature)>1e-8:raise ValueError('Henry temperature differs from native input')
        else:
            parts=raw.split()
            if seen_temperature is None or len(parts)!=2:raise ValueError('unbound Henry component')
            index=int(parts[0]);value=_finite(parts[1])
            if index in records or value<0:raise ValueError('duplicate or negative Henry component')
            records[index]=value
    if set(records)!=set(range(len(gases))):raise ValueError('incomplete Henry components')
    return {gas:records[i] for i,gas in enumerate(gases)}


def inspect_cdft_quality(work_dir, *, owned_root, gases, expected_materials, job_receipt=None):
    """Inspect an exact batch; absent legacy process/convergence proof is unknown.

    Optional execution.json is an instrumented *future* native per-case receipt:
    exit_code, timed_out, input_sha256, stdout_sha256 and stderr_sha256. It must
    bind the actual staged input.dat, solver.log and stderr.log. A JSON claim of
    convergence alone is never used. File hashes identify observations; they do
    not authenticate the producer. The caller must bind this dossier to its
    owned executor ledger before scientific acceptance.
    """
    root=Path(owned_root);work=Path(work_dir)
    work.relative_to(root)
    if not gases or len(set(gases))!=len(gases) or not all(isinstance(x,str) and x for x in gases):raise ValueError('unique ordered gases required')
    names=list(expected_materials)
    if not names or len(names)>4096 or len(set(names))!=len(names) or any(not isinstance(n,str) or Path(n).name!=n or n in {'.','..'} for n in names):raise ValueError('exact safe material identities required')
    def listing(path):
        try:fd=_owned_fd(root,path,directory=True)
        except FileNotFoundError:return []
        try:
            entries=os.listdir(fd)
            if len(entries)>8192:raise ValueError('directory bound exceeded')
            return sorted(entries)
        finally:os.close(fd)
    root_fd=_owned_fd(Path('/'),root,directory=True)
    os.close(root_fd)
    listing(work)
    input_names=listing(work/'input');run_names=listing(work/'input_dir');budget=[0]
    report={'schema_version':1,'work_dir':str(work),'expected_materials':names,'gases':list(gases),
        'input_materials':[n[:-4] for n in input_names if n.endswith('.dat')], 'run_materials':run_names,
        'job_observation':{k:job_receipt[k] for k in ('job_id','state','status','source') if k in (job_receipt or {})},
        'job_ownership_verified':False,'producer_authentication_required':True,
        'hamiltonian_comparison_verified':False,'adsorption_heat_verified':False,'materials':{}}
    for name in names:
        item={'errors':[],'artifacts':{},'native_convergence':None,'exit_code':None,'technical_completion':None,'output_valid':False}
        report['materials'][name]=item
        def read(path,required=False,allow_empty=False):
            try:
                remaining=MAX_TOTAL_BYTES-budget[0]
                if remaining<=0:raise ValueError('batch observation exceeds byte bound')
                data=sha=None
                if allow_empty:
                    fd=_owned_fd(Path('/'),path)
                    try:
                        before=os.fstat(fd)
                        if not stat.S_ISREG(before.st_mode):raise ValueError('diagnostic stream must be regular')
                        if before.st_size==0:
                            linked=_owned_fd(Path('/'),path)
                            try:visible=os.fstat(linked)
                            finally:os.close(linked)
                            identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
                            if identity(before)!=identity(os.fstat(fd)) or identity(before)!=identity(visible):raise ValueError('empty stream changed')
                            data=b'';sha=hashlib.sha256(data).hexdigest()
                    finally:os.close(fd)
                if data is None:data,sha=_read_regular(path,limit=min(8*1024*1024,remaining))
                budget[0]+=len(data)
                item['artifacts'][str(path)]={'sha256':sha,'size':len(data)}
                return data,sha
            except FileNotFoundError:
                if required:item['errors'].append('missing '+str(path))
            except (OSError,ValueError) as error:item['errors'].append(str(path)+': '+str(error))
            return None,None
        data,input_sha=read(work/'input'/(name+'.dat'),True)
        try:
            if data is not None:
                item['input_facts']=_input_facts(data)
                if item['input_facts']['gas_count']!=len(gases):raise ValueError('input component count does not match declared gases')
        except (ValueError,IndexError,UnicodeError) as error:item['errors'].append('input parse: '+str(error))
        run=work/'input_dir'/name
        staged,staged_sha=read(run/'input.dat',True)
        if staged is not None and staged_sha!=input_sha:item['errors'].append('staged input differs from batch source')
        output,_=read(run/'output.dat',True)
        if output is not None:
            try:item['output_facts']=_outputs(output,gases);item['output_valid']=True
            except ValueError as error:item['errors'].append('output parse: '+str(error))
        henry,_=read(run/'hr.dat',True)
        if henry is not None:
            try:item['henry_mol_L_atm']=_henry(henry,gases,item.get('input_facts',{}).get('temperature_K'))
            except (ValueError,UnicodeError) as error:item['errors'].append('Henry parse: '+str(error))
        log,log_sha=read(run/'solver.log',allow_empty=True);stderr,stderr_sha=read(run/'stderr.log',allow_empty=True)
        if log is not None:
            text=log.decode('utf-8',errors='replace');statuses=re.findall(r'Termination status:\s*(-?\d+)',text)
            norms=re.findall(r'maximum norm for gradient:\s*(\S+)',text)
            try:
                if not statuses and not norms:
                    item.setdefault('warnings',[]).append('native optimizer status is not reported; convergence remains unknown')
                else:
                    if len(statuses)!=1 or len(norms)!=1:raise ValueError('ambiguous final native optimizer status')
                    code=int(statuses[0]);norm=_finite(norms[0])
                    if norm<0:raise ValueError('negative gradient norm')
                    item['optimizer_status']=code;item['maximum_gradient_norm']=norm
                    item['native_convergence']=True if code==0 else None if code==1 else False
            except ValueError as error:item['errors'].append('solver log: '+str(error))
        receipt,_=read(run/'execution.json')
        if receipt is not None:
            try:
                observed=json.loads(receipt)
                if (not isinstance(observed,dict) or staged is None or log is None or stderr is None
                        or observed.get('input_sha256')!=staged_sha or observed.get('stdout_sha256')!=log_sha
                        or observed.get('stderr_sha256')!=stderr_sha or type(observed.get('exit_code')) is not int
                        or type(observed.get('timed_out')) is not bool):raise ValueError('unbound native subprocess receipt')
                item['exit_code']=observed['exit_code'];item['timed_out']=observed['timed_out']
                item['technical_completion']=observed['exit_code']==0 and not observed['timed_out'] and item['output_valid']
            except (ValueError,TypeError) as error:item['errors'].append('execution receipt: '+str(error))
        if item['errors']:item['native_convergence']=None;item['technical_completion']=False if item['technical_completion'] is True else item['technical_completion']
    items=list(report['materials'].values())
    report['complete_material_outputs']=(set(report['input_materials'])==set(names) and set(run_names)==set(names)
        and all(x['output_valid'] and not x['errors'] for x in items))
    report['technical_completion']=False if any(x['technical_completion'] is False for x in items) else True if all(x['technical_completion'] is True for x in items) else None
    report['scientific_convergence']=False if any(x['native_convergence'] is False for x in items) else None
    report['limitations']=['No authenticated producer or build attestation is inferred from JSON/log files.',
        'Legacy solver without per-material exit and optimizer receipts remains unknown; finite outputs and scheduler completion do not prove convergence.',
        'Engine convergence does not establish grid/angular convergence, force-field validity, adsorption heat or mixture selectivity.']
    return report


def inspect_cdft_batch_quality(work_dir, *, owned_root, gases, source_input_dir=None):
    """Bind expected materials to original owned inputs, never result-directory rows.

    This observes the declared input source now, not an authenticated historical
    submission manifest. Ledger binding remains the caller's responsibility.
    Without that source, retain an explicit unknown instead of inventing counts.
    """
    unknown={'schema_version':1,'status':'unknown','work_dir':str(work_dir),
        'expected_materials_verified':False,'scientific_convergence':None,
        'technical_completion':None,'hamiltonian_comparison_verified':False,
        'producer_authentication_required':True,'warnings':[]}
    if not source_input_dir:
        return {**unknown,'warnings':['Original declared input source is required to verify the complete material set.']}
    from .workload_budget import collect_workload_facts
    try:
        root=Path(owned_root);source=Path(source_input_dir)
        if not root.is_absolute() or not source.is_absolute():raise ValueError('absolute owned source required')
        facts=collect_workload_facts('run_cdft',{'action':'submit','input_dir':str(source)},owned_root=root)
        if facts.get('status')!='ready':raise ValueError('original input source is unavailable or incomplete')
        files=facts.get('inputs') or []
        if not files:raise ValueError('original input source is empty')
        names=[Path(item['path']).stem for item in files]
        report=inspect_cdft_quality(work_dir,owned_root=root,gases=gases,expected_materials=names)
        report.update(status='observed',expected_materials_verified=True,
            expected_materials_source='owned_original_inputs_observed_at_collection',
            original_input_facts=facts,gas_identity_verified=False,
            preparation_manifest_authenticated=False)
        for item in files:
            material=report['materials'][Path(item['path']).stem]
            copied=material['artifacts'].get(str(Path(work_dir)/'input'/item['path']))
            if not copied or copied['sha256']!=item['sha256'] or copied['size']!=item['bytes']:
                material['errors'].append('copied input does not match the original declared source')
                material['native_convergence']=None
                report['complete_material_outputs']=False
        report['warnings']=sorted({warning for item in report['materials'].values()
            for warning in item.get('input_facts',{}).get('warnings',[])})
        report['collection_errors']=[]
        try:
            raw,sha=_read_regular(Path(work_dir)/'results.csv',limit=8*1024*1024)
            report['collection_errors']=collected_cdft_row_errors(
                list(csv.DictReader(io.StringIO(raw.decode('utf-8')))),report,gases)
            report['collected_csv_sha256']=sha
        except FileNotFoundError:
            pass
        except (ValueError,OSError,UnicodeError) as error:
            report['collection_errors']=['collected CSV observation: '+str(error)]
        if report['collection_errors']:report['complete_material_outputs']=False
        return report
    except (ValueError,OSError,KeyError,TypeError) as error:
        return {**unknown,'warnings':['Quality evidence unavailable: '+str(error)]}


def collected_cdft_row_errors(rows, quality, gases):
    """Compare canonical collected rows against independently parsed native files."""
    if not quality.get('expected_materials_verified'):
        return []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return ['collected cDFT rows are not a native table']
    names=[row.get('MOF') for row in rows]
    if any(not isinstance(name,str) for name in names) or len(names)!=len(set(names)) or set(names)!=set(quality['expected_materials']):
        return ['collected material identities differ from original inputs']
    errors=[]
    for row in rows:
        native=quality['materials'][row['MOF']]
        for gas in gases:
            targets={gas+'_density_mol_L':native.get('output_facts',{}).get('components',{}).get(gas,{}).get('density_mol_L'),
                     gas+'_henry_mol_L_atm':native.get('henry_mol_L_atm',{}).get(gas)}
            for key,expected in targets.items():
                try:
                    if expected is None or not math.isclose(_finite(row.get(key)),expected,rel_tol=1e-10,abs_tol=1e-12):
                        raise ValueError('collected value differs from native result')
                except (ValueError,TypeError):
                    errors.append(str(row['MOF'])+'/'+key+': missing, nonfinite or mismatched native value')
    return errors
