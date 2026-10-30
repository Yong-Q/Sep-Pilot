"""Bounded native RASPA case facts; scripts/log messages are not samples."""
import hashlib
import math
import os
from pathlib import Path
import re
import stat

from .workload_budget import _owned_fd

TOOLS=frozenset({'run_gcmc_isotherm','run_gcmc_batch'})


class GCMCInspectionLimit(ValueError):
    pass


class _StatisticalBlocks:
    """Count native index labels in the same bounded pass used for hashing."""
    pattern=re.compile(rb'\bBlock\s{0,32}\[\s{0,32}([0-9]{1,10})\s{0,32}\]',re.IGNORECASE)
    limit=256

    def __init__(self):
        self.tail=b'';self.bytes_seen=0;self.counts={};self.total=0;self.truncated=False

    def feed(self,chunk):
        window=self.tail+chunk;offset=self.bytes_seen-len(self.tail)
        for match in self.pattern.finditer(window):
            # Ignore matches already complete in the previous chunk. Carry
            # enough bytes for the longest accepted native marker (113 bytes).
            if offset+match.end()<=self.bytes_seen:continue
            index=int(match.group(1));self.total+=1
            if index in self.counts:self.counts[index]+=1
            elif len(self.counts)<self.limit:self.counts[index]=1
            else:self.truncated=True
        self.bytes_seen+=len(chunk);self.tail=window[-128:]

    def result(self):
        indices=sorted(self.counts)
        return {'indices':indices,'unique_index_count':None if self.truncated else len(indices),
            'retained_unique_index_count':len(indices),'marker_occurrences':self.total,
            'occurrences_by_index':{str(index):self.counts[index] for index in indices},
            'duplicate_index_occurrences':None if self.truncated else self.total-len(indices),
            'scan_complete':True,'truncated':self.truncated,'index_storage_limit':self.limit,
            'source':'full_native_data_stream','establishes_statistical_convergence':False}



class _NativeUnitCells:
    """Bounded three-axis/warning scan in the same complete hashed stream."""
    axis=re.compile(rb'(?im)^[ \t]{0,32}Number[ \t]+of[ \t]+unitcells[ \t]*\[[ \t]*([abc])[ \t]*\][ \t]*:[ \t]*([^\s\r\n]{1,32})[ \t]*\r?\n')
    warning=re.compile(rb'inappropriate[ \t]+number[ \t]+of[ \t]+unit[ \t]+cells',re.I)
    def __init__(self):
        self.tail=b'';self.prefix=b'';self.seen=0;self.counts={a:0 for a in 'abc'};self.values={};self.warnings=0
    def feed(self,chunk):
        window=self.tail+chunk;offset=self.seen-len(self.tail)
        for match in self.axis.finditer(window):
            if offset+match.end()<=self.seen:continue
            if match.start()==0 and offset>0 and self.prefix!=b'\n':continue
            axis=match.group(1).decode().lower();self.counts[axis]+=1
            token=match.group(2)
            self.values[axis]=int(token)if re.fullmatch(rb'[1-9][0-9]*',token)else None
        for match in self.warning.finditer(window):
            if offset+match.end()>self.seen:self.warnings+=1
        self.prefix=window[-257:-256]if len(window)>256 else self.prefix
        self.seen+=len(chunk);self.tail=window[-256:]
    def result(self):
        complete=all(self.counts[a]==1 and self.values.get(a)is not None for a in 'abc')
        return {'values':[self.values[a]for a in 'abc']if complete else None,
            'occurrences':dict(self.counts),'complete':complete,'cutoff_warning_count':self.warnings,
            'source':'full_native_data_stream'}


def _geometry_expected(root,directory,tool,arguments,receipt,gas,budget):
    """Verify native geometry and its actual staged bytes, never model lists."""
    from .gcmc_geometry import verify_geometry_proof
    receipt=receipt if isinstance(receipt,dict)else {}
    proof=receipt.get('geometry_proof')if tool=='run_gcmc_isotherm'else (receipt.get('geometry_proofs')or {}).get(directory.name)
    if not isinstance(proof,dict):return None,None
    source=Path(arguments['cif'])if tool=='run_gcmc_isotherm'else Path(arguments['cif_dir'])/(directory.name+'.cif')
    if Path(proof.get('source',{}).get('path',''))!=source or not verify_geometry_proof(proof,source,12.0):
        raise ValueError('native geometry proof does not match the current source CIF and cutoff')
    requested=arguments.get('unit_cells')
    auto=requested is None or requested=='auto'
    if proof.get('mode')!=('auto'if auto else 'explicit'):
        raise ValueError('native geometry mode differs from the approved replication request')
    if not auto and proof.get('unit_cells')!=[int(v)for v in requested.split()]:
        raise ValueError('native geometry axes differ from explicit approved axes')
    staged=proof.get('staged_source')if tool=='run_gcmc_isotherm'else (proof.get('staged_sources')or {}).get(str(gas))
    if not isinstance(staged,dict):raise ValueError('native geometry lacks gas-specific staged CIF identity')
    stage_path=Path(staged.get('path',''))
    expected_case=directory/source.name
    permitted={expected_case,root/source.name}if tool=='run_gcmc_isotherm'else {expected_case}
    if stage_path not in permitted:raise ValueError('native staged CIF path differs from the actual adapter case')
    _,fact=_read(root,stage_path,budget)
    expected_sha=proof.get('source_sha256');expected_size=proof['source']['size']
    if tool=='run_gcmc_isotherm':
        from .raspa_assets import _read as read_cif
        from .raspa_cif_normalization import normalize_raspa_cif_bytes
        original,source_fact=read_cif(source);budget['bytes']+=len(original)
        if budget['bytes']>budget['max_bytes']:raise GCMCInspectionLimit('native geometry source exceeds total read bound')
        if source_fact['sha256']!=proof['source_sha256']:
            raise ValueError('original CIF changed after geometry verification')
        normalized=normalize_raspa_cif_bytes(original)
        expected_sha=hashlib.sha256(normalized).hexdigest();expected_size=len(normalized)
    if (any(fact[key]!=staged.get(key)for key in ('sha256','size'))
            or fact['sha256']!=expected_sha or fact['size']!=expected_size):
        raise ValueError('native staged CIF bytes differ from verified geometry source')
    _,case_fact=_read(root,expected_case,budget)
    if case_fact['sha256']!=fact['sha256']or case_fact['size']!=fact['size']:
        raise ValueError('actual pressure/gas case CIF differs from staged geometry source')
    return proof['unit_cells'],{'source':proof['source'],'staged_source':fact,'case_source':case_fact,
        'geometry_sufficiency':proof.get('geometry_sufficiency'),'face_heights':proof.get('face_heights')}


def _unit_cell_check(text,arguments,native,*,geometry_expected=None,geometry_proof=None,geometry_error=None):
    issues=[]
    def issue(code,message):issues.append({'code':code,'message':message})
    expected_raw=arguments.get('unit_cells','2 2 2')
    expected_tokens=expected_raw.split()if isinstance(expected_raw,str)else []
    valid=lambda tokens:len(tokens)==3 and all(re.fullmatch(r'[1-9][0-9]*',v)for v in tokens)
    expected=[int(v)for v in expected_tokens]if valid(expected_tokens)else None
    if geometry_expected is not None:expected=geometry_expected
    if geometry_error:issue('unit_cells_geometry_proof_invalid',geometry_error)
    if expected is None:issue('unit_cells_contract_invalid','The approved contract requires three positive integer replication axes.')
    directives=re.findall(r'^\s*UnitCells[ \t]+([^\r\n#]+)',text,re.M|re.I)
    tokens=directives[0].split()if len(directives)==1 else []
    observed=[int(v)for v in tokens]if tokens and all(re.fullmatch(r'[0-9]+',v)for v in tokens)else None
    if not valid(tokens):issue('unit_cells_input_invalid','simulation.input must explicitly declare exactly three positive integer axes; engine defaults are not authorization.')
    elif observed!=expected:issue('unit_cells_input_mismatch','simulation.input replication differs from the current approved contract.')
    actual=native.get('values')
    if not native.get('complete'):issue('unit_cells_native_missing_or_ambiguous','Native output must attest one unambiguous positive value for each of a, b and c.')
    elif actual!=expected or actual!=observed:issue('unit_cells_native_mismatch','Native executed replication differs from input or approved contract.')
    return {'verified':not issues,'expected':expected,'input':observed,'native':actual,
        'native_geometry':native,'issues':issues,'geometry_proof_verified':geometry_proof is not None,
        'geometry_proof':geometry_proof,
        'geometry_assessment':{'status':'requires_lattice_cutoff_assessment'if native.get('cutoff_warning_count',0)else 'not_assessed',
            'cutoff_safety_established':False,
            'note':'A warning alone is not a geometric verdict. Use the actual lattice perpendicular face heights and cutoff; do not infer safety from abc lengths or finite loading.'}}

def native_output_contract(tool,arguments):
    if tool not in TOOLS:raise ValueError('unsupported native GCMC layout')
    # The artifact grammar intentionally forbids recursive path globs. Only
    # one concrete native case has an equivalent single-directory contract.
    try:cases=_cases(Path('.'),tool,arguments)
    except (OSError,ValueError,TypeError,KeyError,ImportError):return None
    if len(cases)!=1:return None
    return [{'kind':'directory','path':str(cases[0][0]/'Output/System_0'),'pattern':'*.data','min_count':1}]


def _read(root,path,budget,limit=262144,scan_blocks=False):
    fd=_owned_fd(root,path)
    try:
        before=os.fstat(fd)
        if before.st_size>128*1024*1024:raise GCMCInspectionLimit('native file exceeds bounded reader')
        if not stat.S_ISREG(before.st_mode) or not before.st_size>0:
            raise ValueError('native GCMC evidence must be a bounded nonempty regular file')
        digest=hashlib.sha256();head=b'';tail=b'';size=0;small=[]
        statistics=_StatisticalBlocks() if scan_blocks else None
        geometry=_NativeUnitCells() if scan_blocks else None
        while True:
            block=os.read(fd,65536)
            if not block:break
            size+=len(block);budget['bytes']+=len(block)
            if size>128*1024*1024 or budget['bytes']>budget['max_bytes']:
                raise GCMCInspectionLimit('native GCMC evidence exceeds total read bound')
            digest.update(block)
            if statistics is not None:statistics.feed(block)
            if geometry is not None:geometry.feed(block)
            if size<=limit:small.append(block)
            else:small=[]
            if len(head)<limit//2:head+=(block[:limit//2-len(head)])
            tail=(tail+block)[-limit//2:]
        if geometry is not None:geometry.feed(b'\n')  # EOF terminates a final native line; bytes/hash remain unchanged.
        after=os.fstat(fd)
        linked=_owned_fd(root,path)
        try:current=os.fstat(linked)
        finally:os.close(linked)
        identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        if size!=before.st_size or identity(before)!=identity(after) or identity(after)!=identity(current):
            raise ValueError('native GCMC evidence changed while reading')
        if size<=limit:raw=b''.join(small)
        else:raw=head+b'\n[bounded native middle omitted]\n'+tail
        fact={'path':str(path),'size':size,
            'sha256':digest.hexdigest(),'mtime_ns':after.st_mtime_ns,'ctime_ns':after.st_ctime_ns,
            'inode':after.st_ino}
        if statistics is not None:fact['statistical_blocks']=statistics.result()
        if geometry is not None:fact['native_unit_cells']=geometry.result()
        return raw.decode('utf-8',errors='replace'),fact
    finally:os.close(fd)


def _cases(root,tool,args):
    if tool=='run_gcmc_isotherm':
        import numpy as np
        count=args.get('n_pressure_points',10)
        if not isinstance(count,int) or isinstance(count,bool) or not 1<=count<=512:raise ValueError('invalid pressure point count')
        pressures=np.geomspace(float(args.get('pressure_start',0.1)),float(args.get('pressure_end',10)),count)
        names=['P_'+format(value*1e5,'.2e') for value in pressures]
        if len(set(names))!=count or not all(math.isfinite(value) and value>0 for value in pressures):
            raise ValueError('pressure point native identities collide or are invalid')
        return [(root/name,args.get('gas','CO2'),float(name[2:])) for name in names]
    gases=args.get('gases');source=Path(args.get('cif_dir',''))
    if not isinstance(gases,list) or not gases or len(gases)>64 or not source.is_dir():
        raise ValueError('batch requires its explicit gas and original CIF inventory')
    from itertools import islice
    files=sorted(islice(source.glob('*.cif'),513))
    if not files or len(files)>512 or len(files)*len(gases)>512:raise ValueError('batch case inventory exceeds bound')
    temperature=args.get('temperature',298.0);pressure=float(args.get('pressure',1.0))*1e5
    if not math.isfinite(pressure) or pressure<=0:raise ValueError('invalid batch pressure')
    return [(root/str(gas)/(str(temperature)+'K')/path.stem,gas,pressure) for gas in gases for path in files]


def _input_matches(text,args,gas,pressure,framework=None):
    from .slurm import _gas_preset
    preset=_gas_preset(gas)
    def field(name):
        values=re.findall(r'^\s*'+re.escape(name)+r'\s+([^\r\n#]+)',text,re.MULTILINE)
        if len(values)!=1:raise ValueError('native simulation.input lacks one '+name)
        return values[0].strip()
    cycles=args.get('cycles',50000)
    for label,expected in [('NumberOfCycles',cycles),('NumberOfInitializationCycles',2*cycles),
            ('ExternalTemperature',args.get('temperature',298.0)),('ExternalPressure',pressure)]:
        value=float(field(label))
        if not math.isfinite(value) or not math.isclose(value,float(expected),rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('native simulation.input differs from approved '+label)
    for label,expected in [('Forcefield',args.get('force_field') or preset['forcefield']),
            ('MoleculeDefinition',args.get('guest_ff') or preset['molecule_definition'])]:
        if field(label)!=expected:raise ValueError('native simulation.input differs from approved '+label)
    names=re.findall(r'^\s*Component\s+0\s+MoleculeName\s+(\S+)',text,re.MULTILINE)
    if names!=[preset['molecule_name']]:raise ValueError('native simulation.input gas identity differs')
    if framework and field('FrameworkName')!=framework:raise ValueError('native framework identity differs')
    method=args.get('electrostatics_method',args.get('charge_method'))
    if method and method!='None' and field('ChargeMethod')!=method:raise ValueError('native electrostatics differs')
    return {'production_cycles':cycles,'initialization_cycles':2*cycles,'temperature_K':args.get('temperature',298.0),
        'pressure_Pa':pressure,'gas':gas,'input_matches_contract':True}


def inspect_gcmc_results(root,tool,arguments,*,max_bytes=256*1024*1024,native_receipt=None):
    root=Path(os.path.abspath(root));results=[];budget={'bytes':0,'max_bytes':max_bytes};errors=[];integrity_errors=[]
    try:
        if tool not in TOOLS:raise ValueError('unsupported GCMC native result tool')
        if root.resolve(strict=True)!=root:raise ValueError('native result root contains a symlink')
        guard=_owned_fd(root,root,directory=True);os.close(guard)
        cases=_cases(root,tool,arguments)
        for directory,gas,pressure in cases:
            item={'case_dir':str(directory),'ok':False};stage='output_inventory'
            try:
                fd=_owned_fd(root,directory/'Output/System_0',directory=True)
                try:
                    directory_stat=os.fstat(fd)
                    with os.scandir(fd) as entries:
                        names=[]
                        for index,entry in enumerate(entries):
                            if index>=4096:raise GCMCInspectionLimit('native output directory exceeds entry bound')
                            if entry.name.endswith('.data'):names.append(entry.name)
                finally:os.close(fd)
                item['output_directory']={'path':str(directory/'Output/System_0'),'size':directory_stat.st_size,
                    'inode':directory_stat.st_ino,'mtime_ns':directory_stat.st_mtime_ns,'ctime_ns':directory_stat.st_ctime_ns,
                    'directory':True}
                if len(names)!=1:raise ValueError('native case requires exactly one unambiguous .data result')
                stage='native_input'
                text,input_fact=_read(root,directory/'simulation.input',budget,limit=1024*1024)
                if input_fact['size']>1024*1024:raise ValueError('simulation.input exceeds bounded text format')
                framework=Path(arguments['cif']).stem if tool=='run_gcmc_isotherm' and arguments.get('cif') else directory.name if tool=='run_gcmc_batch' else None
                item.update(_input_matches(text,arguments,gas,pressure,framework))
                input_text=text;stage='native_output'
                text,data_fact=_read(root,directory/'Output/System_0'/names[0],budget,scan_blocks=True)
                geometry_expected=geometry_proof=geometry_error=None
                try:
                    geometry_expected,geometry_proof=_geometry_expected(root,directory,tool,arguments,native_receipt,gas,budget)
                    if arguments.get('unit_cells')=='auto' and geometry_proof is None:
                        raise ValueError('explicit auto replication requires a verified native geometry receipt')
                except (OSError,ValueError,TypeError,KeyError)as error:geometry_error=str(error)
                unit_check=_unit_cell_check(input_text,arguments,data_fact['native_unit_cells'],
                    geometry_expected=geometry_expected,geometry_proof=geometry_proof,geometry_error=geometry_error)
                item['unit_cell_check']=unit_check
                if not unit_check['verified']:
                    item['error_kind']=unit_check['issues'][0]['code']
                    integrity_errors.append({'case_dir':str(directory),'issues':unit_check['issues']})
                lowered=text.lower();completed=any(marker in lowered for marker in ('simulation finished','simulation completed','normal termination'))
                loadings={}
                for match in re.finditer(r'Average loading absolute\s+\[([^\]]+)\]\s+([^\s]+)',text,re.IGNORECASE):
                    try:
                        value=float(match.group(2));loadings[match.group(1)]=value if math.isfinite(value) else None
                    except ValueError:loadings[match.group(1)]=None
                finite=bool(loadings) and all(value is not None for value in loadings.values())
                failed=any(marker in lowered for marker in ('simulation failed','convergence error','fatal error'))
                item.update(ok=completed and finite and not failed and unit_check['verified'],completion_marker=completed,finite_loading=finite,
                    loadings=loadings,data=data_fact,native_input=input_fact,
                    statistical_blocks=data_fact['statistical_blocks'],
                    statistical_convergence_established=False)
                if not item['ok']:item['reason']=('; '.join(x['message']for x in unit_check['issues'])if not unit_check['verified']else 'native data lacks completion or finite final loading')
            except (OSError,ValueError,TypeError,KeyError) as error:
                item['reason']=str(error)
                item['error_kind']='missing_or_invalid_native_input'if stage=='native_input'else 'missing_or_invalid_native_output'
                if stage=='native_input':integrity_errors.append({'case_dir':str(directory),'issues':[{'code':item['error_kind'],'message':str(error)}]})
                if isinstance(error,GCMCInspectionLimit):errors.append(str(error))
            results.append(item)
    except (OSError,ValueError,TypeError,KeyError) as error:errors.append(str(error))
    successful=[item['data']['path'] for item in results if item['ok']]
    return {'schema_version':2,'root':str(root),'tool':tool,'read_only':True,'cases':results,
        'counts':{'attempted':len(results),'succeeded':len(successful),'failed':len(results)-len(successful)},
        'successful_artifacts':successful,'all_cases_passed':bool(results) and len(successful)==len(results) and not errors,
        'errors':errors,'integrity_errors':integrity_errors,'bytes_read':budget['bytes'],'protocol_source':'native_adapter_layout_and_simulation_input',
        'statistical_convergence_established':False}


def proof_files_unchanged(proof,owned_root):
    """After hashing, a final no-follow stat guards the metadata commit."""
    facts=[proof['script'],proof['execution_evidence_fact'],proof['producer_checkpoint_fact']]
    for item in proof['native']['cases']:
        facts.extend(item[key] for key in ('data','native_input','output_directory') if item.get(key))
    for fact in facts:
        fd=_owned_fd(Path(owned_root),Path(fact['path']),directory=fact.get('directory',False))
        try:
            current=os.fstat(fd)
            regular=stat.S_ISDIR(current.st_mode) if fact.get('directory') else stat.S_ISREG(current.st_mode)
            if (not regular or any(getattr(current,'st_'+field)!=fact[field] for field in ('size','mtime_ns','ctime_ns'))
                    or current.st_ino!=fact['inode']):
                return False
        finally:os.close(fd)
    return True


def require_current_gcmc_contract(root,tool,arguments,*,native_receipt=None):
    """Fresh technical/native-input hard gate, never a convergence verdict."""
    native=inspect_gcmc_results(root,tool,arguments,native_receipt=native_receipt)
    if not native.get('all_cases_passed') or native.get('integrity_errors'):
        error=ValueError('GCMC actual input/native output does not satisfy the current contract; owner prose cannot override replication or missing evidence')
        error.details={'error_kind':'gcmc_native_contract','counts':native.get('counts'),
            'integrity_errors':native.get('integrity_errors'),
            'failed_cases':[{'case_dir':c.get('case_dir'),'error_kind':c.get('error_kind'),
                'unit_cell_check':c.get('unit_cell_check'),'reason':c.get('reason')}for c in native.get('cases',[])if not c.get('ok')]}
        raise error
    return native
