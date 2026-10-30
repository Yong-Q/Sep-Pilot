"""Owned native input facts and separately typed scheduler provenance.

No directive is data. Resource-review requests are not scheduler measurements,
job identifiers are not scientific quantities, and unsealed historical CIFs
are never promoted to original execution snapshots.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

SCHEMA = 'native_report_source_metadata_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _binding(owner, source, summary):
    return {'owner': dict(owner), 'step_id': summary['step_id'], 'token': source.get('token'),
            'attempt_id': source.get('attempt_id'), 'producer_call_id': summary['call_id'],
            'producer_sha256': summary['source_sha256']}


def _publish(root, role, public, binding, *, read, budget, file_facts):
    from .ml_dataset import immutable_json
    value = {'schema': SCHEMA, 'role': role, 'public': public, 'binding': binding,
             'execution_authorized': False, 'scientific_acceptance': False}
    path = Path(root) / 'evidence' / ('report-meta-' + _digest(value) + '.json')
    immutable_json(root, path, value)
    raw, fact = read(root, path, budget); file_facts.append(fact)
    return {'path': fact['path'], 'sha256': fact['sha256'], 'role': role,
            'json_root_pointer': '/public', 'untrusted_content': public,
            'native_projection': 'verified host metadata projection; original bytes retained',
            'metadata_schema': SCHEMA}


def _resource_calls(root, allocation, *, read, budget, file_facts):
    identifiers=allocation.get('evidence_call_ids')or[]
    if not 1<=len(identifiers)<=16 or len(set(identifiers))!=len(identifiers):
        return False
    verified=[]
    for identifier in identifiers:
        if not isinstance(identifier,str)or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',identifier):return False
        try:
            raw,fact=read(root,Path(root)/'evidence'/(identifier+'.json'),budget)
            call=json.loads(raw)
            if (call.get('call_id')!=identifier or call.get('tool')not in{'resource_health','assess_job_resources'}
                    or call.get('failed')is not False or not isinstance(call.get('result'),str)
                    or hashlib.sha256(call['result'].encode()).hexdigest()!=call.get('result_sha256')):
                return False
            verified.append(fact)
        except (OSError,ValueError,KeyError,TypeError):return False
    file_facts.extend(verified)
    return True


def scheduler_document(root, owner, state, source, summary, native_result, *,
                       job_lookup, read, budget, file_facts):
    """Historical native review request, bound to actual owner/step/attempt.

    The callback is the host JobWatch getter, never model-supplied. Unknown or
    non-matching receipts yield no provenance rather than guessing resources.
    """
    if not callable(job_lookup):
        return None
    identifiers = source.get('job_ids') or []
    key, attempt = source.get('recovery_key'), source.get('attempt_id')
    if (not identifiers or len(identifiers) > 64 or not key or not attempt
            or len(set(identifiers)) != len(identifiers)
            or any(not isinstance(item, str) or not re.fullmatch(r'[1-9][0-9]{0,17}', item) for item in identifiers)
            or native_result.get('submitted') is not True):
        return None
    ledger_raw, _ = read(root, Path(root) / 'recovery_state.json', budget)
    entry = json.loads(ledger_raw).get(key) or {}
    receipt = entry.get('last_result') or {}
    actual_ids = native_result.get('job_ids') or [native_result.get('job_id')]
    receipt_ids = receipt.get('job_ids') or [receipt.get('job_id')]
    arguments = source.get('effective_arguments') or source['contract'].get('arguments') or {}
    from .recovery import RecoveryGate
    identity_version=entry.get('identity_goal_version',entry.get('goal_version'))
    if (type(identity_version) is not int or type(entry.get('goal_version')) is not int
            or not 1<=identity_version<=entry['goal_version']):
        return None
    expected_key=RecoveryGate().key(summary['tool'],arguments,identity_version,
        summary['step_id'],str(Path(root).parents[2]))
    if (entry.get('key') != key or entry.get('tool') != summary['tool'] or entry.get('attempt_id') != attempt
            or source['contract'].get('step_id') != summary['step_id'] or key != expected_key
            or set(entry.get('job_ids') or []) != set(identifiers)
            or set(actual_ids) != set(identifiers) or set(receipt_ids) != set(identifiers)
            or entry.get('params') != arguments or receipt.get('submitted') is not True
            or type(entry.get('goal_version')) is not int or entry['goal_version'] < 1
            or entry.get('goal_version') != source.get('origin_goal_version')):
        return None
    observed = []
    for identifier in identifiers:
        item = job_lookup(identifier) or {}
        if (not isinstance(item, dict) or item.get('job_id') != identifier
                or (item.get('username'), item.get('conv_id')) != (owner['username'], owner['conv_id'])
                or item.get('attempt_id') != attempt or item.get('recovery_key') != key
                or item.get('tool') != summary['tool']
                or item.get('work_dir') != receipt.get('work_dir')):
            return None
        observed.append({field: item.get(field) for field in (
            'job_id', 'username', 'conv_id', 'attempt_id', 'recovery_key', 'tool', 'work_dir', 'submitted_at', 'state')})
    public = {'job_ids': list(identifiers), 'scheduler_states': [item['state'] for item in observed],
              'native_review_requested_memory_mb': None, 'native_review_requested_cpus': None,
              'native_review_requested_gpus': None, 'scheduler_observed_memory_mb': None,
              'resource_scope': 'Native final-review request only; scheduler effective allocation and process RSS are not measured here.',
              'job_id_role': 'provenance identifier; not a scientific numerical observation'}
    review = source.get('resource_review') or {}
    allocation = source.get('resource_allocation') or {}
    step = summary['step_id']
    reviewed = (review.get('resource_allocations') or {}).get(step)
    if (review.get('status') == 'ready' and reviewed == allocation
            and allocation.get('budget_final') is True
            and allocation.get('resource_review_id') == review.get('resource_review_id')
            and allocation.get('goal_version') == source.get('origin_goal_version')
            and allocation.get('evidence_call_ids')
            and all(type(allocation.get(field)) is int and allocation[field] > 0
                    for field in ('memory_mb', 'cpus'))
            and (allocation.get('gpus') is None or type(allocation.get('gpus')) is int and allocation['gpus'] >= 0)
            and _resource_calls(root,allocation,read=read,budget=budget,file_facts=file_facts)):
        public.update(native_review_requested_memory_mb=allocation.get('memory_mb'),
                      native_review_requested_cpus=allocation.get('cpus'),
                      native_review_requested_gpus=allocation.get('gpus'))
    binding = _binding(owner, source, summary)
    binding.update(recovery_key=key, native_receipt_sha256=_digest(receipt),
                   watch_records_sha256=_digest(observed), native_resource_review_sha256=_digest(review))
    return _publish(root, 'provenance', public, binding, read=read, budget=budget, file_facts=file_facts)


def cdft_document(root, owner, source, summary, native_result, scheduler, *, read, budget, file_facts):
    """Project source-bound cDFT input and native subprocess facts, not convergence."""
    if summary.get('tool')!='run_cdft' or not scheduler:return None
    args=source.get('effective_arguments') or source.get('contract',{}).get('arguments') or {}
    if args.get('action') not in {'submit','pipeline'}:return None
    work=native_result.get('work_dir');original=args.get('input_dir')
    gases=args.get('gases') or ([args['gas']] if args.get('gas') else [])
    if not work or not original or not gases:return None
    from .cdft_result_facts import inspect_cdft_batch_quality
    quality=inspect_cdft_batch_quality(work,owned_root=root,gases=gases,source_input_dir=original)
    if (quality.get('status')!='observed' or quality.get('expected_materials_verified') is not True
            or quality.get('complete_material_outputs') is not True):return None
    batch_path=Path(work)/'batch_execution.json';raw,batch_fact=read(root,batch_path,budget)
    batch=json.loads(raw);job_ids=set(source.get('job_ids') or [])
    if str(batch.get('job_id')) not in job_ids:return None
    facts=[batch_fact];materials={}
    _,csv_fact=read(root,Path(work)/'results.csv',budget)
    if csv_fact['sha256']!=quality.get('collected_csv_sha256'):return None
    facts.append(csv_fact)
    for item in (quality.get('original_input_facts') or {}).get('inputs',[]):
        relative=Path(item.get('path',''))
        if relative.name!=str(relative) or relative.suffix!='.dat':return None
        _,fact=read(root,Path(original)/relative,budget)
        if fact['sha256']!=item.get('sha256') or fact['size']!=item.get('bytes'):return None
        facts.append(fact)
    for name,item in quality.get('materials',{}).items():
        if item.get('errors') or item.get('technical_completion') is not True:return None
        receipt_path=Path(work)/'input_dir'/name/'execution.json'
        receipt_raw,receipt_fact=read(root,receipt_path,budget);receipt=json.loads(receipt_raw)
        if (str(receipt.get('job_id')) not in job_ids or receipt.get('exit_code')!=0
                or receipt.get('timed_out') is not False
                or (batch.get('outcomes') or {}).get(name+'.dat')!=receipt):return None
        artifact=item.get('artifacts',{}).get(str(receipt_path)) or {}
        if artifact.get('sha256')!=receipt_fact['sha256']:return None
        wall=receipt.get('wall_seconds')
        if type(wall) not in (int,float) or not math.isfinite(wall) or wall<0:return None
        observed=item.get('input_facts') or {}
        fields=('temperature_K','native_grid','requested_spacing_A','actual_axis_spacing_A','grid_capped')
        if any(field not in observed for field in fields):return None
        materials[name]={**{field:observed[field] for field in fields},'native_wall_seconds':wall,
                         'exit_code':receipt['exit_code'],'scientific_convergence':None}
        facts.append(receipt_fact)
        for path,expected in item.get('artifacts',{}).items():
            if path.endswith(('/input.dat','.dat')) and expected.get('size',0)>0:
                _,fact=read(root,path,budget)
                if fact['sha256']!=expected.get('sha256'):return None
                facts.append(fact)
    if not materials:return None
    file_facts.extend(facts)
    binding=_binding(owner,source,summary)
    binding.update(scheduler_metadata_sha256=scheduler['sha256'],batch_sha256=batch_fact['sha256'],
        input_sources=[{'path':fact['path'],'sha256':fact['sha256'],'bytes':fact['size']}for fact in facts])
    return _publish(root,'native_input_evidence',{'materials':materials,'producer_build_authenticated':False,
        'limitations':['Native subprocess completion and mesh observations do not prove scientific convergence.',
                       'Axis spacing derives from the supplied native cell and grid; force-field validation is separate.',
                       'The observed receipts do not independently authenticate the solver build.']},
        binding,read=read,budget=budget,file_facts=file_facts)


def cdft_ancestor_documents(root, owner, state, source, native_result, *,
                            job_lookup, attest, read, budget, file_facts):
    """Attach only authenticated direct cDFT inputs of an owned figure.

    A figure title or drawing is never a numerical source. The figure's own
    producer receipt must bind the exact original node, call, and contract;
    scheduler and byte-level native proofs are then rebuilt independently.
    """
    if source.get('contract',{}).get('tool')!='generate_result_figures':return []
    if native_result.get('source_receipts') is None:return []  # Legacy evidence keeps its existing restricted path.
    records=native_result.get('source_receipts') or [];steps=native_result.get('source_steps') or []
    if (not isinstance(records,list) or len(records)>16 or not isinstance(steps,list)
            or len(set(steps))!=len(steps) or {x.get('step_id')for x in records if isinstance(x,dict)}!=set(steps)):
        raise ValueError('figure ancestor receipts do not match original sources')
    documents=[]
    from .recovery import result_object
    from .workflow_patch import execution_contract
    for record in records:
        step=record.get('step_id');node=(state.get('nodes') or {}).get(step) or {};contract=node.get('contract') or {}
        if contract.get('tool')!='run_cdft':continue
        ref=node.get('result_ref') or {};call_id=ref.get('call_id');expected=Path(root)/'evidence'/(str(call_id)+'.json')
        if (step not in (source.get('contract',{}).get('depends_on') or [])
                or node.get('status')!='succeeded' or not node.get('token')
                or not isinstance(call_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',call_id)
                or any(record.get(k)!=node.get(k)for k in ('token','attempt_id','execution_fingerprint'))
                or record.get('call_id')!=call_id or ref.get('evidence_path')!=str(expected)
                or record.get('contract_sha256')!=_digest(execution_contract(contract))):
            raise ValueError('figure cDFT ancestor identity is stale')
        raw,fact=read(root,expected,budget);call=json.loads(raw)
        if (fact['sha256']!=record.get('source_sha256') or call.get('call_id')!=call_id
                or call.get('tool')!='run_cdft' or call.get('failed') is not False
                or call.get('params')!=(node.get('effective_arguments') or contract.get('arguments') or {})
                or result_object(call.get('result'))!=result_object(node.get('result'))):
            raise ValueError('figure cDFT original call changed')
        witness=attest(node,call,raw)
        if not isinstance(witness,dict) or witness.get('checkpoint_verified') is not True:
            raise ValueError('figure cDFT ancestor producer is not authenticated')
        _,checkpoint=read(root,witness.get('checkpoint_path') or ref.get('checkpoint_path'),budget)
        if checkpoint['sha256']!=witness.get('checkpoint_sha256'):
            raise ValueError('figure cDFT ancestor checkpoint changed')
        file_facts.extend([fact,checkpoint]);summary={'step_id':step,'tool':'run_cdft','call_id':call_id,'source_sha256':fact['sha256']}
        result=result_object(call['result'])
        scheduler=scheduler_document(root,owner,state,node,summary,result,job_lookup=job_lookup,
            read=read,budget=budget,file_facts=file_facts)
        if not scheduler:continue
        native=cdft_document(root,owner,node,summary,result,scheduler,read=read,budget=budget,file_facts=file_facts)
        scheduler['native_ancestor_step']=step
        documents.append(scheduler)
        if native:
            native['native_ancestor_step']=step
            documents.append(native)
    return documents


def parse_string_input(raw):
    """Parse fixed native labels, preserving unknown length/angle semantics."""
    text = raw.decode('utf-8-sig')
    if '\x00' in text:
        raise ValueError('native String input contains NUL')
    lines = text.splitlines()
    def values(label, count):
        found = [index for index, line in enumerate(lines) if ' '.join(line.split()) == ' '.join(label.split())]
        if len(found) != 1 or found[0] + 1 >= len(lines):
            raise ValueError('native String input label absent/ambiguous: ' + label)
        parts = lines[found[0] + 1].split()
        if len(parts) != count:
            raise ValueError('native String input width differs: ' + label)
        row = [float(value) for value in parts]
        if not all(math.isfinite(value) for value in row):
            raise ValueError('native String input contains nonfinite metadata')
        return row, {'header_line': found[0] + 1, 'value_line': found[0] + 2, 'header': label}
    cells, cell_anchor = values('La Lb Lc dL', 4)
    angles, angle_anchor = values('Alpha Beta Gamma', 3)
    grid, grid_anchor = values('Nmaxa Nmaxb Nmaxc:', 3)
    settings, settings_anchor = values('cutoff(A) FH_signal Mass(g/mol) Tempearture(K)  Running_steps', 5)
    direction, direction_anchor = values('Direction', 1)
    points, points_anchor = values('#_of_points delta_frac delta_angle_degree', 3)
    if (not all(value > 0 for value in cells + grid)
            or not all(0 < value < 180 for value in angles)
            or any(value != int(value) for value in grid + direction + [settings[4], points[0]])
            or int(direction[0]) not in {1, 2, 3} or not 2 <= points[0] <= 2000
            or settings[0] <= 0 or settings[2] <= 0 or settings[3] <= 0 or settings[4] < 1):
        raise ValueError('native String input metadata outside supported bounds')
    return {'cell_values': cells[:3], 'angles': angles, 'direction': int(direction[0]),
        'nmax': [int(value) for value in grid], 'dL': cells[3], 'cutoff_A': settings[0],
        'mass_g_mol': settings[2], 'temperature_K': settings[3], 'running_steps': int(settings[4]),
        'n_points': int(points[0]), 'native_length_unit': 'unspecified', 'native_angle_unit': 'unspecified',
        'binary_source_binding': False, 'adapter_declaration_only': True,
        'anchors': [cell_anchor, angle_anchor, grid_anchor, settings_anchor, direction_anchor, points_anchor]}


def string_input_documents(root, owner, state, source, summary, native_result, candidates, *,
                           read, budget, file_facts):
    if summary['tool'] != 'run_string_tst':
        return []
    manifest = native_result.get('result_json_path')
    raw, _ = read(root, manifest, budget)
    result = json.loads(raw)
    if not any(row.get('input_path') or row.get('input_sha256') for row in result.get('results', [])):
        return []  # Older output-only receipts do not gain invented input facts.
    path, expected = result.get('manifest_path'), result.get('manifest_sha256')
    if path != native_result.get('manifest_path') or path not in candidates:
        raise ValueError('native String input manifest lacks its declared artifact binding')
    raw, manifest_fact = read(root, path, budget); file_facts.append(manifest_fact)
    if manifest_fact['sha256'] != expected:
        raise ValueError('native String input manifest SHA changed')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if not rows or len(rows) != len(result['results']) or len(rows) > 16:
        raise ValueError('native String input/result inventory differs')
    by_input = {item.get('input_dat'): item for item in rows}
    if len(by_input) != len(rows):
        raise ValueError('duplicate native String input declaration')
    documents = []
    for item in result['results']:
        declared = by_input.get(item.get('input_path'))
        if not declared or declared.get('structure') != item.get('structure') or str(declared.get('direction')) != str(item.get('direction')):
            raise ValueError('native String structure/direction/input identity differs')
        input_path = Path(item['input_path'])
        if not input_path.is_absolute() or not input_path.is_relative_to(Path(manifest).parent):
            raise ValueError('native String input path outside its native branch')
        raw, fact = read(root, input_path, budget); file_facts.append(fact)
        if fact['sha256'] != item.get('input_sha256'):
            raise ValueError('execution-bound native String input SHA changed')
        parsed = parse_string_input(raw)
        if parsed['direction'] != int(declared['direction']) or parsed['n_points'] != item.get('expected_point_count'):
            raise ValueError('native String header disagrees with execution receipt')
        public = {'structure': item['structure'], 'native_input': parsed,
                  'current_cif_geometry': None, 'historical_cif_snapshot_verified': False,
                  'interpretation': 'Input.dat bytes are execution-bound. CIF geometry, if present, is a current observation. Native length/angle semantics and output energy units are not promoted.'}
        inputs = [{'path': str(input_path), 'sha256': fact['sha256'], 'bytes': fact['size']},
                  {'path': path, 'sha256': manifest_fact['sha256'], 'bytes': manifest_fact['size']}]
        cif = Path(declared.get('cif') or '')
        if cif.is_absolute() and cif.is_relative_to(Path(root)):
            _, cif_fact = read(root, cif, budget); file_facts.append(cif_fact)
            from .gcmc_geometry import derive_gcmc_geometry
            geometry = derive_gcmc_geometry(cif, parsed['cutoff_A'], owned_root=root)
            if not geometry.get('ok') or geometry.get('source_sha256') != cif_fact['sha256']:
                raise ValueError('current owned CIF geometry could not be independently parsed')
            public['current_cif_geometry'] = {'lengths': geometry['lengths'], 'angles': geometry['angles'],
                'length_unit': geometry['length_unit'], 'angle_unit': geometry['angle_unit'],
                'sha256': cif_fact['sha256'], 'bytes': cif_fact['size'],
                'observation_scope': 'current owned CIF; historical CIF SHA was not recorded by this older producer',
                'agrees_with_sealed_input': all(math.isclose(a, b, abs_tol=5e-6, rel_tol=0)
                    for a, b in zip(geometry['lengths'] + geometry['angles'], parsed['cell_values'] + parsed['angles']))}
            inputs.append({'path': str(cif), 'sha256': cif_fact['sha256'], 'bytes': cif_fact['size']})
        binding = _binding(owner, source, summary); binding['input_sources'] = inputs
        documents.append(_publish(root, 'native_input_evidence', public, binding,
            read=read, budget=budget, file_facts=file_facts))
    return documents


def metadata_tables(document, source_id, tables):
    if document.get('role') != 'native_input_evidence':
        return
    value = document.get('untrusted_content')
    if not isinstance(value,dict):raise ValueError('native input metadata must be an object')
    if 'materials' in value and 'native_input' in value:
        raise ValueError('ambiguous native input metadata shape')
    if 'materials' in value:
        materials=value['materials']
        if (not isinstance(materials,dict) or not 1<=len(materials)<=512
                or value.get('producer_build_authenticated') is not False):
            raise ValueError('invalid cDFT native input metadata')
        rows=[]
        for name,item in sorted(materials.items()):
            if not isinstance(name,str) or not name or not isinstance(item,dict):raise ValueError('invalid cDFT material evidence')
            grid=item.get('native_grid');spacing=item.get('actual_axis_spacing_A')
            finite=lambda n:type(n) in (int,float) and math.isfinite(n)
            if (not isinstance(grid,list) or len(grid)!=3 or any(type(n)is not int or n<=0 for n in grid)
                    or not isinstance(spacing,list) or len(spacing)!=3 or any(not finite(n) or n<=0 for n in spacing)
                    or not finite(item.get('temperature_K')) or item['temperature_K']<=0
                    or not finite(item.get('requested_spacing_A')) or item['requested_spacing_A']<=0
                    or type(item.get('grid_capped')) is not bool
                    or not finite(item.get('native_wall_seconds')) or item['native_wall_seconds']<0
                    or type(item.get('exit_code')) is not int or item['exit_code']!=0
                    or item.get('scientific_convergence','missing') is not None):
                raise ValueError('invalid cDFT native input values')
            rows.append({'material':name,'temperature_K':item['temperature_K'],
                'grid_x':grid[0],'grid_y':grid[1],'grid_z':grid[2],
                'requested_spacing_A':item['requested_spacing_A'],
                'actual_spacing_x_A':spacing[0],'actual_spacing_y_A':spacing[1],'actual_spacing_z_A':spacing[2],
                'grid_capped':item['grid_capped'],'scientific_convergence':None})
        columns=list(rows[0]);identity=_digest({'kind':'native_cdft_input_observation','sha256':document['sha256']})
        prior=next((table for table in tables if table.get('native_identity_sha256')==identity),None)
        if prior:
            if source_id not in prior['source_ids']:prior['source_ids'].append(source_id)
            return
        tables.append({'id':'T'+str(len(tables)+1),'source_id':source_id,'source_ids':[source_id],
            'format':'native_cdft_input_observation','path':document['path'],'sha256':document['sha256'],
            'native_identity_sha256':identity,'columns':columns,'rows':rows,'row_count':len(rows),
            'row_identity_columns':['material'],
            'column_units':{'temperature_K':'K','requested_spacing_A':'angstrom',
                'actual_spacing_x_A':'angstrom','actual_spacing_y_A':'angstrom','actual_spacing_z_A':'angstrom'},
            'native_context':{'scientific_convergence':'unknown','solver_build_authenticated':False},
            'units':'column-specific; unknown remains unknown'})
        return
    if 'native_input' not in value or 'structure' not in value:
        raise ValueError('unsupported native input metadata shape')
    parsed = value['native_input']
    if not isinstance(parsed,dict):raise ValueError('invalid native String input metadata')
    groups = [('native_string_input_geometry', ['structure', 'direction', 'La', 'Lb', 'Lc', 'alpha', 'beta', 'gamma'],
        [value['structure'], parsed['direction'], *parsed['cell_values'], *parsed['angles']],
        {'La': 'unspecified', 'Lb': 'unspecified', 'Lc': 'unspecified', 'alpha': 'unspecified', 'beta': 'unspecified', 'gamma': 'unspecified'}),
        ('native_string_input_settings', ['structure', 'direction', 'temperature_K', 'running_steps', 'n_points', 'cutoff_A'],
         [value['structure'], parsed['direction'], parsed['temperature_K'], parsed['running_steps'], parsed['n_points'], parsed['cutoff_A']],
         {'temperature_K': 'K (native header)', 'running_steps': 'steps', 'n_points': 'count', 'cutoff_A': 'angstrom (native header)'})]
    current = value.get('current_cif_geometry')
    if current:
        groups.append(('current_cif_geometry', ['structure', 'direction', 'a_A', 'b_A', 'c_A', 'alpha_deg', 'beta_deg', 'gamma_deg'],
            [value['structure'], parsed['direction'], *current['lengths'], *current['angles']],
            {**{key: 'angstrom (current CIF)' for key in ('a_A', 'b_A', 'c_A')},
             **{key: 'degree (current CIF)' for key in ('alpha_deg', 'beta_deg', 'gamma_deg')}}))
    for kind, columns, values, units in groups:
        identity = _digest({'kind': kind, 'sha256': document['sha256']})
        prior = next((table for table in tables if table.get('native_identity_sha256') == identity), None)
        if prior:
            if source_id not in prior['source_ids']: prior['source_ids'].append(source_id)
            continue
        tables.append({'id': 'T' + str(len(tables) + 1), 'source_id': source_id, 'source_ids': [source_id],
            'format': kind, 'path': document['path'], 'sha256': document['sha256'],
            'native_identity_sha256': identity, 'columns': columns, 'rows': [dict(zip(columns, values))],
            'row_count': 1, 'row_identity_columns': ['structure', 'direction'], 'column_units': units,
            'native_context': {'historical_cif_snapshot_verified': False,
                'binary_source_binding': False, 'output_energy_unit': 'unknown'}, 'units': 'column-specific; unknown remains unknown'})


def restore_metadata(root, owner, state, path, raw, *, read, budget, file_facts):
    """Revalidate host metadata inherited through a verified report fragment."""
    if Path(path).parent != Path(root) / 'evidence' or not Path(path).name.startswith('report-meta-'):
        return None
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('schema') != SCHEMA or Path(path).name != 'report-meta-' + _digest(value) + '.json':
        raise ValueError('native report metadata envelope differs from its immutable identity')
    binding = value.get('binding') or {}; source = state.get('nodes', {}).get(binding.get('step_id'), {})
    ref = source.get('result_ref') or {}
    if (binding.get('owner') != owner or source.get('status') != 'succeeded'
            or source.get('token') != binding.get('token') or source.get('attempt_id') != binding.get('attempt_id')
            or ref.get('call_id') != binding.get('producer_call_id')):
        raise ValueError('native report metadata source ownership/attempt changed')
    raw_source, fact = read(root, ref['evidence_path'], budget); file_facts.append(fact)
    if fact['sha256'] != binding.get('producer_sha256'):
        raise ValueError('native report metadata producer changed')
    for item in binding.get('input_sources') or []:
        _, fact = read(root, item['path'], budget); file_facts.append(fact)
        if fact['sha256'] != item['sha256'] or fact['size'] != item['bytes']:
            raise ValueError('native report metadata original input changed')
    if value.get('role') not in {'provenance', 'native_input_evidence'} or not isinstance(value.get('public'), dict):
        raise ValueError('invalid native report metadata role')
    return {'role': value['role'], 'json_root_pointer': '/public', 'untrusted_content': value['public'],
            'native_projection': 'verified host metadata projection; original bytes retained', 'metadata_schema': SCHEMA,
            'native_ancestor_step':binding.get('step_id')}
