"""High-level analyst stage: LLM code at execution time -> existing OS sandbox.

No chart templates, shell fallback, scientific recalculation or graph mutation.
The approved node contains data dependencies and a visual brief, not code that
the planner must guess before seeing data.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .report_authoring import ReportStageBudgetExceeded,ReportCapabilityGap,_control,_relative,_open_parent,_read,sources_unchanged,source_receipts

from .plot_sandbox_worker import native_png_output_contract

PLOT_SCHEMA={'type':'object','required':['plot'],'properties':{'plot':{'type':'object',
    'required':['script','rationale','limitations'],
    'oneOf':[{'required':['table_ids'],'not':{'required':['input_ids']}},{'required':['input_ids'],'not':{'required':['table_ids']}}],'properties':{
        'script':{'type':'string','minLength':1,'maxLength':65536},
        'table_ids':{'type':'array','minItems':1,'maxItems':8,'items':{'type':'string'}},
        'input_ids':{'type':'array','minItems':1,'maxItems':8,'uniqueItems':True,'items':{'type':'string'}},
        'rationale':{'type':'string'},'limitations':{'type':'array','items':{'type':'string'}}}}}}


def _new_json(root,path,value):
    rel=_relative(root,path);parent,name=_open_parent(root,rel)
    temporary='.figure-'+uuid.uuid4().hex
    raw=json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
    created=False
    try:
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent);created=True
        with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False);os.fsync(parent)
    finally:
        if created:os.unlink(temporary,dir_fd=parent)
        os.close(parent)
    return {'path':str(Path(root)/rel),'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}


def _declared_units(table):
    units={}
    for column in table['columns']:
        suffixes={'_mol_kg':'mol/kg','_mol_L':'mol/L','_K':'K','_Pa':'Pa',
                  '_cm3_g':'cm^3/g','_g_cm3':'g/cm^3','_A':'angstrom'}
        for suffix,unit in suffixes.items():
            if column.endswith(suffix):units[column]=unit;break
    return units


def _publish_verified_pngs(root, output_dir, artifacts, visual, packet, control_valid):
    """Publish exact reviewed bytes at the declared directory's top level.

    The attempt remains the independently reviewed original. Content-addressed
    publication grants no visual verdict; report inheritance reopens both PNGs
    and the private visual checkpoint before accepting this byte identity.
    """
    if visual.get('verified') is not True or not visual.get('call_id'):
        raise ValueError('native visual verification required before PNG publication')
    published = []
    for artifact in artifacts:
        raw, fact = _read(root, artifact['path'], {'bytes': 0})
        reviewed = [image for image in visual.get('image_receipts', [])
            if image.get('path') == fact['path'] and image.get('sha256') == fact['sha256']
            and image.get('bytes') == fact['size']]
        if (artifact.get('kind') != 'PNG' or artifact.get('sha256') != fact['sha256']
                or artifact.get('bytes') != fact['size'] or len(reviewed) != 1):
            raise ValueError('PNG publication does not match its actual visual image receipt')
        path = Path(output_dir) / ('figure-' + fact['sha256'] + '.png')
        relative = _relative(root, path)
        parent, name = _open_parent(root, relative)
        temporary = '.figure-publish-' + uuid.uuid4().hex
        created = False
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
            created = True
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            _control(control_valid)
            if not sources_unchanged(root, packet):
                raise ValueError('figure source changed during PNG publication')
            source_raw, source_fact = _read(root, artifact['path'], {'bytes': 0})
            if source_raw != raw or source_fact['sha256'] != fact['sha256']:
                raise ValueError('reviewed PNG changed during publication')
            try:
                os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            except FileExistsError:
                existing, _ = _read(root, path, {'bytes': 0})
                if existing != raw:
                    raise ValueError('existing different PNG preserved; publication refused')
            os.fsync(parent)
        finally:
            if created: os.unlink(temporary, dir_fd=parent)
            os.close(parent)
        _control(control_valid)
        if not sources_unchanged(root, packet):
            raise ValueError('figure source changed after PNG publication')
        actual, actual_fact = _read(root, path, {'bytes': 0})
        if actual != raw:
            raise ValueError('published PNG differs from independently reviewed bytes')
        if any(item['path'] == str(path) for item in published):
            continue
        published.append({**artifact, 'path': str(path), 'bytes': actual_fact['size'],
            'sha256': actual_fact['sha256'], 'publication_schema': 'reviewed-png-publication-v1',
            'review_source': {key: reviewed[0][key] for key in ('path', 'sha256', 'bytes')}})
    return published


def author_figures(packet,output_dir,*,model_call,sandbox_call,control_valid,max_attempts=3,visual_call=None):
    if (not callable(model_call)or not callable(sandbox_call)or not callable(control_valid)
            or type(max_attempts)is not int or not 1<=max_attempts<=3):
        raise ValueError('bounded native figure capabilities are required')
    from .plot_input_adapter import catalogue
    available=catalogue(packet)
    if not available:raise ValueError('figure node lacks supported actual CSV/JSON/text inputs')
    root=Path(packet['root']);rel=_relative(root,output_dir)
    _control(control_valid)
    if not sources_unchanged(root,packet):raise ValueError('figure source changed before authoring')
    parent,name=_open_parent(root,rel,create=True)
    try:os.mkdir(name,0o700,dir_fd=parent)
    finally:os.close(parent)
    tables={t['id']:t for t in packet.get('tables',[])}
    inputs_by_id={item['id']:item for item in available};issues=[];attempts=[];previous_script=None;partial_artifacts=[];terminal_error=None
    for index in range(max_attempts):
        _control(control_valid)
        if not sources_unchanged(root,packet):raise ValueError('figure source changed before model call')
        try:
            proposed=model_call('plot',{'packet':packet,'previous_attempts':issues,
                'output_schema':PLOT_SCHEMA,'sandbox_output_contract':native_png_output_contract(),'previous_script':previous_script,
                'previous_artifacts':partial_artifacts,
                'previous_script_sha256':hashlib.sha256(previous_script.encode()).hexdigest()if previous_script else None,
                'brief':packet['instructions'],'default_backend':'python','available_inputs':available,
                'presentation_guidance':'Prefer a few simple figures, separate panels for incompatible units, readable axes and short labels. Keep full methods, provenance and long limitations in the report/caption rather than dense text inside figures. Use actual integration step values, not row counts; preserve all endpoint observations with axis margins. If repairing, simplify the previous script and resolve the concrete visual issues before adding detail.',
                'required_output':'Return plot={script,input_ids,rationale,limitations}; select actual owned source IDs from available_inputs and read their exact sandbox_path. Applicable hash-verified guidance is available in /inputs/guidance_index.json and its listed read-only sandbox paths. When native_structure is supplied, read its readonly sandbox_path JSON: it contains source-bound native parsed blocks/tables and row line anchors. Preserve separate blocks and native headers; raw original inputs remain available for audit. CSV/JSON/text original bytes stay complete. Parse unsupported inputs inside Python, never assume a CSV or invent values. Legacy table_ids selects only CSVs mapped to /inputs/tableN.csv in selected order.'})
        except (ReportStageBudgetExceeded,ReportCapabilityGap,ValueError)as error:
            _control(control_valid)
            if not sources_unchanged(root,packet):raise ValueError('figure source changed during author failure')
            if isinstance(error,ReportCapabilityGap):kind=error.details.get('error_kind','figure_vision_capability_gap');status='model_capability_gap'
            elif isinstance(error,ReportStageBudgetExceeded):kind='plot_authoring_exhausted';status='model_budget_exhausted'
            else:kind='figure_model_protocol_failed';status='model_protocol_failed'
            terminal_error={'error_kind':kind,'error':str(error)}
            issues.append({'status':status,'reason':str(error)})
            break
        _control(control_valid);plan=proposed.get('plot')if isinstance(proposed,dict)else None
        if (not isinstance(plan,dict)or proposed.get('actor')!='analyst'or not proposed.get('call_id')
                or not isinstance(plan.get('script'),str)or not 0<len(plan['script'].encode())<=65536):
            raise ValueError('analyst did not produce a native bounded Python script')
        previous_script=plan['script']
        multi=bool(plan.get('input_ids'));selected=plan.get('input_ids')if multi else plan.get('table_ids')
        if (('input_ids'in plan)==('table_ids'in plan)or not isinstance(selected,list)or not 1<=len(selected)<=8
                or any(not isinstance(t,str)for t in selected)
                or len(set(selected))!=len(selected)or not set(selected)<=set(inputs_by_id if multi else tables)
                or multi and plan.get('table_ids')):
            issues.append({'status':'invalid_input_selection','reason':'Select exactly one of input_ids or legacy table_ids; all IDs must belong to available_inputs. No sandbox execution occurred for this attempt.', 'available_input_ids':sorted(inputs_by_id),'available_table_ids':sorted(tables),'author_call_id':proposed['call_id'],'model_response_id':proposed.get('model_response_id')})
            continue
        try:compile(plan['script'],'plot.py','exec')
        except SyntaxError as error:
            issue={'status':'syntax_failed','reason':str(error)[:2000],
                   'line':error.lineno,'author_call_id':proposed['call_id']}
            attempts.append({'attempt':index+1,**issue});issues.append(issue)
            _new_json(root,root/rel/('syntax-attempt-'+str(index+1)+'.json'),
                {'script':plan['script'],'issue':issue,'model_response_id':proposed.get('model_response_id')})
            continue
        inputs=[]
        for input_id in selected:
            item=inputs_by_id[input_id];table=tables.get(input_id)
            value={'path':str(root/item['path']),'units':_declared_units(table)if table else{}}
            if table and 'method'in table['columns']:value['method_column']='method'
            if multi:value.update(input_id=input_id,format=item['format'],expected_sha256=item['sha256'])
            inputs.append(value)
        attempt_dir=root/rel/('attempt-'+str(index+1))
        params={'script':plan['script'],'input_files'if multi else'input_tables':inputs,'output_dir':str(attempt_dir),
            'timeout_seconds':30,'guidance_files':['nature-figure/static/core/contract.md',
                'nature-figure/static/core/stance.md','nature-figure/static/fragments/backend/python.md',
                'result-guidance/evidence-closure.md']}
        _control(control_valid)
        if not sources_unchanged(root,packet):raise ValueError('figure source changed before sandbox')
        result=sandbox_call(params)
        _control(control_valid)
        if not isinstance(result,dict):raise ValueError('native sandbox returned an invalid receipt')
        attempts.append({'attempt':index+1,'author_call_id':proposed['call_id'],
            'model_response_id':proposed.get('model_response_id'),'script_sha256':hashlib.sha256(plan['script'].encode()).hexdigest(),
            'status':result.get('status'),'output_dir':str(attempt_dir),
            'exit_code':result.get('exit_code'),'timed_out':result.get('timed_out')})
        if result.get('status')=='sandbox_unavailable':
            receipt={'ok':False,'status':'sandbox_unavailable','attempts':attempts,
                'error':'The required OS sandbox is unavailable; no unsandboxed fallback was executed.',
                'error_kind':'plot_sandbox_unavailable',
                'scientific_acceptance':False,'no_unsandboxed_fallback':True}
            _new_json(root,root/rel/'manifest.json',receipt);return receipt
        if result.get('ok')is True:
            artifacts=result.get('artifacts')or[]
            limits=native_png_output_contract()
            if not limits['image_count']['min']<=len(artifacts)<=limits['image_count']['max']:raise ValueError('sandbox returned no bounded images')
            from PIL import Image
            import io
            for artifact in artifacts:
                if (not isinstance(artifact,dict)or artifact.get('kind')!='PNG'
                        or not Path(artifact['path']).is_relative_to(attempt_dir)):
                    raise PermissionError('sandbox artifact escaped its approved attempt')
                raw,fact=_read(root,artifact['path'],{'bytes':0})
                if fact['sha256']!=artifact.get('sha256'):raise ValueError('sandbox image changed before handoff')
                with Image.open(io.BytesIO(raw))as image:
                    if (image.format!=limits['format']or not limits['width_px']['min']<=image.width<=limits['width_px']['max']
                            or not limits['height_px']['min']<=image.height<=limits['height_px']['max']
                            or not limits['image_bytes']['min']<=len(raw)<=limits['image_bytes']['max']):raise ValueError('sandbox artifact violates native PNG bounds')
                    image.verify()
            if not sources_unchanged(root,packet):raise ValueError('figure source changed during sandbox')
            visual={'status':'pending','reason':'no_native_image_to_provider_bridge'}
            if visual_call is not None:
                from .figure_visual_review import VISUAL_SCHEMA,source_contract,validate_review
                # The PNG has genuinely executed and passed native byte checks.
                # Capability/protocol failure must retain this exact image; it
                # never grants quality acceptance or bypasses current controls.
                partial_artifacts=artifacts
                try:
                    reviewed=visual_call('visual',{'packet':packet,'output_schema':VISUAL_SCHEMA,
                        'artifacts':artifacts,'source_contract':source_contract(packet,available,selected)})
                    _control(control_valid)
                    if not sources_unchanged(root,packet):raise ValueError('figure source changed during visual review')
                except (ReportStageBudgetExceeded,ReportCapabilityGap,ValueError)as error:
                    _control(control_valid)
                    if not sources_unchanged(root,packet):raise ValueError('figure source changed during visual failure')
                    if isinstance(error,ReportCapabilityGap):
                        kind=error.details.get('error_kind','figure_vision_capability_gap');status='visual_capability_gap'
                    elif isinstance(error,ReportStageBudgetExceeded):kind='plot_authoring_exhausted';status='model_budget_exhausted'
                    else:kind='figure_visual_protocol_failed';status='visual_protocol_failed'
                    terminal_error={'error_kind':kind,'error':str(error)}
                    issues.append({'status':status,'reason':str(error)})
                    break
                try:visual=validate_review(reviewed)
                except (ReportCapabilityGap,ValueError,TypeError,KeyError)as error:
                    _control(control_valid)
                    if not sources_unchanged(root,packet):raise ValueError('figure source changed during visual validation')
                    kind=error.details.get('error_kind','figure_vision_capability_gap')if isinstance(error,ReportCapabilityGap)else'figure_visual_protocol_failed'
                    terminal_error={'error_kind':kind,'error':str(error)}
                    issues.append({'status':'visual_capability_gap'if isinstance(error,ReportCapabilityGap)else'visual_protocol_failed','reason':str(error)})
                    break
                _new_json(root,attempt_dir/'visual-review.json',visual)
                attempts[-1]['visual_review']=visual
                if not visual['verified']:
                    partial_artifacts=artifacts
                    issues.append({'status':'visual_review_rejected','issues':visual['issues'],
                        'limitations':visual['limitations'],'script_sha256':attempts[-1]['script_sha256']})
                    continue
                visual['status']='verified'
            attempt_artifacts=artifacts
            if visual.get('verified') is True:
                artifacts=_publish_verified_pngs(root,root/rel,artifacts,visual,packet,control_valid)
            receipt={'ok':True,'status':'rendered_needs_review','backend':'python','output_dir':str(root/rel),
                'artifacts':artifacts,'attempt_artifacts':attempt_artifacts,
                'attempts':attempts,'source_packet_sha256':packet['packet_sha256'],
                'source_steps':[source['step_id']for source in packet['sources']],
                'source_receipts':source_receipts(packet),
                'source_tables':[{'table_id':key,'path':tables[key]['path'],'sha256':tables[key]['sha256']}for key in selected if key in tables],
                'source_inputs':[{key:item[key]for key in('id','path','sha256','format')}for item in available if item['id']in selected],
                'model_stage_receipts':[{'author_call_id':proposed['call_id'],'model_response_id':proposed.get('model_response_id')}],
                'plot_rationale':str(plan.get('rationale')or'')[:8000],
                'plot_limitations':plan.get('limitations')or[],
                'scientific_acceptance':False,'cacheable':False,
                'visual_review':visual,
                'sandbox_receipt':result,'no_unsandboxed_fallback':True}
            manifest=_new_json(root,root/rel/'manifest.json',receipt)
            receipt['manifest']=manifest['path'];receipt['manifest_fact']=manifest
            receipt['output_files']=[manifest['path']]+[a['path']for a in artifacts]
            return receipt
        diagnostic={}
        for log_name in ('stderr.log','launcher.stderr'):
            try:
                raw,_=_read(root,attempt_dir/log_name,{'bytes':0})
                if raw:diagnostic[log_name]=raw.decode('utf-8',errors='replace')[-4000:]
            except (OSError,ValueError):pass
        issues.append({'status':result.get('status'),'exit_code':result.get('exit_code'),
            'reason':str(result.get('error')or result.get('warnings')or'No valid image produced')[:2000],
            'native_logs':diagnostic,'output_validation':result.get('output_validation'),
            'sandbox_output_contract':native_png_output_contract()})
    receipt={'ok':False,'status':'plot_authoring_exhausted','attempts':attempts,
             'partial':bool(partial_artifacts),'partial_artifacts':partial_artifacts,
             'executed':bool(partial_artifacts or any(a.get('exit_code')is not None for a in attempts)),
             'visual_review':attempts[-1].get('visual_review')if attempts else None,
             'correction_diagnostics':issues,
             'error':'Plot authoring exhausted its bounded correction budget without accepted output; retained artifacts are partial and not scientifically accepted.',
             'error_kind':'plot_authoring_exhausted',
             'scientific_acceptance':False,'no_unsandboxed_fallback':True}
    if terminal_error:receipt.update(terminal_error)
    manifest=_new_json(root,root/rel/'manifest.json',receipt)
    receipt['manifest']=manifest['path'];receipt['manifest_fact']=manifest
    receipt['output_files']=[manifest['path']]+[a['path']for a in partial_artifacts]
    return receipt
