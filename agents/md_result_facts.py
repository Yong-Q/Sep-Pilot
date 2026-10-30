"""Deterministic final-stage LAMMPS log facts, without executing a solver.

`ok` means the required native facts were parsed. It does not assert job
completion or physical convergence. Native tolerance stops are distinguished
from line-search/iteration stops, without choosing scientific acceptance limits.
"""
import hashlib
import math
from pathlib import Path
import re


_FIELDS = (
    ('Energy initial, next-to-last, final', ('energy_initial', 'energy_next_to_last', 'energy_final')),
    ('Force two-norm initial, final', ('force_two_norm_initial', 'force_two_norm_final')),
    ('Force max component initial, final', ('force_max_component_initial', 'force_max_component_final')),
    ('Final line search alpha, max atom move', ('line_search_alpha', 'max_atom_move')),
    ('Iterations, force evaluations', ('iterations', 'force_evaluations')),
)


def _numbers(text):
    try:
        values = [float(token) for token in text.split()]
    except ValueError:
        return None
    return values if values and all(math.isfinite(value) for value in values) else None


def parse_md_result_facts(text):
    """Parse the last minimization and its preceding same-stage thermo table."""
    lines = str(text).splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == 'Minimization stats:']
    final = {key: None for _, keys in _FIELDS for key in keys}
    final.update(line_start=None, stopping_criterion=None, native_tolerance_stop=None)
    thermo = {'header_line': None, 'last_row_line': None, 'values': {}, 'unit_style': None,
              'mpi_tasks': None}
    result = {'schema_version': 1, 'ok': False,
        'scope': 'native final-minimization facts only; not job completion or scientific acceptance',
        'source_sha256': hashlib.sha256(str(text).encode()).hexdigest(),
        'minimization_stage_count': len(starts), 'final_minimization': final,
        'final_minimization_thermo': thermo, 'subsequent_nonzero_run': False,
        'warnings': [], 'errors': []}
    if not starts:
        result['errors'].append('no native minimization statistics')
        return result
    start = starts[-1]
    result['subsequent_nonzero_run'] = any(
        int(match.group(1)) > 0 for line in lines[start + 1:]
        if (match := re.search(r'Loop time .* for\s+(\d+)\s+steps', line)))
    if result['subsequent_nonzero_run']:
        result['warnings'].append('later nonzero run exists; minimization thermo is not a final dynamics state')
    final['line_start'] = start + 1
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip() == 'MPI task timing breakdown:'), len(lines))
    block = lines[start + 1:end]
    for line in block:
        stop = re.match(r'\s*Stopping criterion\s*=\s*(.+)', line)
        if stop:
            final['stopping_criterion'] = stop.group(1).strip()
            final['native_tolerance_stop'] = final['stopping_criterion'].casefold() in {
                'energy tolerance', 'force tolerance'}
            break
    if final['stopping_criterion'] is None:
        result['errors'].append('missing final stopping criterion')
    elif not final['native_tolerance_stop']:
        result['warnings'].append('final native stopping criterion is not a tolerance stop')
    for label, keys in _FIELDS:
        pattern = r'\s*' + re.escape(label).replace(r'\ ', r'\s+') + r'\s*=\s*(.*)$'
        values = None
        for index, line in enumerate(block):
            match = re.match(pattern, line)
            if match:
                raw = match.group(1) or next((value.strip() for value in block[index + 1:] if value.strip()), '')
                values = _numbers(raw)
                break
        if values is None or len(values) != len(keys):
            result['errors'].append('missing or nonfinite final ' + label)
            continue
        if keys[0] == 'iterations':
            if any(value < 0 or not value.is_integer() for value in values):
                result['errors'].append('invalid final iteration/evaluation counts')
                continue
            values = list(map(int, values))
        final.update(zip(keys, values))

    previous_stats = starts[-2] if len(starts) > 1 else -1
    setups = [i for i in range(previous_stats + 1, start)
              if re.search(r'Setting up .+ minimization', lines[i])]
    if setups:
        stage_start = setups[-1]
        headers = [i for i in range(stage_start, start)
                   if lines[i].split() and lines[i].split()[0] == 'Step']
        for line in lines[stage_start:start]:
            unit = re.match(r'\s*Unit style\s*:\s*(\w+)', line)
            if unit:
                thermo['unit_style'] = unit.group(1)
            parallel = re.search(r'(\d+) MPI tasks', line)
            if parallel:
                thermo['mpi_tasks'] = int(parallel.group(1))
        if headers:
            header = headers[-1]
            columns = lines[header].split()
            thermo['header_line'] = header + 1
            for index in range(header + 1, start):
                if lines[index].lstrip().startswith('Loop time of'):
                    break
                tokens = lines[index].split()
                if len(tokens) != len(columns):
                    continue
                try:
                    row = [float(token) for token in tokens]
                except ValueError:
                    continue
                # Preserve the final row even when invalid: do not replace NaN
                # with an earlier finite row and mislabel it as the final state.
                thermo['last_row_line'] = index + 1
                thermo['values'] = dict(zip(columns, [value if math.isfinite(value) else None for value in row]))
    if not thermo['values']:
        result['errors'].append('same-stage final minimization thermo table unavailable')
    elif any(value is None for value in thermo['values'].values()):
        result['errors'].append('nonfinite final minimization thermo row')
    if thermo['unit_style'] is None:
        result['warnings'].append('native unit style unavailable; do not infer physical units')
    result['ok'] = not result['errors']
    return result


def read_md_result_facts(path, max_bytes=4_000_000):
    """Read a bounded existing log. Never rewrites logs or scientific outputs."""
    source = Path(path).expanduser().resolve()
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        return {'ok': False, 'source_path': str(source), 'errors': ['parse size bound must be a positive integer']}
    try:
        with source.open('rb') as stream:
            raw = stream.read(max_bytes + 1)
    except OSError:
        return {'ok': False, 'source_path': str(source), 'errors': ['native MD log unavailable']}
    if len(raw) > max_bytes:
        return {'ok': False, 'source_path': str(source), 'errors': ['native MD log exceeds parse size bound']}
    result = parse_md_result_facts(raw.decode('utf-8', errors='replace'))
    result.update(source_path=str(source), source_bytes=len(raw), source_sha256=hashlib.sha256(raw).hexdigest())
    return result


# Additive complete-history reader. The existing final_minimization API above
# retains its documented narrow scope and exact behavior.
THERMO_HISTORY_VERSION = 'native-lammps-blocks-v1'
UNIT_CONVENTION_SOURCE = 'https://docs.lammps.org/units.html'


def _thermo_units(style, headers):
    systems = {'real': ('kcal/mol', 'K', 'atm', 'angstrom', 'fs'),
               'metal': ('eV', 'K', 'bar', 'angstrom', 'ps'),
               'si': ('J', 'K', 'Pa', 'm', 's'),
               'lj': ('reduced', 'reduced', 'reduced', 'reduced', 'reduced')}
    system = systems.get(style)
    energies = {'TotEng', 'PotEng', 'KinEng', 'E_pair', 'E_mol', 'E_bond', 'E_angle',
                'E_dihed', 'E_impro', 'E_coul', 'E_long', 'E_tail', 'Enthalpy'}
    result = {}
    for name in headers:
        unit = None
        if name in {'Step', 'Atoms', 'Elapsed', 'Elaplong'}: unit = 'count'
        elif system:
            energy, temperature, pressure, length, time = system
            if name in energies: unit = energy
            elif name == 'Temp': unit = temperature
            elif name == 'Press' or re.fullmatch(r'P[xyz][xyz]', name): unit = pressure
            elif name in {'Lx', 'Ly', 'Lz', 'Xlo', 'Xhi', 'Ylo', 'Yhi', 'Zlo', 'Zhi', 'Xy', 'Xz', 'Yz'}: unit = length
            elif name in {'Volume', 'Vol'}: unit = length + '^3'
            elif name in {'Time', 'Dt'}: unit = time
        result[name] = unit
    return result


def parse_md_thermo(raw, *, max_bytes=4_000_000, max_blocks=128, max_rows=20000):
    """Read complete independent native executions; never join their trajectories.

    Rows and statistics retain native line anchors. Step ranges/row counts are
    not integration counts: only exact native Loop summaries supply those.
    Nonfinite/invalid cells remain null with errors; no earlier row substitutes.
    """
    import json
    if not isinstance(raw, bytes): raise TypeError('native thermo reader requires original bytes')
    for value in (max_bytes, max_blocks, max_rows):
        if type(value) is not int or value < 1: raise ValueError('positive finite parser bounds required')
    if len(raw) > max_bytes: raise ValueError('native thermo source exceeds byte bound')
    text = raw.decode('utf-8-sig')
    if any(c in text for c in('\x85','\u2028','\u2029')):raise ValueError('unsupported non-native Unicode line separator')
    if '\x00' in text: raise ValueError('native thermo source contains NUL bytes')
    lines = text.splitlines(); raw_lines = raw.splitlines()
    result = {'schema_version': 1, 'format': THERMO_HISTORY_VERSION,
        'source_sha256': hashlib.sha256(raw).hexdigest(), 'source_bytes': len(raw),
        'line_count': len(lines), 'parser_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'blocks': [], 'stage_markers': [], 'unassigned_messages': [], 'errors': [],
        'scientific_acceptance': False, 'convergence': 'unknown', 'replicate_n': None,
        'unit_convention_source': UNIT_CONVENTION_SOURCE,
        'normalization': 'unknown; native column units do not establish per-atom or replicate normalization',
        'scope': 'Each block is independent. Never concatenate blocks or infer integration steps from row counts/Step differences. Native tolerance stops do not establish dynamics convergence.'}
    def anchor(index):
        return {'line': index + 1, 'line_sha256': hashlib.sha256(raw_lines[index]).hexdigest(), 'text': lines[index]}
    current = None; active_table = None; stage = None; unit_style = None; unit_line = None; row_total = 0
    def begin(index, kind, setup=None):
        if len(result['blocks']) >= max_blocks: raise ValueError('native thermo execution block bound exceeded')
        block = {'id': 'B' + str(len(result['blocks']) + 1), 'kind': kind,
            'setup': setup, 'stage': stage['name'] if stage else None,
            'stage_marker_sequence': stage['sequence'] if stage else None,
            'line_start': index + 1, 'line_end': None, 'unit_style': unit_style,
            'unit_source': anchor(unit_line) if unit_line is not None else None,
            'tables': [], 'loop_summaries': [], 'native_steps': None, 'native_mpi_tasks': None,
            'native_minimization': None, 'stats_field_anchors': {}, 'messages': [],
            'current_step': None, 'timestep': None, 'complete': False, 'errors': [],
            'convergence': 'unknown', 'continuity_to_other_blocks': 'not_established'}
        result['blocks'].append(block)
        if stage: stage['block_ids'].append(block['id'])
        return block
    def finish(end):
        nonlocal current, active_table
        if current is None: return
        current['line_end'] = end
        start = current['line_start'] - 1
        current['source_span_sha256'] = hashlib.sha256(b'\n'.join(raw_lines[start:end])).hexdigest()
        if current['kind'] == 'minimization':
            # Reuse the tested native final-stage parser on this one segment;
            # the segment cannot overwrite any other minimization in history.
            facts = parse_md_result_facts('\n'.join(lines[start:end]))
            stats = dict(facts['final_minimization'])
            if stats.get('line_start') is not None: stats['line_start'] += start
            current['native_minimization'] = stats
            current['errors'].extend(facts['errors'])
            stats_start = stats.get('line_start')
            if stats_start:
                for index in range(stats_start, end):
                    for label, keys in _FIELDS:
                        if re.match(r'\s*' + re.escape(label).replace(r'\ ', r'\s+') + r'\s*=', lines[index]):
                            refs = [anchor(index)]
                            if not lines[index].split('=', 1)[1].strip():
                                value_line = next((i for i in range(index + 1, end) if lines[i].strip()), None)
                                if value_line is not None: refs.append(anchor(value_line))
                            for key in keys: current['stats_field_anchors'][key] = refs
                    if re.match(r'\s*Stopping criterion\s*=', lines[index]):
                        current['stats_field_anchors']['stopping_criterion'] = [anchor(index)]
        loops = current['loop_summaries']
        if len(loops) == 1:
            current['native_steps'] = loops[0]['steps']; current['native_mpi_tasks'] = loops[0]['mpi_tasks']
        elif not loops: current['errors'].append('native_loop_summary_missing')
        else: current['errors'].append('multiple_loop_summaries_in_one_execution; steps remain unknown')
        if not current['tables'] or not any(t['rows'] for t in current['tables']):
            current['errors'].append('native_thermo_rows_missing')
        for table in current['tables']:
            table['row_count']=len(table['rows'])
            table['rows_complete']=not table['rejected_rows']
            table['rows_preview']=False
            if not table['rows']: current['errors'].append('empty_native_thermo_table')
            current['errors'].extend(table['errors'])
        current['complete'] = not current['errors']
        current = None; active_table = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        marker = re.fullmatch(r'SEPPILOT_STAGE (\S+) (begin|end)', stripped)
        if marker:
            if marker[2] == 'begin':
                finish(index)
                if stage:
                    stage['errors'].append('missing_matching_end_marker')
                    result['errors'].append('nested_or_unclosed_stage_marker')
                stage = {'name': marker[1], 'sequence': len(result['stage_markers']) + 1,
                    'begin': anchor(index), 'end': None, 'block_ids': [], 'errors': []}
                result['stage_markers'].append(stage)
            else:
                finish(index)
                if stage and stage['name'] == marker[1]: stage['end'] = anchor(index); stage = None
                else: result['errors'].append('unmatched_stage_end_at_line_' + str(index + 1))
            continue
        setup = re.match(r'\s*Setting up (.+?)(?:\s+\.\.\.)?\s*$', line)
        if setup and ('minimization' in setup[1].lower() or 'run' in setup[1].lower()):
            finish(index);kind = 'minimization' if 'minimization' in setup[1].lower() else 'run'
            current = begin(index, kind, anchor(index));continue
        if re.match(r'\s*reset_timestep\s+', line):
            finish(index);continue
        unit = re.match(r'\s*Unit style\s*:\s*(\w+)\s*$', line)
        if unit:
            unit_style, unit_line = unit[1], index
            if current:
                if current['tables'] and current['unit_style'] != unit_style: current['errors'].append('unit_style_changed_inside_execution')
                current['unit_style'] = unit_style;current['unit_source'] = anchor(index)
            continue
        if current:
            step = re.match(r'\s*Current step\s*:\s*([-+]?\d+)\s*$', line)
            dt = re.match(r'\s*Time step\s*:\s*(\S+)\s*$', line)
            if step: current['current_step'] = {'value': int(step[1]), 'source': anchor(index)}
            if dt:
                values = _numbers(dt[1]);current['timestep'] = {'value': values[0] if values and len(values) == 1 else None, 'source': anchor(index)}
        if stripped.startswith(('WARNING:', 'ERROR:', 'ERROR ')):
            message = {'kind': 'error' if stripped.startswith('ERROR') else 'warning', 'source': anchor(index)}
            (current['messages'] if current else result['unassigned_messages']).append(message)
            if current and message['kind'] == 'error': current['errors'].append('native_solver_error_at_line_' + str(index + 1))
            continue
        tokens = stripped.split()
        if (tokens and tokens[0] in {'Step', 'Temp', 'Time', 'Elapsed', 'CPU', 'TotEng'} and 2 <= len(tokens) <= 256
                and all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_\[\]()./+-]*', token) for token in tokens)):
            if current is None: current = begin(index, 'unknown')
            active_table = {'id': current['id'] + '.T' + str(len(current['tables']) + 1),
                'headers': tokens, 'header_source': anchor(index), 'unit_style': unit_style,
                'unit_source': anchor(unit_line) if unit_line is not None else None,
                'column_units': _thermo_units(unit_style, tokens), 'rows': [], 'rejected_rows': [], 'errors': [],
                'step_monotonic': None, 'row_count_is_not_integration_steps_or_replicates': True}
            if len(set(tokens)) != len(tokens): active_table['errors'].append('duplicate_native_thermo_headers')
            current['tables'].append(active_table);continue
        loop = re.match(r'\s*Loop time of .*? on (\d+) procs for (\d+) steps(?: with (\d+) atoms)?', line)
        if loop:
            if current is None: current = begin(index, 'unknown')
            current['loop_summaries'].append({'mpi_tasks': int(loop[1]), 'steps': int(loop[2]),
                'atoms': int(loop[3]) if loop[3] else None, 'source': anchor(index)})
            active_table = None;continue
        if stripped == 'Minimization stats:':
            if current is None: current = begin(index, 'minimization')
            elif current['kind'] == 'unknown': current['kind'] = 'minimization'
            active_table = None;continue
        if active_table and tokens:
            try: float(tokens[0])
            except ValueError: continue
            if len(tokens) != len(active_table['headers']):
                active_table['errors'].append('thermo_row_width_mismatch_at_line_' + str(index + 1))
                active_table['rejected_rows'].append({'source':anchor(index),'raw_tokens':tokens,'error':'row_width_mismatch'});continue
            values=[];invalid=[];nonfinite=[]
            for column, token in enumerate(tokens):
                try: value=float(token)
                except ValueError: value=None;invalid.append(column)
                if value is not None and not math.isfinite(value): value=None;nonfinite.append(column)
                values.append(value)
            row_total += 1
            if row_total > max_rows: raise ValueError('native thermo row bound exceeded')
            row={'source':anchor(index),'values':values}
            if invalid: row['invalid_columns']=invalid;active_table['errors'].append('invalid_thermo_cell_at_line_' + str(index + 1))
            if nonfinite: row['nonfinite_columns']=nonfinite;active_table['errors'].append('nonfinite_thermo_cell_at_line_' + str(index + 1))
            active_table['rows'].append(row)
    finish(len(lines))
    if stage:
        stage['errors'].append('missing_matching_end_marker');result['errors'].append('unclosed_stage_marker')
    for block in result['blocks']:
        for table in block['tables']:
            if 'Step' in table['headers']:
                column=table['headers'].index('Step');values=[row['values'][column] for row in table['rows']]
                table['step_monotonic']=None if any(v is None for v in values) else all(b >= a for a,b in zip(values,values[1:]))
                if table['step_monotonic'] is False: table['continuity_warning']='native_step_reset_or_nonmonotonic_order; do not infer continuous trajectory'
    result['ok']=bool(result['blocks']) and not result['errors'] and all(block['complete'] for block in result['blocks'])
    result['parse_status']='complete' if result['ok'] else 'partial' if result['blocks'] else 'no_supported_thermo_blocks'
    # Verify strict finite JSON serialization now, before the object is handed to
    # any model or mounted as an input. NaN tokens are explicit null/errors.
    json.dumps(result,allow_nan=False)
    return result


def verify_md_thermo(raw, history):
    """Reject changed source bytes or modified blocks, values, units and anchors."""
    import json
    if not isinstance(history,dict) or history.get('source_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('native thermo source SHA mismatch')
    actual=parse_md_thermo(raw)
    if json.dumps(history,sort_keys=True,allow_nan=False) != json.dumps(actual,sort_keys=True,allow_nan=False):
        raise ValueError('native thermo history/row/block/source anchors changed')
    return True


def md_thermo_summary(history):
    """Bounded model view; full rows remain in the readonly native JSON input."""
    blocks=[]
    for block in history['blocks']:
        item={k:block[k] for k in('id','kind','stage','stage_marker_sequence','line_start','line_end',
            'unit_style','native_steps','native_mpi_tasks','native_minimization','complete','errors','convergence')}
        item['tables']=[{'id':t['id'],'headers':t['headers'],'column_units':t['column_units'],
            'header_source':t['header_source'],'row_count':len(t['rows']),
            'rows':t['rows'][:1]+t['rows'][-1:] if len(t['rows'])>1 else t['rows'],
            'rows_complete':len(t['rows'])<=2 and not t['rejected_rows'],'rows_preview':len(t['rows'])>2,
            'errors':t['errors'],'step_monotonic':t['step_monotonic']}for t in block['tables']]
        blocks.append(item)
    return {k:history[k] for k in('format','source_sha256','source_bytes','parser_sha256','parse_status','scientific_acceptance','convergence','normalization','scope')}|{'blocks':blocks,'errors':history['errors']}
