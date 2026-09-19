"""Read-only discovery of project-owned parameter assets, not auto-typing."""
import hashlib
import json
import math
from pathlib import Path
import re


def parse_towhee_metadata(path):
    path = Path(path)
    if path.stat().st_size > 8_000_000: raise ValueError('forcefield exceeds bounded parser size')
    raw = path.read_bytes()
    lines = raw.decode('utf-8', errors='strict').splitlines()
    def next_value(key):
        for index, line in enumerate(lines):
            if line.strip() == key: return lines[index+1].strip()
        return None
    if next_value('towhee_ff Version') != '15':
        name = path.name.removeprefix('towhee_ff_')
        return {'name': name, 'family': name, 'file': path.name, 'metadata_status': 'unsupported_format', 'sha256': hashlib.sha256(raw).hexdigest(),
                'format_version': next_value('towhee_ff Version')}
    counts = {}
    for index, line in enumerate(lines[:-1]):
        key = line.strip()
        if key.startswith('Number of ') and key != 'Number of Atoms with Same Parameters':
            value = lines[index+1].strip()
            if value.isdigit(): counts[key] = int(value)
    name = path.name.removeprefix('towhee_ff_')
    atoms = parse_atom_types(lines)
    expected = int(next_value('Number of Nonbonded Types'))
    if len(atoms) != expected: raise ValueError('incomplete nonbonded type index: ' + path.name)
    return {'name': name, 'file': path.name, 'family': 'OPLS' if name.upper().startswith('OPLS') else name,
        'representation': {'OPLS-aa': 'all_atom', 'OPLS-ua': 'united_atom'}.get(name, 'version_specific_check_required'),
        'format': 'towhee_ff', 'format_version': 15, 'potential': next_value('Potential Type'),
        'mixing_rule': next_value('Classical Mixrule'), 'counts': counts, 'nonbonded_type_count': expected,
        'elements': sorted({atom['element'] for atom in atoms if atom['element']}),
        'atom_names': sorted({value for atom in atoms for value in atom['names'].values() if value and value != 'null'}),
        'metadata_status': 'indexed_not_scientifically_assigned', 'size_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
        'charge_warning': 'Base Charge is a bond-increment base, not a complete molecular charge assignment; zero base does not prove neutral atoms.',
        'native_units': {'distance': 'angstrom', 'energy_terms': 'kelvin', 'mass': 'g/mol',
                         'coefficient_semantics': 'potential/style dependent; preserve raw values until validated conversion'},
        'official_format_documentation': 'https://towhee.sourceforge.net/towhee_ff.html'}


def parse_atom_types(lines):
    text = '\n'.join(lines)
    text = text.split('\nNumber of Bonded Terms', 1)[0]
    atoms = []
    for block in re.split(r'(?m)^Atom Type Number\s*\n', text)[1:]:
        rows = block.splitlines()
        def values(label, length=1):
            for index, line in enumerate(rows):
                if line.strip() == label: return [s.strip() for s in rows[index+1:index+1+length]]
            return []
        names = values('Atom Names', 4)
        element = values('Element')
        atom = {'type_number': int(rows[0].strip()), 'element': element[0] if element else None,
            'names': dict(zip(('nonbonded', 'bonded', 'angle', 'torsion'), names)),
            'base_charge_raw': (values('Base Charge') or [None])[0], 'nonbond_coefficients_raw': [],
            'nonbond_coefficients_numeric': []}
        if 'Nonbond Coefficients' in [s.strip() for s in rows]:
            start = next(i for i,s in enumerate(rows) if s.strip() == 'Nonbond Coefficients') + 1
            for row in rows[start:]:
                try: numeric = float(row.strip().replace('D','E').replace('d','e'))
                except ValueError: break
                if not math.isfinite(numeric): raise ValueError('non-finite forcefield coefficient')
                atom['nonbond_coefficients_raw'].append(row.strip())
                atom['nonbond_coefficients_numeric'].append(numeric)
        atoms.append(atom)
    return atoms


def load_catalog(root):
    path = Path(root) / 'forcefields/towhee/catalog.json'
    if not path.is_file(): raise ValueError('project forcefield catalog not installed')
    return json.loads(path.read_text())


def scientific_facts(root, entry):
    path = Path(root) / 'forcefields/towhee/knowledge.json'
    knowledge = json.loads(path.read_text()) if path.exists() else {}
    policy_path = Path(root) / 'env/forcefield_sources.json'
    policy = json.loads(policy_path.read_text()).get('scientific_policy', {}) if policy_path.exists() else {}
    parsed = entry.get('metadata_status') == 'indexed_not_scientifically_assigned'
    count = entry.get('counts', {}).get('Number of Bond Increments')
    return {'parsed_sections': {'nonbonded_atom_types': parsed, 'nonbonded_raw_coefficients': parsed,
        'nonbonded_numeric_coefficients': parsed, 'bonded_section_counts': parsed,
        'bonded_coefficients_and_assignment': False, 'molecular_charge_assignment': False},
        'bond_increment_count': count,
        'charge_assignment_status': 'no_increment_table_molecular_charge_source_required' if count == 0 else 'molecular_charge_source_and_increment_coverage_unverified',
        'reviewed_knowledge': knowledge.get('forcefields', {}).get(entry.get('name'), {}),
        'framework_charge_policy': knowledge.get('framework_charge_policy', {}),
        'global_limits': knowledge.get('global_limits', []), 'project_scientific_policy': policy}


def discover_forcefield(root, query, target_engine=None, required_atom_types=None, max_results=12):
    catalog = load_catalog(root)
    query = query.strip().casefold()
    knowledge_path = Path(root) / 'forcefields/towhee/knowledge.json'
    knowledge = json.loads(knowledge_path.read_text()) if knowledge_path.exists() else {}
    group = next((values for name,values in knowledge.get('groups', {}).items() if name.casefold() == query), [])
    matches = [entry for entry in catalog['entries'] if query in entry.get('name', '').casefold()
               or query == entry.get('family', '').casefold() or entry.get('name') in group]
    result = []
    for entry in matches[:max_results]:
        base = (Path(root) / 'forcefields/towhee/ForceFields').resolve()
        path = (base / entry['file']).resolve()
        if path.parent != base or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('parameter asset escaped catalog or changed since import; operator review required')
        indexed = dict(entry)
        indexed.update(scientific_facts(root, entry))
        indexed.pop('atom_names', None)
        missing = sorted(set(required_atom_types or []) - set(entry.get('atom_names', [])))
        indexed.update(asset_path=str(path), integrity_verified=True, target_engine=target_engine,
            engine_status='native_parameter_format_only' if target_engine == 'towhee' else
                          'conversion_not_implemented' if target_engine else 'engine_not_selected',
            ready_to_submit=False, missing_named_types=missing,
            coverage_status='not_checked_without_structure' if not required_atom_types else
                            'named_types_missing' if missing else 'names_present_but_chemical_assignment_unverified')
        indexed['scientific_use_limits'] = [
            'An element label or a larger atom-type count does not establish chemical-environment coverage or applicability.',
            'Towhee native parameter format does not establish an integrated Towhee execution backend or a complete simulation input.',
            'Guest molecular charge assignment and framework charge assignment are separate scientific choices; do not replace OPLS molecular charges with framework methods by default.',
            'United-atom is a specific representation, not interchangeable with arbitrary coarse-grained models.']
        if entry.get('metadata_status') == 'unsupported_format': indexed['engine_status'] = 'unparsed_legacy_format_requires_review'
        result.append(indexed)
    return {'ok': True, 'read_only': True, 'query': query, 'total_matches': len(matches), 'candidates': result,
        'selection_status': 'ambiguous_variant' if len(matches) > 1 else 'candidate_found' if matches else 'not_found',
        'scientific_checks_required': ['simulation_scope_and_engine', 'forcefield_variant', 'structure_bond_orders_and_atom_typing',
            'molecular_charge_assignment', 'bond_angle_dihedral_improper_coverage', 'mixing_and_1_4_rules', 'validated_engine_conversion'],
        'note': 'Framework defaults to UFF when unspecified. Verified missing OPLS parameters may use the user-approved UFF fallback with explicit provenance and compatibility validation; conversion absence is NOT a parameter gap. A collection is not a complete assigned model.'}


def inspect_forcefield(root, forcefield_name, atom_type_names=None, max_types=12, element=None, offset=0, name_namespace='any'):
    if max_types < 1 or max_types > 32 or offset < 0 or name_namespace not in {'any','nonbonded','bonded','angle','torsion'}:
        raise ValueError('invalid type page/namespace; use bounded offset/max_types and documented name namespace')
    entry = next((e for e in load_catalog(root)['entries'] if e.get('name') == forcefield_name), None)
    if entry is None:
        normalize_label = lambda value: re.sub(r'[^a-z0-9]', '', value.casefold())
        requested_label = normalize_label(forcefield_name)
        suggestions = [e['name'] for e in load_catalog(root)['entries']
                       if requested_label in {normalize_label(e['name']), normalize_label(e.get('file', ''))}]
        return {'error': 'unknown exact forcefield name; use metadata.name, not the asset filename',
                'suggested_exact_names': suggestions, 'next_action': 'Use the suggested exact name or discover variants first; do not guess a scientific variant.'}
    if entry.get('metadata_status') != 'indexed_not_scientifically_assigned':
        return {'ok': True, 'read_only': True, 'supported': False, 'terminal_capability_result': True,
            'resolution_required': True, 'ready_to_submit': False, 'metadata': entry,
            'next_action': 'report the known format version to main chat; request reviewed adapter/source. Repeating this parser cannot enable legacy support.'}
    path = (Path(root) / 'forcefields/towhee/ForceFields' / entry['file']).resolve()
    base = (Path(root) / 'forcefields/towhee/ForceFields').resolve()
    raw = path.read_bytes() if path.parent == base else b''
    if path.parent != base or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('parameter asset escaped catalog or changed since indexing')
    atoms = parse_atom_types(raw.decode('utf-8').splitlines())
    pool = [a for a in atoms if not element or a['element'] == element]
    def names(atom):
        return set(atom['names'].values()) if name_namespace == 'any' else {atom['names'].get(name_namespace)}
    selected = [a for a in pool if not atom_type_names or set(atom_type_names) & names(a)]
    found = {value for a in selected for value in names(a) if value and value != 'null'}
    return {'ok': True, 'read_only': True, 'asset_path': str(path), 'sha256': entry['sha256'],
        'metadata': {**{k:v for k,v in entry.items() if k != 'atom_names'}, **scientific_facts(root, entry)},
        'atom_types': selected[offset:offset+max_types],
        'matched_count': len(selected), 'missing_named_types': sorted(set(atom_type_names or []) - found),
        'offset': offset, 'next_offset': offset+max_types if offset+max_types < len(selected) else None,
        'element_filter': element, 'name_namespace': name_namespace,
        'available_nonbonded_names': [a['names'].get('nonbonded') for a in pool[offset:offset+max_types]],
        'typing_status': 'lookup_only_coordination_and_oxidation_state_not_assigned',
        'coefficient_lookup_status': 'raw_and_numeric_values_verified_from_hashed_asset_not_yet_assigned_to_user_structure',
        'ready_to_submit': False, 'note': 'Raw coefficients and exact names are evidence, not automatic molecular atom typing or final charges.'}
