"""Dimension-checked unit conversion with source/value provenance, not mental math."""
import hashlib
import json
import math
from pathlib import Path


def convert_physical_units(value, quantity, from_unit, to_unit):
    path = Path(__file__).resolve().parents[1] / 'env/physical_units.json'
    raw = path.read_bytes()
    config = json.loads(raw)
    constants = config['constants']
    config['energy_J_per_particle'].update({'kJ/mol':1000/constants['avogadro_per_mol'],
        'kcal/mol':4184/constants['avogadro_per_mol'], 'eV':constants['elementary_charge_C'],
        'kelvin_energy':constants['boltzmann_J_per_K']})
    units = {'energy':'energy_J_per_particle','length':'length_m','pressure':'pressure_Pa','time':'time_s'}
    if quantity == 'temperature':
        allowed = {'K','C'}
        if from_unit not in allowed or to_unit not in allowed:
            raise ValueError('Temperature units are K or C; energy Kelvin is a different quantity.')
    else:
        table = config.get(units.get(quantity, ''), {})
        if from_unit not in table or to_unit not in table:
            raise ValueError(f'Unit/dimension mismatch: {quantity}; allowed units={list(table)}. Energy epsilon/kB uses kelvin_energy, not K.')
    values = value if isinstance(value, list) else [value]
    if not 1 <= len(values) <= 64 or any(not math.isfinite(float(v)) for v in values):
        raise ValueError('1..64 finite physical values required.')
    converted = []
    for v in values:
        v = float(v)
        if quantity == 'temperature':
            kelvin = v + 273.15 if from_unit == 'C' else v
            if kelvin < 0: raise ValueError('Negative absolute temperature is unsupported here.')
            out = kelvin - 273.15 if to_unit == 'C' else kelvin
        else:
            out = v * table[from_unit] / table[to_unit]
        if not math.isfinite(out): raise ValueError('Converted value overflowed.')
        converted.append(out)
    return {'ok':True,'read_only':True,'quantity':quantity,'input_value':value,'from_unit':from_unit,'to_unit':to_unit,
            'value':converted if isinstance(value,list) else converted[0], 'unit_definition_sha256':hashlib.sha256(raw).hexdigest(),
            'sources':config['sources'],'forcefield_conversion_ready':False,'limits':config['limits']}
