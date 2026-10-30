"""Native source-bound CIF descriptors for the existing ML feature contract.

Elemental heuristic definitions and legacy ``to_row`` names are retained.
Volume uses the shared full triclinic metric. Atom sites are read by ASE, never
by counting arbitrary lines. Partial/disordered or non-P1 inputs are rejected
explicitly; no dominant-element choice or silent symmetry expansion occurs.
This is descriptor extraction, not chemical/force-field validation.
"""
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import io
import math
import os
from pathlib import Path
import re
from typing import Dict, List, Optional
import warnings

from .gcmc_geometry import derive_gcmc_geometry
from .raspa_assets import _read
from .raspa_cif_normalization import normalize_raspa_cif_bytes

DESCRIPTOR_VERSION = 'mof-listed-sites-v2-triclinic-ase-p1'
MAX_CIF_BYTES = 8 * 1024 * 1024
MAX_ATOM_SITES = 100000
ELEMENT_FEATURES = ('H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'V', 'Cu', 'Zn', 'Zr')
NUMERIC_FEATURES = ELEMENT_FEATURES + ('volume_A3', 'total_unsaturation', 'metallic_pct',
    'en_ratio', 'weighted_en', 'n_o_ratio')
FEATURE_ALIASES = {'cell_volume_A3': 'volume_A3'}
_METALS = frozenset('Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf'.split())
# Historical categorical order, with its repeated Zr entry removed.
_METAL_TYPES = tuple(dict.fromkeys(('V', 'Cu', 'Zn', 'Zr', 'Fe', 'Co', 'Ni', 'Mn', 'Cr', 'Ti', 'Al', 'Mg', 'Zr')))
_ELECTRONEG = {'O': 3.44, 'N': 3.04, 'F': 3.98, 'Cl': 3.16, 'Br': 2.96}
_CELL_KEYS = ('_cell_length_a', '_cell_length_b', '_cell_length_c',
    '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma')


class _ExtractionError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass
class MOFFeatures:
    cif_name: str
    atom_counts: Dict[str, int] = field(default_factory=dict)
    volume_A3: Optional[float] = None
    metal_types: List[str] = field(default_factory=list)
    total_unsaturation: Optional[float] = None
    metallic_pct: Optional[float] = None
    en_ratio: Optional[float] = None
    weighted_en: Optional[float] = None
    n_o_ratio: Optional[float] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    source_sha256: Optional[str] = None
    source_size_bytes: Optional[int] = None
    descriptor_version: str = DESCRIPTOR_VERSION
    atom_count: Optional[int] = None
    atom_count_scope: str = 'listed_full_occupancy_atom_sites; no symmetry expansion'
    symmetry_scope: Optional[str] = None
    occupancy_scope: Optional[str] = None
    cell_lengths_A: List[float] = field(default_factory=list)
    cell_angles_deg: List[float] = field(default_factory=list)
    parser_warnings: List[str] = field(default_factory=list)
    source_verified_unchanged: bool = False
    scientific_acceptance: bool = False

    def get(self, element: str) -> int:
        return self.atom_counts.get(element, 0)

    def to_row(self, *, include_provenance=False) -> dict:
        """Legacy CSV column names/order; provenance is explicitly opt-in.

        A failed extraction retains its explicit error row for batch reporting.
        Prediction callers MUST reject ``error`` rather than train/predict from
        the absent-element zero defaults of a failed descriptor object.
        """
        row = {'cif_name': self.cif_name}
        row.update({element: self.get(element) for element in ELEMENT_FEATURES})
        row.update(metal_types=';'.join(self.metal_types), volume_A3=self.volume_A3,
            total_unsaturation=self.total_unsaturation, metallic_pct=self.metallic_pct,
            en_ratio=self.en_ratio, weighted_en=self.weighted_en,
            n_o_ratio=self.n_o_ratio, error=self.error or '')
        if include_provenance:
            row.update(source_sha256=self.source_sha256, source_size_bytes=self.source_size_bytes,
                descriptor_version=self.descriptor_version, atom_count_scope=self.atom_count_scope,
                symmetry_scope=self.symmetry_scope, source_verified_unchanged=self.source_verified_unchanged)
        return row

    def feature_values(self, names, *, allow_aliases=False):
        """Select exact trained columns in the requested order, never position.

        Missing O leaves n_o_ratio=None for a TRAIN-fitted imputer. Unknown
        names and metadata/categorical fields are errors, never zero features.
        """
        if self.error or not self.source_verified_unchanged:
            raise ValueError('CIF feature extraction failed; prediction is not permitted')
        if not isinstance(names, (list, tuple)) or not names or len(set(names)) != len(names):
            raise ValueError('unique trained feature names are required')
        keys = [FEATURE_ALIASES.get(name, name) if allow_aliases else name for name in names]
        if len(set(keys)) != len(keys) or any(key not in NUMERIC_FEATURES for key in keys):
            raise ValueError('unknown, duplicate alias or nonnumeric CIF feature')
        row = self.to_row()
        return [row[key] for key in keys]


def _finite_column(block, key, size):
    values = block.get(key)
    if values is None or not isinstance(values, (list, tuple)) or len(values) != size:
        raise _ExtractionError('invalid_atom_sites', 'atom-site coordinate columns must have equal complete row counts')
    try:
        values = [float(value) if not isinstance(value, bool) else math.nan for value in values]
    except (ValueError, TypeError, OverflowError):
        raise _ExtractionError('invalid_atom_sites', 'atom-site coordinates must be finite numbers')
    if not all(math.isfinite(value) for value in values):
        raise _ExtractionError('invalid_atom_sites', 'atom-site coordinates must be finite numbers')
    return values


def _symmetry_scope(block, np):
    declared = False
    for key in ('_space_group_it_number', '_symmetry_int_tables_number'):
        value = block.get(key)
        if value in (None, '.', '?'): continue
        declared = True
        try: number = float(value)
        except (ValueError, TypeError): number = math.nan
        if number != 1:
            raise _ExtractionError('unsupported_symmetry', 'non-P1 or unknown symmetry must be standardized explicitly; listed-site descriptors are not silently expanded')
    for key in ('_space_group_name_h-m_alt', '_symmetry_space_group_name_h-m',
                '_space_group_name_hall', '_symmetry_space_group_name_hall'):
        value = block.get(key)
        if value in (None, '.', '?'): continue
        declared = True
        if not isinstance(value, str) or re.sub(r'[\s_]', '', value).upper() != 'P1':
            raise _ExtractionError('unsupported_symmetry', 'non-P1 or contradictory space-group names require explicit structure standardization')
    for key in ('_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz'):
        values = block.get(key)
        if values is None: continue
        declared = True
        if isinstance(values, str): values = [values]
        if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 192:
            raise _ExtractionError('unsupported_symmetry', 'invalid or excessive CIF symmetry operations')
        try:
            from ase.spacegroup.spacegroup import parse_sitesym
            rotations, translations = parse_sitesym([value.replace(' ', '').lower() for value in values])
        except (ValueError, TypeError, AttributeError, IndexError, ZeroDivisionError):
            raise _ExtractionError('unsupported_symmetry', 'CIF symmetry operations are invalid')
        if not (np.array_equal(rotations, np.repeat(np.eye(3, dtype=int)[None, :, :], len(values), axis=0))
                and np.allclose(translations, np.rint(translations), rtol=0, atol=1e-12)):
            raise _ExtractionError('unsupported_symmetry', 'nonidentity symmetry must be standardized explicitly; no dominant-element or silent expansion fallback')
    return 'explicit_P1_identity' if declared else 'symmetry_undeclared; listed-site scope only'


def extract_features(cif_path, *, owned_root=None, expected_sha256=None) -> MOFFeatures:
    """Read one bounded stable CIF, preserving legacy listed-site definitions.

    Unsupported occupancy, disorder and nontrivial symmetry return an explicit
    failed feature object. Original bytes are never rewritten or normalized.
    """
    try:
        path = Path(os.path.abspath(cif_path))
    except (TypeError, ValueError, OSError):
        return MOFFeatures(cif_name='', error_code='invalid_source_path', error='invalid_source_path: one literal CIF file path is required')
    feature = MOFFeatures(cif_name=path.stem)
    try:
        if expected_sha256 is not None and (not isinstance(expected_sha256, str)
                or not re.fullmatch(r'[a-f0-9]{64}', expected_sha256)):
            raise _ExtractionError('invalid_expected_sha256', 'expected_sha256 must be an exact lowercase SHA-256 digest')
        if owned_root is not None:
            root = Path(os.path.abspath(owned_root))
            if not path.is_relative_to(root) or path == root:
                raise _ExtractionError('source_outside_scope', 'CIF source is outside the supplied owner root')
        raw, source = _read(path)
        feature.source_sha256 = source['sha256']; feature.source_size_bytes = source['size']
        if source['size'] > MAX_CIF_BYTES:
            raise _ExtractionError('source_too_large', 'CIF exceeds the bounded source size')
        if expected_sha256 is not None and source['sha256'] != expected_sha256:
            raise _ExtractionError('source_sha256_mismatch', 'CIF source differs from the supplied immutable input digest')
        # The shared CIF tokenizer validates full loop cardinality. ASE may
        # otherwise warn and discard malformed atom rows, which is not a valid
        # successful composition extraction. Aliased bytes are not published.
        try: normalize_raspa_cif_bytes(raw)
        except (ValueError, UnicodeError):
            raise _ExtractionError('invalid_cif_syntax', 'CIF fields or loop rows are incomplete or ambiguous')
        import numpy as np
        from ase.io.cif import parse_cif
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            blocks = list(parse_cif(io.BytesIO(raw)))
            cells = [block for block in blocks if any(block.get(key) is not None for key in _CELL_KEYS)]
            if len(cells) != 1:
                raise _ExtractionError('ambiguous_cell', 'exactly one structural CIF cell block is required')
            block = cells[0]
            if any(any(token in str(item.message).lower() for token in ('duplicat', 'wrong number', 'discard', 'badly formed')) for item in caught):
                raise _ExtractionError('invalid_cif_syntax', 'duplicate CIF tags or atom-site definitions are ambiguous')
            geometry = derive_gcmc_geometry(path, 1.0, owned_root=owned_root)
            if geometry.get('ok') is not True:
                raise _ExtractionError('invalid_cell', 'CIF requires six finite, nondegenerate cell lengths/angles')
            if geometry.get('source_sha256') != source['sha256']:
                raise _ExtractionError('source_changed', 'CIF source changed during metric extraction')
            symmetry = _symmetry_scope(block, np)
            raw_sites = block.get('_atom_site_type_symbol')
            if raw_sites is None: raw_sites = block.get('_atom_site_label')
            if not isinstance(raw_sites, (list, tuple)) or not 1 <= len(raw_sites) <= MAX_ATOM_SITES:
                raise _ExtractionError('invalid_atom_sites', 'one bounded complete atom-site loop is required')
            # ASE reads only the real atom-site columns and their chemical
            # symbols. It does not expand symmetry or collapse mixed occupancy.
            atoms = block.get_unsymmetrized_structure()
        feature.parser_warnings = [str(item.message)[:240] for item in caught]
        if any(any(token in item.lower() for token in ('duplicat', 'wrong number', 'discard', 'badly formed')) for item in feature.parser_warnings):
            raise _ExtractionError('invalid_atom_sites', 'duplicate CIF tags or atom-site definitions are ambiguous')
        count = len(atoms)
        if not 1 <= count <= MAX_ATOM_SITES:
            raise _ExtractionError('invalid_atom_count', 'CIF atom-site count is empty or exceeds the supported bound')
        symbols = atoms.get_chemical_symbols()
        raw_symbols = block.get('_atom_site_type_symbol')
        if raw_symbols is None: raw_symbols = block.get('_atom_site_label')
        if not isinstance(raw_symbols, (list, tuple)) or len(raw_symbols) != count:
            raise _ExtractionError('invalid_atom_sites', 'CIF atom-site symbols must be one complete site loop')
        for original, symbol in zip(raw_symbols, symbols):
            prefix = 'D' if isinstance(original, str) and original.startswith('D') and symbol == 'H' else symbol
            if not isinstance(original, str) or not re.match(r'^' + re.escape(prefix) + r'(?=$|[^A-Za-z])', original):
                raise _ExtractionError('invalid_atom_symbols', 'CIF atom-site symbols must have an unambiguous chemical-element prefix')
        occupations = block.get('_atom_site_occupancy')
        if occupations is not None:
            values = _finite_column(block, '_atom_site_occupancy', count)
            if not all(abs(value - 1.0) <= 1e-12 for value in values):
                raise _ExtractionError('unsupported_occupancy', 'partial, zero or nonunit occupancy requires an explicitly resolved structure; no dominant species is selected')
        for key in ('_atom_site_disorder_group', '_atom_site_disorder_assembly'):
            values = block.get(key)
            if values is not None and (not isinstance(values, (list, tuple)) or len(values) != count
                    or any(str(value) not in {'.', '?', '0', '0.0'} for value in values)):
                raise _ExtractionError('unsupported_disorder', 'declared disordered atom sites require an explicitly resolved structure')
        multiplicities = block.get('_atom_site_symmetry_multiplicity')
        if multiplicities is not None and any(value != 1.0 for value in _finite_column(block, '_atom_site_symmetry_multiplicity', count)):
            raise _ExtractionError('unsupported_symmetry', 'nonunit site multiplicities cannot be treated as a full P1 site list')
        fractional = ('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z')
        cartesian = ('_atom_site_cartn_x', '_atom_site_cartn_y', '_atom_site_cartn_z')
        has_fractional = any(block.get(key) is not None for key in fractional)
        has_cartesian = any(block.get(key) is not None for key in cartesian)
        if has_fractional:
            fractional_values = np.array([_finite_column(block, key, count) for key in fractional]).T
        if has_cartesian:
            cartesian_values = np.array([_finite_column(block, key, count) for key in cartesian]).T
        if not has_fractional and not has_cartesian:
            raise _ExtractionError('invalid_atom_sites', 'CIF lacks complete atom-site coordinates')
        if has_fractional and has_cartesian and not np.allclose(
                fractional_values @ np.asarray(atoms.cell), cartesian_values, rtol=0, atol=1e-7):
            raise _ExtractionError('ambiguous_coordinates', 'fractional and Cartesian coordinates disagree')
        coordinates = atoms.get_scaled_positions(wrap=True)
        if not np.isfinite(coordinates).all() or not np.isfinite(atoms.positions).all():
            raise _ExtractionError('invalid_atom_sites', 'CIF coordinates are nonfinite')
        keys = [tuple(round(float(value), 10) for value in row) for row in coordinates]
        if len(set(keys)) != count:
            raise _ExtractionError('duplicate_atom_sites', 'coincident periodic atom sites are ambiguous; no dominant element is selected')
        if not math.isclose(float(atoms.get_volume()), geometry['volume'], rel_tol=1e-10, abs_tol=1e-9):
            raise _ExtractionError('cell_metric_mismatch', 'ASE cell and native triclinic metric disagree')
        again, current = _read(path)
        if current != source or again != raw:
            raise _ExtractionError('source_changed', 'CIF source changed before descriptor publication')
        counts = dict(sorted(Counter(symbols).items())); total = sum(counts.values())
        carbon = counts.get('C', 0)
        h_equivalent = counts.get('H', 0) + counts.get('N', 0) / 2
        h_equivalent += sum(counts.get(element, 0) for element in ('F', 'Cl', 'Br', 'I'))
        feature.atom_counts = counts; feature.atom_count = count
        feature.volume_A3 = float(geometry['volume'])
        feature.cell_lengths_A = geometry['lengths']; feature.cell_angles_deg = geometry['angles']
        feature.metal_types = [metal for metal in _METAL_TYPES if metal in counts]
        feature.total_unsaturation = (2 * carbon + 2 - h_equivalent) / 2
        feature.metallic_pct = sum(value for element, value in counts.items() if element in _METALS) / carbon * 100 if carbon else 0.0
        feature.en_ratio = sum(counts.get(element, 0) for element in _ELECTRONEG) / total
        feature.weighted_en = sum(counts.get(element, 0) * value for element, value in _ELECTRONEG.items()) / total
        feature.n_o_ratio = counts.get('N', 0) / counts['O'] if counts.get('O') else None
        feature.symmetry_scope = symmetry
        feature.occupancy_scope = 'explicit_unit_occupancy' if occupations is not None else 'CIF_default_unit_occupancy'
        feature.source_verified_unchanged = True
        return feature
    except _ExtractionError as error:
        feature.error_code = error.code; feature.error = error.code + ': ' + str(error)
    except (ValueError, TypeError, OSError, UnicodeError, AssertionError, IndexError, KeyError, AttributeError, OverflowError) as error:
        feature.error_code = 'invalid_cif_source'
        feature.error = 'invalid_cif_source: CIF syntax, atom-site symbols/coordinates or stable regular source access within 1..8 MiB could not be verified'
    return feature
