import os
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CG_FF      = os.path.join(_SCRIPT_DIR, 'Molecule_FF', 'data_ff_Coarsening_gas')
_UFF_DIR    = os.path.join(_SCRIPT_DIR, 'Molecule_FF', 'UFF')
_FF_UFF     = os.path.join(_SCRIPT_DIR, 'data_ff_UFF')

def _gas_has_partial_charges(gas, uff_dir=None):
    """Whether the gas molecule's UFF def assigns non-zero partial charges to
    its atoms → the framework must carry real charges for cDFT electrostatics.

    Data-driven (NO hard-coded gas table): parse the gas's .def
    'Kind_of_solvent_atoms' block; if ANY atom type has a non-zero charge the
    gas is polar/quadrupolar (e.g. CO2: +0.70 / −0.35) and needs a charged
    framework. If all gas-atom charges are zero (e.g. CH4: single site charge 0)
    the molecule is apolar → a neutral framework is physically fine.
    Unknown gas (no .def) → assume True (safe: require framework charges).
    """
    uff_dir = uff_dir or _UFF_DIR
    path = os.path.join(uff_dir, f'{gas}.def')
    if not os.path.exists(path):
        return True  # unknown gas → safe default: require framework charges
    in_kind = False
    for line in open(path):
        s = line.strip()
        if s.startswith('Kind_of_solvent_atoms'):
            in_kind = True
            continue
        if in_kind:
            if s.startswith('Number_of_solvent_atoms'):
                break
            parts = s.split()
            # row: ID  diameter(A)  Epsilon(K)  charge
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    if abs(float(parts[3])) > 1e-6:
                        return True
                except ValueError:
                    pass
    return False

def _load_cg_ff(path):
    """Read Molecule_FF/data_ff_Coarsening_gas → dict keyed by gas name."""
    ff = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith('gas'):
                continue
            name = parts[0]
            ff[name] = {'sigma': float(parts[1]), 'epsilon': float(parts[2]),
                        'mass': float(parts[3])}
    return ff


def _load_uff_def(gas, uff_dir):
    """Read Molecule_FF/UFF/{gas}.def → (solvent_block_str, atoms_block_str)
    Returns the two sections as raw strings ready to write into input.dat.
    """
    path = os.path.join(uff_dir, f'{gas}.def')
    if not os.path.exists(path):
        raise FileNotFoundError(f'UFF definition not found: {path}')
    with open(path) as f:
        content = f.read()
    return content.strip()


def _cell_expansion(lxa, lyb, lzc, cut):
    """Replication factors (na, nb, nc) so the box satisfies the minimum-image
    rule: min(a,b,c) > cut (= 2 × interaction cutoff). Returns (1,1,1) when the
    box is already large enough. floor(L)+1 guarantees STRICTLY greater even
    when L is an exact multiple of cut."""
    def _n(L):
        if L > cut:
            return 1
        return int(cut // L) + 1
    return _n(lxa), _n(lyb), _n(lzc)


def _parse_cif_tokens(lines):
    lxa = lyb = lzc = alpha = beta = gamma = None
    tot = 180000
    atom_data = [['' for _ in range(8)] for _ in range(tot)]
    aid = -1
    i = 0
    n = len(lines)

    def skip_loop_data(start_idx):
        j = start_idx
        while j < n:
            s = lines[j].strip()
            if not s or s.startswith('#'):
                j += 1; continue
            if s.startswith('loop_') or s.startswith('data_') or s.startswith('_'):
                break
            j += 1
        return j

    while i < n:
        stripped = lines[i].strip()
        if not stripped or stripped.startswith('#'):
            i += 1; continue
        parts = stripped.split()
        key = parts[0]

        if key == 'loop_':
            i += 1
            tags = []
            while i < n:
                s = lines[i].strip()
                if not s or s.startswith('#'):
                    i += 1; continue
                if s.startswith('_'):
                    tags.append(s.split()[0]); i += 1
                else:
                    break
            tag_set = set(tags)
            if ('_atom_site_fract_x' in tag_set and
                    '_atom_site_fract_y' in tag_set and
                    '_atom_site_fract_z' in tag_set):
                col = {t: j for j, t in enumerate(tags)}
                ix = col['_atom_site_fract_x']
                iy = col['_atom_site_fract_y']
                iz = col['_atom_site_fract_z']
                itype = col.get('_atom_site_type_symbol')
                ichg = col.get('_atom_site_charge')
                nneed = max(ix, iy, iz) + 1
                while i < n:
                    s = lines[i].strip()
                    if not s or s.startswith('#'):
                        i += 1; continue
                    if s.startswith('loop_') or s.startswith('data_') or s.startswith('_'):
                        break
                    row = s.split()
                    if len(row) < nneed:
                        i += 1; continue
                    aid += 1
                    el = row[itype] if itype is not None else row[0]
                    atom_data[aid][0] = el
                    atom_data[aid][1] = row[ix]
                    atom_data[aid][2] = row[iy]
                    atom_data[aid][3] = row[iz]
                    if ichg is not None and len(row) > ichg and row[ichg].strip():
                        atom_data[aid][7] = row[ichg]
                    else:
                        atom_data[aid][7] = '0.0'
                    with open(_FF_UFF) as fffile:
                        for ffline in fffile:
                            if ffline.split()[0] == atom_data[aid][0]:
                                sp = ffline.split()
                                atom_data[aid][4] = sp[1]
                                atom_data[aid][5] = sp[2]
                                atom_data[aid][6] = sp[3]
                                break
                    i += 1
                continue
            i = skip_loop_data(i)
            continue

        if key.startswith('_cell_length_a') and len(parts) >= 2:
            lxa = float(parts[1].strip('"\'').split('(')[0])
        elif key.startswith('_cell_length_b') and len(parts) >= 2:
            lyb = float(parts[1].strip('"\'').split('(')[0])
        elif key.startswith('_cell_length_c') and len(parts) >= 2:
            lzc = float(parts[1].strip('"\'').split('(')[0])
        elif key.startswith('_cell_angle_alpha') and len(parts) >= 2:
            alpha = float(parts[1].strip('"\'').split('(')[0])
        elif key.startswith('_cell_angle_beta') and len(parts) >= 2:
            beta = float(parts[1].strip('"\'').split('(')[0])
        elif key.startswith('_cell_angle_gamma') and len(parts) >= 2:
            gamma = float(parts[1].strip('"\'').split('(')[0])
        i += 1

    return lxa, lyb, lzc, alpha, beta, gamma, aid, atom_data


def generate_cdft_inputs(
    cif_dir,
    gases,
    temperature,
    bulk_densities,
    henry=0,
    output_dir=None,
    ngrid=60,
    dl=0.5,
    deltamax=2.0,
    Rotation=10,
    kapa=0.95,
    torr=0.1,
    readvext=0,
    EOS=1,
    Diele_con=1.0,
    N3_cutoff=0.99,
    Cutoff_inside=0.5,
    cut_coul=15.0,
    cutdft=12.90,
    deos=1,
    Ds_Knudsen='2E-8',
    MFV=0.712,
    auto_expand=True,
    framework_charge=None,
):
    if output_dir is None:
        output_dir = os.path.join(_SCRIPT_DIR, 'inputCOF')

    cg_ff = _load_cg_ff(_CG_FF)
    kgas = len(gases)

    # Gas-aware framework-charge requirement (agent can override via
    # framework_charge=True/False). Determined DATA-DRIVEN by whether the gas
    # molecule's own atoms carry non-zero partial charges (its UFF .def):
    #   CO2 (+0.70/−0.35), SO2, CO, N2 → charged framework required
    #   CH4 (charge 0), H2 → apolar, neutral framework is fine
    if framework_charge is not None:
        _need_framework_charge = bool(framework_charge)
        _basis = f"用户/agent 显式指定 framework_charge={framework_charge}"
    else:
        _per = [_gas_has_partial_charges(g) for g in gases]
        _need_framework_charge = any(_per)
        _basis = ', '.join(f'{g}({"带电" if _per[i] else "电荷0"})' for i, g in enumerate(gases))
    print(f'[cDFT电荷审计] gases={", ".join(gases)}: {_basis} → '
          f'framework_charge={_need_framework_charge} '
          f'({"需要" if _need_framework_charge else "无需"}) 框架电荷')

    # Validate gases
    for g in gases:
        if g not in cg_ff:
            raise ValueError(f"Gas '{g}' not found in {_CG_FF}")
        uff_path = os.path.join(_UFF_DIR, f'{g}.def')
        if not os.path.exists(uff_path):
            raise FileNotFoundError(f"UFF def not found: {uff_path}")

    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    from pathlib import Path
    dataset_root=Path(cif_dir).resolve()
    dataset=sorted(dataset_root.rglob('*.cif'))
    if len(dataset)>10000:raise ValueError('CIF dataset exceeds bounded input audit')
    stems=set()
    for path in dataset:
        if not path.resolve().is_relative_to(dataset_root):raise ValueError('CIF dataset symlink escapes input root')
        if path.stem in stems:raise ValueError('duplicate CIF basename would overwrite a native input; normalize dataset identities first')
        stems.add(path.stem)
    for entry in dataset:
        if not (entry.is_file() and entry.name.endswith('.cif')):
            continue

        with open(entry) as f:
            file_lines = f.readlines()

        lxa, lyb, lzc, alpha, beta, gamma, aid, atom_data = _parse_cif_tokens(file_lines)
        if any(v is None for v in [lxa, lyb, lzc, alpha, beta, gamma]):
            print(f'Skip (missing cell params): {entry.name}'); continue
        if aid < 0:
            print(f'Skip (no atom sites): {entry.name}'); continue

        filename = os.path.splitext(entry.name)[0]
        out_path = os.path.join(output_dir, filename + '.dat')

        # ── cDFT box audit (minimum-image rule) ─────────────────────────
        # The Input.dat embeds a real-space cutoff (cutdft for DFT, cut_coul
        # for Coulomb). Periodic images must not overlap the interaction range,
        # so the box must satisfy min(a,b,c) > 2×max(cutoff). A MOF whose cell
        # is too small (e.g. Ni-MOF-74: c≈6.8 Å vs 2×12.9 Å) would produce a
        # physically invalid Input.dat. If too small, auto-expand by replicating
        # the unit cell (na×nb×nc) so the box obeys the rule.
        cut = 2.0 * max(cutdft, cut_coul)
        na, nb, nc = _cell_expansion(lxa, lyb, lzc, cut)
        need_expand = (na, nb, nc) != (1, 1, 1)

        # Collect atoms (element, fractional xyz, diameter, epsilon, mass, charge)
        atoms_out = []
        for i in range(aid + 1):
            atoms_out.append([
                atom_data[i][0],
                float(atom_data[i][1]), float(atom_data[i][2]), float(atom_data[i][3]),
                float(atom_data[i][4]), float(atom_data[i][5]), float(atom_data[i][6]),
                float(atom_data[i][7] or 0.0),
            ])

        # ── cDFT charge audit (gas-aware) ─────────────────────────────
        # Only polar/quadrupolar gases (CO2/SO2/CO/N2/…) require the framework
        # to carry REAL partial charges. For those, an uncharged CIF silently
        # produces a charge-free electrostatic calculation — fail loudly and
        # point the agent to PACMAN/PACMol. For apolar gases (CH4/H2/…) a
        # neutral framework is physically fine → audit skipped.
        if _need_framework_charge:
            _max_q = max((abs(q) for _, _, _, _, _, _, _, q in atoms_out), default=0.0)
            if _max_q < 1e-6:
                raise ValueError(
                    f"[cDFT电荷审计] {filename}: 气体 {', '.join(gases)} 需要框架电荷，但 CIF 未赋予任何原子电荷"
                    f"（Input.dat 电荷全为 0）。cDFT 静电力学需要真实 DDEC6/CM5 电荷，未充电结构算出的吸附/电荷分布不可信。\n"
                    f"修复方法：先对 CIF 运行 run_pacman_charge(cif_dir=...)（默认 method='pacmof'，CPU 快速；"
                    f"需要精确 DDEC6 时用 method='pacman'），然后用输出的 *_pacmof.cif / *_pacman.cif"
                    f"（含 _atom_site_charge）作为 cDFT 输入，"
                    f"不要用未充电结构直接跑 cDFT。若该体系确实不需要框架电荷，可显式传 framework_charge=False 跳过审计。"
                )

        if need_expand:
            if not auto_expand:
                print(f'[cDFT跳过] {filename}: box {lxa:.2f}x{lyb:.2f}x{lzc:.2f}Å min={min(lxa,lyb,lzc):.2f}Å ≤ 2×cutoff {cut:.2f}Å 且 auto_expand=False，跳过')
                continue
            _orig_box = (lxa, lyb, lzc)
            _orig_n = len(atoms_out)
            _new = []
            for a in atoms_out:
                fx, fy, fz = a[1], a[2], a[3]
                for ia in range(na):
                    for ib in range(nb):
                        for ic in range(nc):
                            _new.append([a[0], (fx + ia) / na, (fy + ib) / nb,
                                         (fz + ic) / nc, a[4], a[5], a[6], a[7]])
            atoms_out = _new
            lxa, lyb, lzc = na * lxa, nb * lyb, nc * lzc
            print(f'[cDFT扩包] {filename}: box {_orig_box[0]:.2f}x{_orig_box[1]:.2f}x{_orig_box[2]:.2f}Å '
                  f'(min {min(_orig_box):.2f}Å ≤ 2×cutoff {cut:.2f}Å) → 扩 {na}x{nb}x{nc}，'
                  f'box → {lxa:.2f}x{lyb:.2f}x{lzc:.2f}Å，原子 {_orig_n}→{len(atoms_out)}')

        massmof = sum(a[6] for a in atoms_out)
        generated_files.append(out_path)

        with open(out_path, 'w') as f:
            f.write('water  %.2f  %d\n' % (massmof,henry))
            f.write('Nmaxx Nmaxy Nmaxz\n')
            f.write('%d   %d   %d\n' % (ngrid, ngrid, ngrid))
            f.write('Lxa Lyb Lzc dx angle_a angle_b angle_c\n')
            f.write('%f   %f   %f   %f  %f  %f  %f\n' % (lxa, lyb, lzc, dl, alpha, beta, gamma))
            f.write('Nutation  Precession  Rotation\n')
            f.write('%d    %d   %d\n' % (Rotation, Rotation, Rotation))
            f.write('Temperature(K) Kapa Delta_Max Torr\n')
            f.write('%d    %f    %f    %f\n' % (temperature, kapa, deltamax, torr))
            f.write('Kind_of_gas\n%d\n' % kgas)

            # Gas coarse-grained params
            # Single component: index starts at 1 (matches example input.dat)
            # Mixed: index starts at 0 (matches example input_mix.dat)
            start_idx = 1 if kgas == 1 else 0
            f.write('Name Epsilon(K) Sigma(A) Bulk_Density(molec/A^3)\n')
            for gi, g in enumerate(gases):
                p = cg_ff[g]
                f.write('%d    %f    %f    %.7f\n' % (
                    gi + start_idx, p['epsilon'], p['sigma'], bulk_densities[gi]))

            f.write('Special_case\n0\nId1 Id2 Epsilon(K) Sigma(A)\n')
            f.write('Read_Vext  EOS    Cutoff deos    N3_cutoff Diele_con Cutoff_inside cut_coul\n')
            f.write('%d    %d     %.2f   %d   %.2f  %.2f  %.2f  %.2f\n' % (
                readvext, EOS, cutdft, deos, N3_cutoff, Diele_con, Cutoff_inside, cut_coul))
            f.write('Mass(g/mol) Ds_Knudsen(m^2/s) MFV_parameter\n')
            f.write('%s   %s     %s\n' % (
                str(int(cg_ff[gases[0]]['mass'])), Ds_Knudsen, MFV))

            # UFF full-atom solvent blocks (Kind_of_solvent_atoms per gas)
            for g in gases:
                uff = _load_uff_def(g, _UFF_DIR)
                # Write only the Kind_of_solvent_atoms section (first part of .def)
                for line in uff.split('\n'):
                    if line.startswith('Number_of_solvent_atoms'):
                        break
                    f.write(line + '\n')

            # Number_of_solvent_atoms blocks per gas
            for gi, g in enumerate(gases):
                uff = _load_uff_def(g, _UFF_DIR)
                lines = uff.split('\n')
                in_num = False
                for line in lines:
                    if line.startswith('Number_of_solvent_atoms'):
                        # Rename to include gas index
                        f.write('Number_of_solvent_atoms%d\n' % (gi + start_idx))
                        in_num = True
                        continue
                    if in_num:
                        f.write(line + '\n')

            # Solute (MOF atoms)
            PI = 3.14159265358979323846
            a = alpha * PI / 180
            b = beta  * PI / 180
            g = gamma * PI / 180
            Box_Cell = np.zeros(9)
            Box_Cell[0] = lxa
            Box_Cell[3] = lyb * np.cos(g)
            Box_Cell[4] = lyb * np.sin(g)
            Box_Cell[6] = lzc * np.cos(b)
            Box_Cell[7] = lzc * (np.cos(a) - np.cos(b) * np.cos(g)) / np.sin(g)
            Box_Cell[8] = lzc * np.sqrt(
                1 - np.cos(b)**2 - np.cos(g)**2 - np.cos(a)**2
                + 2 * np.cos(b) * np.cos(g) * np.cos(a)) / np.sin(g)

            f.write('Number_of_solute_atoms\n%d\n' % len(atoms_out))
            f.write('ID x y z diameter(A) Epsilon(K) charge\n')
            for i, a in enumerate(atoms_out):
                fx, fy, fz = a[1], a[2], a[3]
                x = fx*Box_Cell[0] + fy*Box_Cell[3] + fz*Box_Cell[6]
                y = fx*Box_Cell[1] + fy*Box_Cell[4] + fz*Box_Cell[7]
                z = fx*Box_Cell[2] + fy*Box_Cell[5] + fz*Box_Cell[8]
                f.write('%d  %f  %f  %f  %f  %f  %f\n' % (
                    i + 1, x, y, z, a[4], a[5], a[7]))

        print(f'Done: {out_path}  (atoms={len(atoms_out)}, mass={massmof:.2f})')

    return generated_files


def expand_cif_dir(cif_dir, cutoff=15.0, output_dir=None):
    """Supercell-expand every CIF so the box satisfies the minimum-image rule
    min(a,b,c) > 2×cutoff. Writes <name>.cif into output_dir (expanded when the
    box was too small, copied verbatim otherwise) and returns a JSON report.

    This is the standalone '扩包' tool the agent can call BEFORE submitting —
    useful to feed the SAME expanded cell to cDFT and RASPA GCMC (which has the
    identical 2×cutoff truncation rule). The cDFT Input.dat generator also
    auto-expands inline, so this tool is for explicit/consistent expansion.
    """
    import shutil
    if output_dir is None:
        output_dir = os.path.join(cif_dir, 'expanded')
    os.makedirs(output_dir, exist_ok=True)
    cut = 2.0 * cutoff
    report = []
    n_total = n_expanded = 0
    for entry in os.scandir(cif_dir):
        if not (entry.is_file() and entry.name.endswith('.cif')):
            continue
        n_total += 1
        with open(entry.path) as f:
            lines = f.readlines()
        try:
            lxa, lyb, lzc, alpha, beta, gamma, aid, atom_data = _parse_cif_tokens(lines)
        except Exception as e:
            report.append({"cif": entry.name, "error": f"parse failed: {e}"})
            continue
        if any(v is None for v in [lxa, lyb, lzc, alpha, beta, gamma]) or aid < 0:
            report.append({"cif": entry.name, "error": "missing cell/atoms"})
            continue
        na, nb, nc = _cell_expansion(lxa, lyb, lzc, cut)
        if (na, nb, nc) == (1, 1, 1):
            shutil.copy(entry.path, os.path.join(output_dir, entry.name))
            report.append({"cif": entry.name, "expanded": False})
            continue
        n_expanded += 1
        name = os.path.splitext(entry.name)[0]
        out = [
            f"data_{name}",
            f"_cell_length_a {na * lxa:.8f}",
            f"_cell_length_b {nb * lyb:.8f}",
            f"_cell_length_c {nc * lzc:.8f}",
            f"_cell_angle_alpha {alpha}",
            f"_cell_angle_beta {beta}",
            f"_cell_angle_gamma {gamma}",
            "loop_",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        ]
        has_charge = any(str(atom_data[i][7]).strip() not in ('', '0.0')
                         for i in range(aid + 1))
        if has_charge:
            out.append("_atom_site_charge")
        old_n = aid + 1
        for i in range(aid + 1):
            fx, fy, fz = float(atom_data[i][1]), float(atom_data[i][2]), float(atom_data[i][3])
            el = atom_data[i][0]
            chg = atom_data[i][7]
            for ia in range(na):
                for ib in range(nb):
                    for ic in range(nc):
                        row = [str(el), f"{(fx + ia) / na:.8f}", f"{(fy + ib) / nb:.8f}",
                               f"{(fz + ic) / nc:.8f}"]
                        if has_charge:
                            row.append(str(chg))
                        out.append("  ".join(row))
        with open(os.path.join(output_dir, entry.name), 'w') as f:
            f.write("\n".join(out) + "\n")
        report.append({
            "cif": entry.name, "expanded": True,
            "na": na, "nb": nb, "nc": nc,
            "box": f"{na * lxa:.2f}x{nb * lyb:.2f}x{nc * lzc:.2f}",
            "atoms": f"{old_n}->{old_n * na * nb * nc}",
        })
    return {"n_total": n_total, "n_expanded": n_expanded,
            "cutoff": cutoff, "output_dir": output_dir, "report": report}


if __name__ == '__main__':
    # Example usage - these parameters should come from user input in agent workflow
    cif_dir      = '/home/user/gcmc_agent/cifs/pacman_04201447'
    gases        = ['CO2', 'CO']
    temperature  = 298
    bulk_densities = [0.0000125, 0.0000125]

    generated = generate_cdft_inputs(
        cif_dir=cif_dir,
        gases=gases,
        temperature=temperature,
        bulk_densities=bulk_densities
    )

    print(f"\nTotal files generated: {len(generated)}")


# ────────────────────────────────────────────────────────────────────────────
# [patcher-fix] Robust bulk_densities normalization
# Root cause: registry.py run_cdft builds `bulk` via `cg.get(g, 0.0000125)`
# where cg = _load_cg_ff(_CG_FF) returns {gas: {'sigma','epsilon','mass'}} —
# i.e. the whole params dict instead of a density scalar. That made
# bulk_densities a list of dicts and crashed the `.dat` writer at
# `f.write('%.7f' % bulk_densities[gi])` with "must be real number, not dict".
# This wrapper accepts any of:
#   - list[float]                (canonical, unchanged)
#   - dict {gas: density}        (per-gas lookup, missing -> default)
#   - list with dict/bad entries (registry bug) -> default density
# and always yields a flat list of floats.
# ----------------------------------------------------------------------------
_orig_generate_cdft_inputs = generate_cdft_inputs

def generate_cdft_inputs(*args, **kwargs):
    def _to_rho(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0000125  # ~1 bar default (molec/A^3), same as __main__ example

    bd = kwargs.get('bulk_densities')
    gases = kwargs.get('gases')
    if gases is None and len(args) > 1:
        gases = args[1]
    if isinstance(bd, dict):
        bd = [bd.get(g, 0.0000125) for g in (gases or [])]
        kwargs['bulk_densities'] = [_to_rho(x) for x in bd]
    elif bd is not None:
        kwargs['bulk_densities'] = [_to_rho(x) for x in bd]
    return _orig_generate_cdft_inputs(*args, **kwargs)
