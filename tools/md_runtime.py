"""Framework MD adapter for user-installed LAMMPS/MPI and lammps-interface.

No solver binaries, private installation paths, modules or environment activation.
Minimization and NPT templates retain the existing application physics settings.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess


def _executable(value):
    resolved = shutil.which(str(value))
    if not resolved:
        raise ValueError(f'MD executable is unavailable: {value}')
    return resolved


def _quoted(path):
    value = str(path)
    if any(char in value for char in ('"', '\n', '\r')):
        raise ValueError('MD paths cannot contain double quotes or newlines')
    return '"' + value + '"'


def run(cif_path, output_dir, *, executable, mpi_executable='mpirun',
        interface_executable='lammps-interface', library_path='', mode='single',
        cutoff=6., force_field='UFF', temperature=None, pressure=1., npt_ps=700.,
        timestep=1., n_procs=4, skip_convert=False):
    source, work = Path(cif_path).resolve(), Path(output_dir).resolve()
    result = dict(cif_name=source.stem, success=False, energy=None, output_cif='',
                  work_dir=str(work), error=None)
    try:
        if not source.is_file():
            raise ValueError('MD CIF input is missing')
        if mode not in {'single', 'md'} or not isinstance(n_procs, int) or n_procs < 1:
            raise ValueError('MD mode/process count is invalid')
        if mode == 'md' and (temperature is None or not all(math.isfinite(float(v))
                for v in (temperature, pressure, npt_ps, timestep))
                or temperature <= 0 or pressure < 0 or npt_ps <= 0 or timestep <= 0):
            raise ValueError('MD requires finite temperature, pressure, duration and timestep')
        solver = _executable(executable)
        interface = _executable(interface_executable)
        mpi = _executable(mpi_executable) if n_procs > 1 else None
        environment = os.environ.copy()
        if library_path:
            inherited = environment.get('LD_LIBRARY_PATH', '')
            environment['LD_LIBRARY_PATH'] = inherited + (':' if inherited else '') + library_path
        work.mkdir(parents=True, exist_ok=True)
        stage = work / 'interface'; stage.mkdir(exist_ok=True)
        native_dir = work / ('md_data' if mode == 'md' else 'opt_data'); native_dir.mkdir(exist_ok=True)
        log_dir = work / 'log'; log_dir.mkdir(exist_ok=True)
        def execute(argv, cwd, log):
            with log.open('x') as stream:
                subprocess.run(argv, cwd=cwd, env=environment, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=7200, check=True)
        execute([interface, f'--cutoff={cutoff}', '-ff', force_field, str(source)],
                stage, work / 'interface.log')
        original = stage / ('in.' + source.stem)
        data = stage / ('data.' + source.stem)
        if not original.is_file() or not data.is_file():
            raise ValueError('lammps-interface did not produce native input and data')
        native = native_dir / (source.stem + '.lammps-data')
        if native.exists():
            raise FileExistsError('MD output already exists; use a new workspace')
        lines = original.read_text().splitlines()
        for i, line in enumerate(lines):
            command = line.split('#', 1)[0].strip().split()
            if command and command[0] == 'read_data':
                lines[i] = 'read_data ' + _quoted(data)
            elif command and command[0] == 'log':
                lines[i] = 'log ' + _quoted(log_dir / ('log.' + source.stem))
        if mode == 'md':
            steps = int(npt_ps * 1000 / timestep)
            if steps < 1:
                raise ValueError('MD duration must include at least one timestep')
            block = _MD_TEMPLATE.format(temperature=temperature, pressure=pressure,
                npt_ps=npt_ps, timestep=timestep, npt_steps=steps, output_path=_quoted(native))
        else:
            block = _MINIMIZE_TEMPLATE.format(n_iter=10, etol=1e-6, ftol=1e-8,
                maxiter=1000, maxeval=10000, cif_id=source.stem, output_path=_quoted(native))
        prepared = stage / (mode + '_in.' + source.stem)
        prepared.write_text('\n'.join(lines) + '\n' + block)
        argv = ([mpi, '-np', str(n_procs)] if mpi else []) + [solver, '-in', str(prepared)]
        execute(argv, work, work / 'solver.log')
        text = (work / 'solver.log').read_text(errors='replace')
        if re.search(r'^\s*ERROR(?:\s|:)', text, re.M) or not native.is_file() or not native.stat().st_size:
            raise ValueError('LAMMPS failed or did not produce a final structure; inspect solver.log')
        thermo = re.findall(r'^Temp Press Volume PotEng KinEng\s*\n([^\n]+)', text, re.M)
        if thermo:
            values = [float(x) for x in thermo[-1].split()]
            if len(values) != 5 or not all(math.isfinite(x) for x in values):
                raise ValueError('LAMMPS final thermo contains nonfinite data')
            result['energy'] = values[3]
        else:
            energy = re.search(re.escape(source.stem) + r':\s*([-\d.eE+]+)', text)
            if energy:
                result['energy'] = float(energy[1])
        output = native
        if not skip_convert:
            from ase.io import read, write
            from ase.data import atomic_masses, chemical_symbols
            atoms = read(native, format='lammps-data', atom_style='full', units='real')
            if not len(atoms) or not all(math.isfinite(float(x)) for row in atoms.positions for x in row):
                raise ValueError('LAMMPS final structure has missing or nonfinite coordinates')
            for atom in atoms:
                index = min(range(1, len(atomic_masses)), key=lambda i: abs(atomic_masses[i] - atom.mass))
                atom.symbol = chemical_symbols[index]
            output = work / (source.stem + ('_md.cif' if mode == 'md' else '_opt.cif'))
            if output.exists():
                raise FileExistsError('MD CIF output already exists')
            write(output, atoms)
        result.update(success=True, output_cif=str(output))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', required=True, help='JSON object written by the registered MD tool')
    args = parser.parse_args()
    result = run(**json.loads(args.parameters))
    print(json.dumps(result, indent=2))
    return 0 if result['success'] else 1

_MINIMIZE_TEMPLATE = """
variable a loop {n_iter}
label loop

minimize {etol} {ftol} {maxiter} {maxeval}
fix 1 all box/relax iso 0.0 vmax 0.001
minimize {etol} {ftol} {maxiter} {maxeval}
unfix 1
thermo 1000

next a
jump SELF loop
run 0

compute 1 all pe
variable A equal c_1

thermo_style custom temp c_1
thermo 1
run 0

print "{cif_id}:$A" append energy.txt

write_data {output_path}
"""

_MD_TEMPLATE = """
# ==== Neighbor list (prevent atom loss) ====
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# ==== Step 1: Thorough minimization ====
min_style cg
minimize 1.0e-6 1.0e-8 5000 50000
fix boxrelax all box/relax iso 0.0 vmax 0.001
minimize 1.0e-6 1.0e-8 5000 50000
unfix boxrelax
minimize 1.0e-6 1.0e-8 5000 50000

# ==== Step 2: NVE + Langevin gentle heating (0.5 fs, 20ps) ====
# Use NVE + Langevin to prevent atom loss and bond breaking
timestep 0.5
velocity all create 50.0 12345 dist gaussian
fix lang all langevin 50.0 {temperature} 100.0 12345
fix nve all nve
thermo 2000
run 40000
unfix lang
unfix nve

# ==== Step 3: NVT equilibration at target T (0.5 fs, 10ps) ====
fix nvt_eq all nvt temp {temperature} {temperature} 100.0
thermo 2000
run 20000
unfix nvt_eq

# ==== Step 4: NPT production (1.0 fs, {npt_ps}ps) ====
timestep {timestep}
fix npt all npt temp {temperature} {temperature} 100.0 iso {pressure} {pressure} 1000.0
thermo 1000
run {npt_steps}
unfix npt

# ==== Write final structure ====
thermo_style custom temp press vol pe ke
thermo 1
run 0
write_data {output_path}
"""

if __name__ == '__main__':
    raise SystemExit(main())
