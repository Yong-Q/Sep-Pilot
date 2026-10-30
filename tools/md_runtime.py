"""Framework MD adapter for user-installed LAMMPS/MPI and lammps-interface.

No solver binaries, private installation paths, modules or environment activation.
Minimization and NPT templates retain the existing application physics settings.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
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


def _execution_input_fact(path):
    """Hash one bounded no-symlink script tied to the linked file identity."""
    path=Path(path)
    fd=os.open(path,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024*1024:
            raise ValueError('bounded regular solver input required')
        digest=hashlib.sha256();size=0
        while True:
            block=os.read(fd,65536)
            if not block:break
            size+=len(block)
            if size>1024*1024:raise ValueError('solver input grew beyond bound')
            digest.update(block)
        after=os.fstat(fd);linked=os.stat(path,follow_symlinks=False)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
        if identity(before)!=identity(after) or identity(after)!=identity(linked) or size!=before.st_size:
            raise ValueError('solver input changed during hashing')
        return {'path':str(path),'sha256':digest.hexdigest(),'bytes':size}
    finally:os.close(fd)


def _actual_stage_steps(text):
    """Count native run summaries inside exact adapter stage markers only."""
    completed, active, observed = {}, None, None
    for line in text.splitlines():
        begin = re.fullmatch(r'SEPPILOT_STAGE (warmup|equilibration|production) begin', line.strip())
        end = re.fullmatch(r'SEPPILOT_STAGE (warmup|equilibration|production) end', line.strip())
        if begin:
            active, observed = begin[1], None
        elif end:
            if active == end[1] and observed is not None:
                completed[active] = observed
            active, observed = None, None
        elif active:
            match = re.match(r'^Loop time of .* for (\d+) steps(?: with|$)', line.strip())
            if match:
                observed = int(match[1]) if observed is None else observed + int(match[1])
    if set(completed) != {'warmup', 'equilibration', 'production'}:
        return None
    return {**completed, 'total': sum(completed.values())}



def _native_thermo_history(raw, work):
    """Load the stdlib native reader without importing the full agent backend."""
    import hashlib
    import importlib.util
    helper = Path(__file__).resolve().parents[1] / 'agents' / 'md_result_facts.py'
    spec = importlib.util.spec_from_file_location('_native_md_result_facts', helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    history = module.parse_md_thermo(raw)
    module.verify_md_thermo(raw, history)
    encoded = json.dumps(history, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    path = work / 'thermo_history.json'
    with path.open('xb') as stream: stream.write(encoded)
    return {**module.md_thermo_summary(history),
        'artifact': {'path': str(path), 'sha256': hashlib.sha256(encoded).hexdigest(), 'bytes': len(encoded)},
        'source_log': str(work / 'solver.log'), 'read_only_source': True}


def run(cif_path, output_dir, *, executable, mpi_executable='mpirun',
        interface_executable='lammps-interface', library_path='', mode='single',
        cutoff=6., force_field='UFF', temperature=None, pressure=1., npt_ps=700.,
        timestep=1., n_procs=4, skip_convert=False, warmup_steps=None,
        equilibration_steps=None, production_steps=None):
    source, work = Path(cif_path).resolve(), Path(output_dir).resolve()
    result = dict(cif_name=source.stem, success=False, energy=None, output_cif='',
                  work_dir=str(work), error=None)
    try:
        if not source.is_file():
            raise ValueError('MD CIF input is missing')
        if mode not in {'single', 'md'} or isinstance(n_procs, bool) or not isinstance(n_procs, int) or n_procs < 1:
            raise ValueError('MD mode/process count is invalid')
        allocated = os.environ.get('SLURM_CPUS_PER_TASK')
        if allocated is not None:
            available = int(allocated)
            if available < 1:
                raise ValueError('SLURM_CPUS_PER_TASK must be a positive integer')
            n_procs = min(n_procs, available)
        result['n_procs'] = n_procs
        if mode == 'md' and (temperature is None or not all(math.isfinite(float(v))
                for v in (temperature, pressure, npt_ps, timestep))
                or temperature <= 0 or pressure < 0 or npt_ps <= 0 or timestep <= 0):
            raise ValueError('MD requires finite temperature, pressure, duration and timestep')
        if mode != 'md' and any(value is not None for value in (warmup_steps, equilibration_steps, production_steps)):
            raise ValueError('MD stage budgets require mode=md; geometry minimization has no MD integration steps')
        if mode == 'md':
            for name, value, minimum in (('warmup_steps', warmup_steps, 0),
                    ('equilibration_steps', equilibration_steps, 0), ('production_steps', production_steps, 1)):
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < minimum):
                    raise ValueError(name + ' must be an integer >= ' + str(minimum))
            budgets = {'warmup_steps': 40000 if warmup_steps is None else warmup_steps,
                       'equilibration_steps': 20000 if equilibration_steps is None else equilibration_steps,
                       'production_steps': int(npt_ps * 1000 / timestep) if production_steps is None else production_steps}
            if budgets['production_steps'] < 1:
                raise ValueError('MD duration must include at least one timestep')
            result.update(stage_budgets=budgets, actual_steps=None, stage_budget_completed=None,
                stage_steps_source='unverified until native solver summaries are collected',
                preliminary_minimization={'stages':3, 'maximum_iterations_each':5000, 'maximum_force_evaluations_each':50000},
                equilibration_verified=False)
        solver = _executable(executable)
        interface = _executable(interface_executable)
        mpi = _executable(mpi_executable)
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
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
            block = _MD_TEMPLATE.format(temperature=temperature, pressure=pressure,
                npt_ps=budgets['production_steps'] * timestep / 1000, timestep=timestep,
                npt_steps=budgets['production_steps'], warmup_steps=budgets['warmup_steps'],
                equilibration_steps=budgets['equilibration_steps'],
                initial_temperature=50.0 if budgets['warmup_steps'] else temperature,
                output_path=_quoted(native))
        else:
            block = _MINIMIZE_TEMPLATE.format(n_iter=10, etol=1e-6, ftol=1e-8,
                maxiter=1000, maxeval=10000, cif_id=source.stem, output_path=_quoted(native))
        prepared = stage / (mode + '_in.' + source.stem)
        with prepared.open('x', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n' + block)
        executed_input = _execution_input_fact(prepared)
        argv = [mpi, '-np', str(n_procs), solver, '-in', str(prepared)]
        execute(argv, work, work / 'solver.log')
        if _execution_input_fact(prepared) != executed_input:
            raise ValueError('solver input changed during native execution')
        result['execution_input'] = {**executed_input,
            'verification':'sha256_before_and_after_solver', 'scientific_acceptance':False}
        raw_log = (work / 'solver.log').read_bytes()
        text = raw_log.decode('utf-8', errors='replace')
        try:
            result['thermo_history'] = _native_thermo_history(raw_log, work)
        except (OSError, ValueError, UnicodeError) as parse_error:
            import hashlib
            result['thermo_history'] = {'parse_status': 'unavailable',
                'error_kind': 'md_thermo_parse_gap', 'error': str(parse_error),
                'source_sha256': hashlib.sha256(raw_log).hexdigest(),
                'scientific_acceptance': False, 'convergence': 'unknown'}
        if re.search(r'^\s*ERROR(?:\s|:)', text, re.M) or not native.is_file() or not native.stat().st_size:
            raise ValueError('LAMMPS failed or did not produce a final structure; inspect solver.log')
        if mode == 'md':
            actual = _actual_stage_steps(text)
            result['actual_steps'] = actual
            if actual is not None:
                result['stage_steps_source'] = 'solver.log stage-delimited native Loop time summaries'
                result['stage_budget_completed'] = all(actual[key] == budgets[key + '_steps']
                    for key in ('warmup', 'equilibration', 'production'))
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

# ==== Step 2: NVE + Langevin heating ({warmup_steps} steps, 0.5 fs) ====
# Use NVE + Langevin to prevent atom loss and bond breaking
timestep 0.5
velocity all create {initial_temperature} 12345 dist gaussian
fix lang all langevin 50.0 {temperature} 100.0 12345
fix nve all nve
thermo 2000
print "SEPPILOT_STAGE warmup begin"
run {warmup_steps}
print "SEPPILOT_STAGE warmup end"
unfix lang
unfix nve

# ==== Step 3: NVT equilibration ({equilibration_steps} steps, 0.5 fs) ====
fix nvt_eq all nvt temp {temperature} {temperature} 100.0
thermo 2000
print "SEPPILOT_STAGE equilibration begin"
run {equilibration_steps}
print "SEPPILOT_STAGE equilibration end"
unfix nvt_eq

# ==== Step 4: NPT production (1.0 fs, {npt_ps}ps) ====
timestep {timestep}
fix npt all npt temp {temperature} {temperature} 100.0 iso {pressure} {pressure} 1000.0
thermo 1000
print "SEPPILOT_STAGE production begin"
run {npt_steps}
print "SEPPILOT_STAGE production end"
unfix npt

# ==== Write final structure ====
thermo_style custom temp press vol pe ke
thermo 1
run 0
write_data {output_path}
"""

if __name__ == '__main__':
    raise SystemExit(main())
