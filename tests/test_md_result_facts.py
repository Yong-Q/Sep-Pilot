"""Synthetic native log fragments; deterministic facts, not MD physics."""
import pytest


def stage(stop='linesearch alpha is zero', force='14.3778', pressure='-19.611668'):
    return f'''Setting up cg style minimization ...
  Unit style    : real
Step Temp E_pair E_mol TotEng Press Volume
58 0 253.12874 424.22886 677.3576 {pressure} 16755.846
59 0 253.12874 424.22886 677.3576 {pressure} 16755.846
Loop time of 0.009 on 1 procs for 1 steps with 424 atoms
99.6% CPU use with 1 MPI tasks x no OpenMP threads
Minimization stats:
  Stopping criterion = {stop}
  Energy initial, next-to-last, final =
      677.357595995 677.357595995 677.357595995
  Force two-norm initial, final = {force} {force}
  Force max component initial, final = 14.3773 14.3773
  Final line search alpha, max atom move = 3.31659e-11 4.76837e-10
  Iterations, force evaluations = 1 43
MPI task timing breakdown:
'''


def test_final_minimization_and_its_thermo_override_earlier_atomic_and_later_run_zero():
    from agents.md_result_facts import parse_md_result_facts
    earlier = stage('energy tolerance', '0.282444', '-22.888999')
    run_zero = '''Setting up Verlet run ...
Step Temp Press PotEng
59 0 -999 1234
Loop time of 0.0 on 1 procs for 0 steps
'''
    facts = parse_md_result_facts(earlier + stage() + run_zero)
    assert facts['ok'] and facts['minimization_stage_count'] == 2
    final = facts['final_minimization']
    assert final['stopping_criterion'] == 'linesearch alpha is zero'
    assert final['force_two_norm_final'] == 14.3778
    assert final['energy_final'] == 677.357595995
    assert final['native_tolerance_stop'] is False
    assert final['iterations'] == 1 and final['force_evaluations'] == 43
    thermo = facts['final_minimization_thermo']
    assert thermo['values']['Press'] == -19.611668
    assert thermo['values']['Step'] == 59
    assert thermo['unit_style'] == 'real'
    assert thermo['mpi_tasks'] == 1
    assert thermo['last_row_line'] < final['line_start']
    assert len(facts['source_sha256']) == 64
    assert facts['subsequent_nonzero_run'] is False


def test_later_dynamics_is_explicitly_distinguished_from_minimization_thermo():
    from agents.md_result_facts import parse_md_result_facts
    facts = parse_md_result_facts(stage() + 'Loop time of 1 on 1 procs for 100 steps\n')
    assert facts['subsequent_nonzero_run'] is True
    assert facts['final_minimization_thermo']['values']['Press'] == -19.611668
    assert any('dynamics' in warning for warning in facts['warnings'])


def test_final_nonfinite_thermo_row_does_not_fall_back_to_finite_previous_row():
    from agents.md_result_facts import parse_md_result_facts
    text = stage().replace('59 0 253.12874 424.22886 677.3576 -19.611668',
                           '59 0 253.12874 424.22886 677.3576 nan')
    facts = parse_md_result_facts(text)
    assert not facts['ok'] and facts['final_minimization_thermo']['values']['Press'] is None


def test_truncated_final_block_never_falls_back_to_earlier_success():
    from agents.md_result_facts import parse_md_result_facts
    text = stage('energy tolerance') + 'Setting up cg style minimization ...\nMinimization stats:\n'
    facts = parse_md_result_facts(text)
    assert not facts['ok']
    assert facts['final_minimization']['stopping_criterion'] is None
    assert facts['final_minimization']['native_tolerance_stop'] is None
    assert facts['warnings']


@pytest.mark.parametrize('mutation', ['force_nan', 'pressure_nan', 'no_minimization'])
def test_missing_or_nonfinite_final_facts_are_not_accepted(mutation):
    from agents.md_result_facts import parse_md_result_facts
    text = stage(force='nan') if mutation == 'force_nan' else stage(pressure='nan') if mutation == 'pressure_nan' else 'Loop time complete\n'
    assert not parse_md_result_facts(text)['ok']


def test_read_wrapper_is_bounded_and_hashes_original_log(tmp_path):
    from agents.md_result_facts import read_md_result_facts
    log = tmp_path / 'solver.log'; log.write_text(stage())
    result = read_md_result_facts(log)
    assert result['ok'] and result['source_path'] == str(log)
    assert not read_md_result_facts(log, max_bytes=10)['ok']
    assert log.read_text() == stage()
