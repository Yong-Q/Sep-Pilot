"""Slurm reads directives only before the first executable script line."""
from agents.slurm import render_sbatch_script


def test_gpu_exclusion_is_in_the_scheduler_header_before_commands():
    script = render_sbatch_script('string-smoke', 'echo complete', '/path/to/work',
                                  partition='gpu', gres='gpu:1',
                                  extra_directives=['--exclude=gpu3'])
    scheduler_header = []
    for line in script.splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            break
        scheduler_header.append(line)
    assert '#SBATCH --exclude=gpu3' in scheduler_header
