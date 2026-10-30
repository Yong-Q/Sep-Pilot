"""Independent review regressions. No scheduler/model/scientific execution."""
import pytest
from agents.workload_budget import collect_workload_facts, size_workload_budget
from tests.test_workload_budget import snapshot, estimate


def test_missing_fingerprint_cannot_bind_operator_memory_evidence():
    facts={'tool':'run_cdft','status':'ready','execution_model':'independent_processes',
           'work_items':10,'threads_per_worker':1}
    result=size_workload_budget(facts,snapshot(),memory_evidence={
        'source':'operator','basis':'caller text','per_worker_mb':1000,'workload_fingerprint':None})
    assert result['status']!='ready' and not result['candidates']


@pytest.mark.parametrize('tool', ['run_cdft','run_md_optimize'])
def test_exact_cpu_allocation_cannot_outgrow_memory_sized_parallelism(tmp_path,tool):
    if tool=='run_cdft':
        directory=tmp_path/'inputs';directory.mkdir()
        for n in range(10):(directory/f'{n}.dat').write_text('fixture')
        args={'action':'submit','input_dir':str(directory)}
    else:
        p=tmp_path/'in.cif';p.write_text('fixture');args={'cif_path':str(p)}
    facts=collect_workload_facts(tool,args,owned_root=tmp_path)
    result=size_workload_budget(facts,snapshot(cpu=8,ram=8000),memory_evidence=estimate(facts,mb=4000),
        constraints=[{'source':'user','field':'cpus','kind':'exact','value':4,'basis':'approved four processes/ranks'}])
    # The adapters launch CPU-count processes/ranks. Two units of RAM cannot
    # justify reserving four CPUs that will launch four processes/ranks.
    assert result['status']!='ready' and not result['candidates']
