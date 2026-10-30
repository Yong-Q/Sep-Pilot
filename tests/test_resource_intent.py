from types import SimpleNamespace

from agents.resource_intent import resource_intent


def intent(*messages):
    return resource_intent(SimpleNamespace(original_goal=messages[0], directives=list(messages)), [
        {'step_id':'cdft','tool':'run_cdft'}, {'step_id':'md','tool':'run_md_optimize'}])


def test_model_resource_arguments_never_establish_user_constraints():
    assert intent('计算CO2吸附，自动选择资源') == {'resource_constraints': [], 'resource_pins': {}}
    assert not intent('1核必须改，智能化调度')['resource_constraints']


def test_explicit_cpu_limit_is_scoped_to_named_tool_and_has_native_provenance():
    out = intent('MD只用4核，cDFT最多使用16核')
    assert {(c['step_id'],c['field'],c['kind'],c['value']) for c in out['resource_constraints']} == {
        ('md','cpus','exact',4),('cdft','cpus','max',16)}
    assert all(c['source']=='user' and c['basis'].startswith('authenticated_goal:') for c in out['resource_constraints'])


def test_memory_limit_and_node_selection_require_explicit_user_language():
    out = intent('cDFT最多使用32GB内存，限定节点node25或node26')
    assert out['resource_constraints'][0]['value']==32768
    assert out['resource_pins']['cdft']['nodelist']==['node25','node26']
    assert 'md' not in out['resource_pins']
    assert not intent('node25好像有空闲，为什么1核？')['resource_pins']


def test_user_can_release_old_exact_cpu_constraint_for_automatic_sizing():
    assert not intent('cDFT只用1核', 'cDFT核数改为自动调度')['resource_constraints']


def test_latest_explicit_cpu_change_overrides_previous_value_but_not_other_limits():
    out = intent('固定使用4核，内存最多8GiB', '核数改为8核')
    for step in ('cdft','md'):
        constraints=[c for c in out['resource_constraints'] if c['step_id']==step]
        assert {(c['field'],c['kind'],c['value']) for c in constraints} == {
            ('cpus','exact',8),('memory_mb','max',8192)}


def test_automated_messages_and_negated_requests_cannot_pin_allocation():
    out=intent('执行计算', '[系统]固定使用1核', '不要使用1核')
    assert not out['resource_constraints']


def test_negative_and_question_mentions_are_not_cpu_one_instructions():
    for message in ('不是让你用1核，应该自动调度','不需要固定用1核','为什么只用1核'):
        assert not intent(message)['resource_constraints']


def test_maximum_and_explicit_revision_are_honored():
    controls=intent('cDFT不超过4核')['resource_constraints']
    assert [(c['kind'],c['value']) for c in controls]==[('max',4)]
    controls=intent('cDFT最多使用4核','cDFT改为8核')['resource_constraints']
    assert [(c['kind'],c['value']) for c in controls]==[('exact',8)]


def test_single_tool_memory_clause_inherits_that_tool_scope():
    controls=intent('MD使用4核，内存16GB')['resource_constraints']
    assert {c['step_id'] for c in controls}=={'md'}


def test_explicit_mpi_processes_are_native_cpu_controls_for_md():
    controls=intent('做分子动力学，使用4个MPI进程')['resource_constraints']
    assert [(c['step_id'],c['field'],c['value']) for c in controls]==[('md','cpus',4)]
    controls=intent('cDFT使用8个CPU')['resource_constraints']
    assert [(c['step_id'],c['value']) for c in controls]==[('cdft',8)]
