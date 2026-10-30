"""Provider token telemetry never serializes response text or credentials."""
from types import SimpleNamespace

from agents import provider_metrics


def test_anthropic_usage_keeps_reported_counts_and_normalizes_cache_aliases():
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=320, output_tokens=16,
        cache_read_input_tokens=2048, cache_creation_input_tokens=0))
    assert provider_metrics.provider_usage_metrics(response) == {
        'input_tokens': 320, 'output_tokens': 16, 'cache_read_input_tokens': 2048,
        'cache_creation_input_tokens': 0, 'cache_hit_tokens': 2048, 'cache_write_tokens': 0}


def test_deepseek_extra_usage_is_available_without_serializing_the_response():
    class Response:
        usage = SimpleNamespace(input_tokens=64, output_tokens=8,
            model_extra={'prompt_cache_hit_tokens': 4096, 'prompt_cache_miss_tokens': 64,
                         'api_key': 'private-fixture-value', 'content': 'private-response'})
        @property
        def content(self):
            raise AssertionError('Do not read response content')
        def model_dump(self, **kwargs):
            raise AssertionError('Do not serialize response')
    assert provider_metrics.provider_usage_metrics(Response()) == {
        'input_tokens': 64, 'output_tokens': 8, 'prompt_cache_hit_tokens': 4096,
        'prompt_cache_miss_tokens': 64, 'cache_hit_tokens': 4096, 'cache_miss_tokens': 64}


def test_nested_cache_fields_are_retained_and_agreeing_sources_share_an_alias():
    usage = {'prompt_tokens': 5000, 'completion_tokens': 9, 'total_tokens': 5009,
             'prompt_cache_hit_tokens': 4000, 'prompt_cache_miss_tokens': 1000,
             'prompt_tokens_details': {'cached_tokens': 4000, 'private': 'ignore'},
             'input_tokens_details': {'cached_tokens': 4000},
             'output_tokens_details': {'reasoning_tokens': 3}}
    result = provider_metrics.provider_usage_metrics({'usage': usage})
    assert result['cache_hit_tokens'] == 4000
    assert result['cache_miss_tokens'] == 1000
    assert result['prompt_tokens_details.cached_tokens'] == 4000
    assert result['input_tokens_details.cached_tokens'] == 4000
    assert result['output_tokens_details.reasoning_tokens'] == 3
    assert result['total_tokens'] == 5009
    assert all(isinstance(value, int) and not isinstance(value, bool) for value in result.values())


def test_invalid_counters_and_unrelated_payloads_are_not_logged():
    for invalid in [-1, 1.5, '16', True, False, None, {'token': 3}]:
        assert provider_metrics.provider_usage_metrics({'usage': {
            'input_tokens': invalid, 'cache_read_input_tokens': invalid,
            'authorization': 'private-fixture-value', 'raw_prompt': 'private-request'}}) == {}
    assert provider_metrics.provider_usage_metrics(None) == {}
    assert provider_metrics.provider_usage_metrics(SimpleNamespace(usage=None)) == {}


def test_absent_cache_counter_is_distinct_from_reported_zero():
    absent = provider_metrics.provider_usage_metrics({'input_tokens': 17})
    zero = provider_metrics.provider_usage_metrics({'input_tokens': 17, 'cache_read_input_tokens': 0})
    assert absent == {'input_tokens': 17}
    assert zero == {'input_tokens': 17, 'cache_read_input_tokens': 0, 'cache_hit_tokens': 0}
    assert 'cache_miss_tokens' not in zero


def test_inconsistent_cache_sources_are_not_silently_reconciled():
    result = provider_metrics.provider_usage_metrics({'usage': {
        'prompt_cache_hit_tokens': 1000, 'cache_read_input_tokens': 2000}})
    assert result == {'prompt_cache_hit_tokens': 1000, 'cache_read_input_tokens': 2000}
