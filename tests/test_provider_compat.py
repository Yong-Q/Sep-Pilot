"""DeepSeek forced-tool compatibility never changes tool authority or defaults."""
import copy
from types import SimpleNamespace

import pytest

from agents import provider_compat


@pytest.mark.parametrize('choice', [{'type': 'any'}, {'type': 'tool', 'name': 'resource_review_decision'}])
@pytest.mark.parametrize('thinking', [None, {'type': 'enabled', 'budget_tokens': 1024}, {'type': 'disabled'}])
def test_deepseek_forced_tool_disables_thinking_without_mutating_request(choice, thinking):
    kwargs = {'model': 'deepseek-v4-flash', 'max_tokens': 64,
              'system': 'fixed instructions', 'messages': [{'role': 'user', 'content': 'choose'}],
              'tool_choice': choice, 'tools': [{'name': 'resource_review_decision', 'input_schema': {'type': 'object'}}]}
    if thinking is not None: kwargs['thinking'] = thinking
    original = copy.deepcopy(kwargs)
    config = SimpleNamespace(model_provider='anthropic', base_url='https://api.deepseek.com/anthropic')
    result = provider_compat.apply_provider_compat(kwargs, config)
    assert kwargs == original
    assert result['thinking'] == {'type': 'disabled'}
    assert {k: v for k, v in result.items() if k != 'thinking'} == {k: v for k, v in original.items() if k != 'thinking'}


@pytest.mark.parametrize('choice', [None, {'type': 'auto'}, {'type': 'none'}])
@pytest.mark.parametrize('thinking', [None, {'type': 'enabled', 'budget_tokens': 2048}])
def test_deepseek_auto_retains_the_original_thinking_default(choice, thinking):
    kwargs = {'model': 'deepseek-v4-flash'}
    if choice is not None:kwargs['tool_choice'] = choice
    if thinking is not None:kwargs['thinking'] = thinking
    original = copy.deepcopy(kwargs)
    result = provider_compat.apply_provider_compat(kwargs,
        SimpleNamespace(model_provider='anthropic', base_url='https://api.deepseek.com/anthropic'))
    assert result == original and kwargs == original
    assert ('thinking' in result) == ('thinking' in original)


@pytest.mark.parametrize('provider,url', [
    ('codex_cli', 'https://api.deepseek.com/anthropic'),
    ('openai', 'https://api.deepseek.com'),
    ('anthropic', 'https://api.anthropic.com'),
    ('anthropic', 'https://api.deepseek.com.attacker.invalid/anthropic'),
    ('anthropic', 'https://proxy.invalid/api.deepseek.com/anthropic'),
    ('anthropic', 'http://api.deepseek.com/anthropic'),
    ('anthropic', 'https://api.deepseek.com:8443/anthropic'),
    ('anthropic', 'https://demo-user:demo-password@api.deepseek.com/anthropic'),
    ('anthropic', 'not-a-url'),
])
def test_other_providers_and_unverified_endpoints_are_unchanged(provider, url):
    kwargs = {'tool_choice': {'type': 'tool', 'name': 'resource_review_decision'},
              'thinking': {'type': 'enabled'}, 'tools': [{'name': 'resource_review_decision'}]}
    original = copy.deepcopy(kwargs)
    result = provider_compat.apply_provider_compat(kwargs,
        SimpleNamespace(model_provider=provider, base_url=url))
    assert kwargs == original and result == original


def test_compatibility_does_not_read_credentials():
    class Config:
        model_provider = 'anthropic'
        base_url = 'https://API.DEEPSEEK.COM:443/anthropic'
        @property
        def api_key(self):
            raise AssertionError('Compatibility must not inspect credentials')
    result = provider_compat.apply_provider_compat({'tool_choice': {'type': 'any'}}, Config())
    assert result['thinking'] == {'type': 'disabled'}
