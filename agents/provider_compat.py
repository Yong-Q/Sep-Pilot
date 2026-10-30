"""Small provider request adaptations that preserve host tool authorization."""
from urllib.parse import urlsplit


def apply_provider_compat(kwargs, config, *, coordination_effort=None):
    """Adapt forced data returns and native coordinator effort on DeepSeek.

    DeepSeek's Anthropic-compatible API rejects forced tool choices in thinking
    mode. This changes that transport setting, never the selected tool, tool
    definitions, model or permissions. Native routing effort is separate from
    specialist model settings and never disables the coordinator's reasoning.
    Credentials are never inspected. Client tool history is repaired only on
    an independent transport copy, after all caller compaction.
    """
    result = dict(kwargs)
    if getattr(config, 'model_provider', None) == 'anthropic' and 'messages' in kwargs:
        from .provider_transcript import normalize_anthropic_messages
        result['messages'] = normalize_anthropic_messages(kwargs['messages'])
    choice = kwargs.get('tool_choice')
    if getattr(config, 'model_provider', None) != 'anthropic':
        return result
    base_url = getattr(config, 'base_url', None)
    if not isinstance(base_url, str):
        return result
    try:
        endpoint = urlsplit(base_url)
        official = (endpoint.scheme == 'https' and endpoint.hostname == 'api.deepseek.com'
                    and endpoint.port in {None, 443}
                    and endpoint.username is None and endpoint.password is None)
    except ValueError:
        return result
    if official and isinstance(choice, dict) and choice.get('type') in {'any', 'tool'}:
        result['thinking'] = {'type': 'disabled'}
    elif official and coordination_effort in {'low', 'high', 'max'}:
        # Native role policy selects this routing budget. Explicit request
        # settings win, and tool definitions/permissions/model stay unchanged.
        options = dict(result.get('output_config') or {})
        options.setdefault('effort', coordination_effort)
        result['output_config'] = options
    return result
