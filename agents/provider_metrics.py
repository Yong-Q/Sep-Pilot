"""Content-free token/cache counters from provider responses or usage objects."""

_MISSING = object()
_COUNTERS = (
    'input_tokens', 'output_tokens', 'prompt_tokens', 'completion_tokens', 'total_tokens',
    'cache_read_input_tokens', 'cache_creation_input_tokens',
    'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens',
)
_DETAILS = (
    ('prompt_tokens_details', 'cached_tokens'),
    ('input_tokens_details', 'cached_tokens'),
    ('output_tokens_details', 'reasoning_tokens'),
)


def _field(value, name):
    if isinstance(value, dict):
        return value.get(name, _MISSING)
    try:
        result = getattr(value, name, _MISSING)
        if result is not _MISSING:
            return result
        extra = getattr(value, 'model_extra', None)
    except Exception:
        return _MISSING
    return extra.get(name, _MISSING) if isinstance(extra, dict) else _MISSING


def _counter(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def provider_usage_metrics(response_or_usage):
    """Return only explicitly reported, nonnegative integer token counters.

    Attribute access is limited to the known usage schema: no content, prompt,
    headers, credentials, or whole-response serialization is inspected. Missing
    counters remain absent. In particular, do not infer cache misses from
    input_tokens: provider-compatible schemas may define its total differently.
    """
    usage = _field(response_or_usage, 'usage')
    if usage is _MISSING:
        usage = response_or_usage
    result = {}
    for name in _COUNTERS:
        value = _field(usage, name)
        if _counter(value):
            result[name] = value
    for parent, name in _DETAILS:
        value = _field(_field(usage, parent), name)
        if _counter(value):
            result[parent + '.' + name] = value
    hits = {result[name] for name in (
        'prompt_cache_hit_tokens', 'cache_read_input_tokens',
        'prompt_tokens_details.cached_tokens', 'input_tokens_details.cached_tokens',
    ) if name in result}
    if len(hits) == 1:
        result['cache_hit_tokens'] = hits.pop()
    if 'prompt_cache_miss_tokens' in result:
        result['cache_miss_tokens'] = result['prompt_cache_miss_tokens']
    if 'cache_creation_input_tokens' in result:
        result['cache_write_tokens'] = result['cache_creation_input_tokens']
    return result
