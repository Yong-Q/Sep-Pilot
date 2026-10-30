"""Append changing runtime facts to a model-only view, after stable history."""
import copy


def with_runtime_context(messages, context):
    view = copy.deepcopy(messages)
    if not context:
        return view
    block = {'type': 'text', 'text': '[CURRENT HOST RUNTIME CONTEXT]\n' + context}
    if view and view[-1].get('role') == 'user':
        content = view[-1].get('content')
        if isinstance(content, list):
            content.append(block)
        elif isinstance(content, str):
            view[-1]['content'] = [{'type': 'text', 'text': content}, block]
        else:
            view.append({'role': 'user', 'content': [block]})
    else:
        view.append({'role': 'user', 'content': [block]})
    return view
