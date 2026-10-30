"""Bounded advisory event ordering; never a submission or acceptance grant."""
import math


_LIMIT = 512


def _graph(state):
    nodes = state.get('nodes')
    if not isinstance(nodes, dict) or len(nodes) > _LIMIT:
        raise ValueError('unsupported advisory graph')
    children = {key: [] for key in nodes}
    indegrees = {}
    for key, node in nodes.items():
        deps = node.get('contract', {}).get('depends_on', [])
        if (not isinstance(deps, list) or any(not isinstance(dep, str) or dep not in nodes for dep in deps)
                or len(set(deps)) != len(deps)):
            raise ValueError('advisory dependency graph is incomplete')
        indegrees[key] = len(deps)
        for dep in deps:
            children[dep].append(key)
    queue = [key for key, count in indegrees.items() if count == 0]
    order = []
    for key in queue:
        order.append(key)
        for child in children[key]:
            indegrees[child] -= 1
            if not indegrees[child]:
                queue.append(child)
    if len(order) != len(nodes):
        raise ValueError('advisory graph contains a cycle')
    return nodes, children, order


def _scores(state):
    nodes, children, order = _graph(state)
    reach, depth = {}, {}
    for key in reversed(order):
        descendants = {child for child in children[key] if nodes[child].get('status') != 'succeeded'}
        for child in children[key]:
            descendants.update(reach[child])
        reach[key] = descendants
        depth[key] = max((depth[child] + 1 for child in children[key]), default=0)
    scores = {}
    for key, node in nodes.items():
        immediate = sum(nodes[child].get('status') == 'pending' and all(
            dep == key or nodes[dep].get('status') == 'succeeded'
            for dep in nodes[child]['contract'].get('depends_on', [])) for child in children[key])
        scores[key] = (int(immediate), len(reach[key]), depth[key])
    return scores


def review_score(state, step_id):
    """Count immediate unblocks, unfinished descendants, then remaining depth."""
    try:
        return _scores(state).get(step_id, (0, 0, 0))
    except (ValueError, TypeError, KeyError, AttributeError):
        return (0, 0, 0)


def order_events(events, state):
    """Safety reconciliation first, then current prefinish review, then FIFO.

    This only changes selection order. Consumers must revalidate native owner,
    attempt and authority before acting; arbitrary payload labels grant nothing.
    """
    ordered = list(events)
    try:
        scores = _scores(state)
        def key(pair):
            index, event = pair
            step = (event.get('payload') or {}).get('step_id')
            node = state['nodes'].get(step, {})
            kind = event.get('kind')
            safety = kind in {'worker_uncertain', 'tool_uncertain'}
            review = (kind in {'worker_prefinish', 'worker_scientific_owner_review'}
                      and node.get('status') == 'prefinish')
            stamp = event.get('time', index)
            if type(stamp) not in (float, int) or not math.isfinite(stamp):
                stamp = index
            score = scores.get(step, (0, 0, 0)) if review else (0, 0, 0)
            return (0 if safety else 1 if review else 2, *(-v for v in score), stamp, index)
        return [event for _, event in sorted(enumerate(ordered), key=key)]
    except (ValueError, TypeError, KeyError, AttributeError):
        return ordered
