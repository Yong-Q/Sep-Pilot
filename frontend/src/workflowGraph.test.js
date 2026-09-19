import { retainGraph, selectGraph } from './workflowGraph';

const snapshot = (version = 1, conv = 'a', username = 'u') => ({
  scope: { username, conv_id: conv },
  goal_contract: { approved_plan_version: version, approved_nodes: [
    { step_id: 'input', depends_on: [] },
    { step_id: 'result', depends_on: ['input'] },
  ] },
});

test('committed chain is the graph source including dependency changes', () => {
  const next = snapshot(2);
  next.graph_projection = {source: 'persisted_chain', version: 2, revision: 10,
    nodes: [{step_id: 'new-root', depends_on: []}]};
  expect(selectGraph(next)).toBe(next.graph_projection);
});

test('first approved plan replaces a repeatedly revised draft', () => {
  const draft = {scope: snapshot().scope, graph_projection: {source:'workflow_draft', version:99,
    nodes:[{step_id:'draft', depends_on:[]}]}};
  expect(retainGraph(draft, snapshot(1)).display_graph.source).toBe('approved');
});

test('empty updates retain a complete graph until a replacement arrives', () => {
  const initial = snapshot();
  const empty = { scope: initial.scope, graph_projection: { source: 'none', nodes: [] } };
  const retained = retainGraph(initial, empty);
  expect(retained.display_graph.nodes).toHaveLength(2);
  expect(retained.display_graph.retained).toBe(true);
  expect(retainGraph(retained, snapshot(2)).display_graph.version).toBe(2);
});

test('new approved topology wins over older running plan without stale success', () => {
  const next = snapshot(2);
  next.parallel_runtime = { plan_version: 1, nodes: {
    input: { status: 'succeeded', done: true, job_ids: ['old'] },
  } };
  const graph = selectGraph(next);
  expect(graph.nodes).toHaveLength(2);
  expect(graph.nodes[0].status).toBe('pending');
  expect(graph.nodes[0].job_ids).toBeUndefined();
});

test('empty snapshots never borrow another conversation or user graph', () => {
  for (const scope of [{username: 'u', conv_id: 'b'}, {username: 'v', conv_id: 'a'}]) {
    expect(retainGraph(snapshot(), {scope}).display_graph).toBeUndefined();
  }
});

test('late older plans cannot replace newer graph and matching states do update', () => {
  expect(retainGraph(snapshot(3), snapshot(2)).display_graph.version).toBe(3);
  const next = snapshot(3);
  next.parallel_runtime = { plan_version: 3, nodes: { input: { status: 'succeeded' } } };
  expect(retainGraph(snapshot(3), next).display_graph.nodes[0].status).toBe('succeeded');
});
