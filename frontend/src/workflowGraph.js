// Graph replacement is atomic and scoped; execution facts belong to their plan.
export function selectGraph(workflow = {}) {
  const goal = workflow.goal_contract || {};
  const runtime = workflow.parallel_runtime || {};
  const approved = goal.approved_nodes || [];
  const chain = workflow.graph_projection || {};
  if (chain.source === 'persisted_chain' && chain.nodes?.length &&
      Number(chain.version || 0) >= Math.max(Number(goal.approved_plan_version || 0), Number(runtime.plan_version || 0))) {
    if (Number(chain.version || 0) !== Number(runtime.plan_version || 0) || !Object.keys(runtime.nodes || {}).length) return chain;
    return { ...chain, nodes: chain.nodes.map(node => ({
      ...node, ...(runtime.nodes?.[node.step_id] || {}),
      step_id: node.step_id,
      depends_on: node.depends_on || node.contract?.depends_on || [],
    })) };
  }
  if (approved.length && Number(goal.approved_plan_version) >= Number(runtime.plan_version || 0)) {
    const matches = goal.approved_plan_version === runtime.plan_version;
    return { version: goal.approved_plan_version, source: 'approved', nodes: approved.map(node => ({
      ...node, status: 'pending', done: false,
      ...(matches ? runtime.nodes?.[node.step_id] || {} : {}),
      step_id: node.step_id, depends_on: node.depends_on || [],
    })) };
  }
  if (Object.keys(runtime.nodes || {}).length) return {
    version: runtime.plan_version, source: 'runtime',
    nodes: Object.entries(runtime.nodes).map(([step_id, node]) => ({ ...node, step_id })),
  };
  return workflow.graph_projection || { nodes: [], source: 'none' };
}

export function retainGraph(previous, next) {
  if (!previous || previous.scope?.username !== next.scope?.username ||
      previous.scope?.conv_id !== next.scope?.conv_id) return next;
  const oldGraph = previous.display_graph || selectGraph(previous);
  const newGraph = selectGraph(next);
  const complete = graph => graph.nodes?.length && graph.source !== 'workflow_planning_state';
  const oldDraft = oldGraph.source === 'workflow_draft';
  const newDraft = newGraph.source === 'workflow_draft';
  const partialSameVersion = complete(oldGraph) && complete(newGraph) &&
    Number(newGraph.version || 0) === Number(oldGraph.version || 0) &&
    new Set((newGraph.nodes || []).map(node => node.step_id)).size <
      new Set((oldGraph.nodes || []).map(node => node.step_id)).size;
  const keep = complete(oldGraph) && (!complete(newGraph) || partialSameVersion || (!oldDraft && newDraft) ||
    oldDraft === newDraft && Number(newGraph.version || 0) < Number(oldGraph.version || 0));
  return { ...next, display_graph: keep ? { ...oldGraph, retained: true } : newGraph };
}

export function graphViewportKey(workflow = {}, graph = selectGraph(workflow)) {
  const scope = workflow.scope || {};
  const topology = (graph.nodes || []).map(node => [
    node.step_id,
    [...(node.depends_on || node.contract?.depends_on || [])].sort(),
  ]);
  return JSON.stringify([
    scope.username || '', scope.conv_id || '', Number(graph.version || 0), topology,
  ]);
}

const CACHE_PREFIX = 'sep_pilot_workflow_v1:';

function cacheKey(username, convId) {
  return CACHE_PREFIX + JSON.stringify([username || '', convId || '']);
}

// Keep only the durable display contract. Task history and evidence can be large
// and are fetched from the server after the first paint.
export function saveCachedWorkflow(storage, workflow = {}) {
  const scope = workflow.scope || {};
  const graph = workflow.display_graph || selectGraph(workflow);
  if (!storage || !scope.username || !scope.conv_id || !graph.nodes?.length) return false;
  try {
    storage.setItem(cacheKey(scope.username, scope.conv_id), JSON.stringify({
      scope,
      snapshot_at: Number(workflow.snapshot_at || 0),
      goal_contract: workflow.goal_contract || {},
      current_activity: workflow.current_activity || {},
      display_graph: graph,
    }));
    return true;
  } catch (_) {
    return false;
  }
}

export function loadCachedWorkflow(storage, username, convId) {
  if (!storage || !username || !convId) return null;
  try {
    const value = JSON.parse(storage.getItem(cacheKey(username, convId)) || 'null');
    if (value?.scope?.username !== username || value?.scope?.conv_id !== convId ||
        !value.display_graph?.nodes?.length) return null;
    return value;
  } catch (_) {
    return null;
  }
}
