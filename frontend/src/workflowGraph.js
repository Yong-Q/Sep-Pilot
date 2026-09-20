// Graph replacement is atomic and scoped; execution facts belong to their plan.
export function selectGraph(workflow = {}) {
  const goal = workflow.goal_contract || {};
  const runtime = workflow.parallel_runtime || {};
  const approved = goal.approved_nodes || [];
  const chain = workflow.graph_projection || {};
  if (chain.source === 'persisted_chain' && chain.nodes?.length &&
      Number(chain.version || 0) >= Math.max(Number(goal.approved_plan_version || 0), Number(runtime.plan_version || 0))) return chain;
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
  const keep = complete(oldGraph) && (!complete(newGraph) || (!oldDraft && newDraft) ||
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
