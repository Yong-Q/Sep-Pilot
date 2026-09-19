import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Background, Controls, Handle, MarkerType, MiniMap, Position, ReactFlow } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './App.css';
import { selectGraph, retainGraph } from './workflowGraph';
import { readJsonResponse } from './apiResponse';

const API = '/api';
const TOKEN_KEY = 'bimem_token';
const USER_KEY = 'bimem_user';

function getToken() { return sessionStorage.getItem(TOKEN_KEY); }
function setToken(t) { sessionStorage.setItem(TOKEN_KEY, t); }
function clearToken() { sessionStorage.removeItem(TOKEN_KEY); }

function authHeaders(extra) {
  const h = extra || {};
  h['Content-Type'] = 'application/json';
  const t = getToken();
  if (t) h['Authorization'] = 'Bearer ' + t;
  return h;
}

function relativeTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const now = new Date();
  const diff = now - d;
  if (diff < 60000) return '刚刚';
  if (diff < 3600000) return Math.floor(diff / 60000) + '分钟前';
  if (diff < 86400000) return Math.floor(diff / 3600000) + '小时前';
  return (d.getMonth() + 1) + '/' + d.getDate();
}

function SmartResponseRenderer({ text }) {
  if (!text) return null;
  const parts = text.split(/\n\n+/).filter(Boolean);
  return (
    <div className="response-text">
      {parts.map((p, i) => {
        const lines = p.split('\n').filter(l => l.trim());
        if (lines.length >= 2 && lines[0].includes('|') && lines.some(l => /^[\s|:-]+$/.test(l))) {
          const headerCells = lines[0].split('|').map(s => s.trim()).filter(Boolean);
          const bodyLines = lines.filter(l => !/^[\s|:-]+$/.test(l)).slice(1);
          return (
            <table key={i}>
              <thead><tr>{headerCells.map((c, j) => <th key={j}>{c}</th>)}</tr></thead>
              <tbody>{bodyLines.map((row, j) => (
                <tr key={j}>{row.split('|').map(s => s.trim()).filter(Boolean).map((c, k) => <td key={k}>{c}</td>)}</tr>
              ))}</tbody>
            </table>
          );
        }
        if (lines.length === 1 && lines[0].length < 40 && !lines[0].startsWith('•') && !lines[0].startsWith('#')) {
          return <div key={i} style={{ fontWeight: 600, marginBottom: '0.5rem' }}>{lines[0]}</div>;
        }
        if (lines.length > 1 && lines.every(l => l.startsWith('• ') || l.startsWith('- '))) {
          return <ul key={i}>{lines.map((l, j) => <li key={j}>{l.replace(/^[•\-]\s*/, '')}</li>)}</ul>;
        }
        return <p key={i}>{p}</p>;
      })}
    </div>
  );
}

function formatLogTime(t) {
  if (!t) return '';
  const d = new Date(t);
  if (isNaN(d.getTime())) return '';
  return d.toLocaleTimeString();
}

function ExecutionLogs({ logs }) {
  const [open, setOpen] = useState(false);
  if (!logs || logs.length === 0) return null;
  return (
    <div className="execution-logs">
      <button className="toggle-logs" onClick={() => setOpen(o => !o)}>
        {open ? '▼ 隐藏日志' : '▶ 日志 (' + logs.length + ')'}
      </button>
      {open && (
        <div className="logs-content">
          {logs.map((log, i) => (
            <div className="log-item" key={i}>
              <span className="log-time">{formatLogTime(log.time || log.timestamp)}</span>
              <span className="log-msg">{log.tool ? ('[' + log.tool + '] ') : ''}{log.message || ''}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Task panel（任务栏）──
const STATE_META = {
  RUNNING:   { color: '#22c55e', label: '运行中', icon: '▶', pulse: true },
  PENDING:   { color: '#eab308', label: '排队中', icon: '⏳' },
  QUEUED:    { color: '#eab308', label: '排队中', icon: '⏳' },
  COMPLETED: { color: '#3b82f6', label: '已完成', icon: '✓' },
  FAILED:    { color: '#ef4444', label: '失败',   icon: '✕' },
  TIMEOUT:   { color: '#f97316', label: '超时',   icon: '⏱' },
  CANCELLED: { color: '#71717a', label: '已取消', icon: '⊘' },
  UNKNOWN:   { color: '#71717a', label: '未知',   icon: '?' },
};

function parseJobTime(s) {
  if (!s) return null;
  const d = new Date(String(s).replace(' ', 'T'));
  return isNaN(d.getTime()) ? null : d;
}

function fmtTime(d) {
  if (!d) return '—';
  return (d.getMonth() + 1) + '-' + d.getDate() + ' ' +
    String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
}

function fmtElapsed(ms) {
  if (ms < 0) ms = 0;
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h > 0) return h + 'h ' + String(m).padStart(2, '0') + 'm';
  return m + 'm ' + String(sec).padStart(2, '0') + 's';
}

function TaskPanel({ tasks, activeConvId }) {
  const [hideAfter, setHideAfter] = useState(() => localStorage.getItem('task_hide_after') || '30m');
  const [onlySession, setOnlySession] = useState(() => localStorage.getItem('task_only_session') !== '0');
  const [expandedId, setExpandedId] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [explaining, setExplaining] = useState({});
  const [explain, setExplain] = useState({});

  useEffect(() => { localStorage.setItem('task_hide_after', hideAfter); }, [hideAfter]);
  useEffect(() => { localStorage.setItem('task_only_session', onlySession ? '1' : '0'); }, [onlySession]);

  const isActive = t => !t.terminal;
  // 1s live ticker while any task is still running（只刷新耗时数字）
  const anyActive = (tasks || []).some(t => !t.terminal);
  useEffect(() => {
    if (!anyActive) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [anyActive]);

  const visible = t => {
    if (isActive(t)) return true;                 // 运行中/排队中始终显示
    if (hideAfter === 'running') return false;    // 只显示运行中的
    if (hideAfter === 'always') return true;      // 始终显示
    const sub = parseJobTime(t.submitted_at);
    if (!sub) return true;
    const mins = { '5m': 5, '30m': 30, '1h': 60 }[hideAfter] || 0;
    return (now - sub.getTime()) < mins * 60000;  // 完成超过间隔则隐藏
  };

  let list = (tasks || []).filter(t => t.job_id);
  if (onlySession) list = list.filter(t => t.conv_id === activeConvId);
  list = list.filter(visible);
  list = [...list].sort((a, b) => {
    if (isActive(a) !== isActive(b)) return isActive(a) ? -1 : 1;
    return (b.submitted_at || '').localeCompare(a.submitted_at || '');
  });

  const doCancel = async jobId => {
    if (!window.confirm('确认取消作业 ' + jobId + '？（SLURM scancel，已算的进度保留）')) return;
    try {
      const r = await fetch(API + '/jobs/' + jobId + '/cancel', { method: 'POST', headers: authHeaders() });
      const d = await readJsonResponse(r);
      window.alert(d.message || (d.cancelled ? '已发送取消' : '取消失败'));
    } catch (e) { window.alert('取消请求失败: ' + (e.message || e)); }
  };

  // 委派只读 monitor 小智能体检查作业进度（后端 /api/jobs/{id}/explain）
  const doExplain = async jobId => {
    if (explaining[jobId]) return;
    setExplaining(s => ({ ...s, [jobId]: true }));
    setExplain(s => ({ ...s, [jobId]: '' }));
    try {
      const r = await fetch(API + '/jobs/' + jobId + '/explain', { headers: authHeaders() });
      const d = await readJsonResponse(r);
      setExplain(s => ({ ...s, [jobId]: d.error ? ('⚠️ ' + d.error) : (d.verdict || '（monitor 未返回结论）') }));
    } catch (e) {
      setExplain(s => ({ ...s, [jobId]: '⚠️ 请求失败: ' + (e.message || e) }));
    } finally {
      setExplaining(s => ({ ...s, [jobId]: false }));
    }
  };

  return (
    <div className="task-panel">
      <div className="task-panel-header">
        <span className="task-panel-title">📋 任务</span>
        <div className="task-panel-controls">
          <label className="task-session-toggle" title="只显示当前会话提交的作业">
            <input type="checkbox" checked={onlySession} onChange={e => setOnlySession(e.target.checked)} />
            <span>当前会话</span>
          </label>
          <select value={hideAfter} onChange={e => setHideAfter(e.target.value)} title="已完成任务保留多久后隐藏">
            <option value="running">仅运行中</option>
            <option value="5m">保留5分钟</option>
            <option value="30m">保留30分钟</option>
            <option value="1h">保留1小时</option>
            <option value="always">始终显示</option>
          </select>
        </div>
      </div>
      <div className="task-list">
        {list.length === 0 && (
          <div className="task-empty">
            {onlySession ? '当前会话暂无任务' : '暂无任务'}
          </div>
        )}
        {list.map(t => {
          const meta = STATE_META[t.state] || STATE_META.UNKNOWN;
          const active = isActive(t);
          const sub = parseJobTime(t.submitted_at);
          const upd = parseJobTime(t.updated_at);
          const elapsed = sub ? now - sub.getTime() : 0;
          const cif = (t.cif || '').split('/').pop() || '';
          const expanded = expandedId === t.job_id;
          return (
            <div key={t.job_id} className={'task-item' + (expanded ? ' task-item-open' : '')}>
              <div className="task-row" onClick={() => setExpandedId(expanded ? null : t.job_id)}
                   title={t.job_id + '（点击查看进度）'}>
                <span className={'task-dot' + (meta.pulse ? ' task-dot-pulse' : '')} style={{ background: meta.color }} />
                <span className="task-title">{t.gas ? (t.gas + ' · ' + (cif || t.job_id)) : t.job_id}</span>
                <span className="task-state" style={{ color: meta.color }}>{meta.label}</span>
                {t.state === 'PENDING' && t.resource_warning && <span title={t.pending_reason || '排队原因待查证'}>⚠️</span>}
                {active && <span className="task-elapsed">{fmtElapsed(elapsed)}</span>}
              </div>
              {expanded && (
                <div className="task-detail">
                  <div className="task-detail-grid">
                    <span className="task-k">作业号</span><span className="task-v">{t.job_id}</span>
                    <span className="task-k">状态</span><span className="task-v" style={{ color: meta.color }}>{meta.icon} {t.state}</span>
                    <span className="task-k">提交</span><span className="task-v">{fmtTime(sub)}</span>
                    <span className="task-k">更新</span><span className="task-v">{fmtTime(upd)}</span>
                    {t.state === 'PENDING' && <React.Fragment><span className="task-k">排队原因</span><span className="task-v">{t.pending_reason || '待调度器返回原因'}{t.pending_age_seconds ? ' · ' + fmtElapsed(t.pending_age_seconds * 1000) : ''}</span></React.Fragment>}
                    {t.gas ? <React.Fragment><span className="task-k">气体</span><span className="task-v">{t.gas}</span></React.Fragment> : null}
                    {t.temperature ? <React.Fragment><span className="task-k">温度</span><span className="task-v">{t.temperature} K</span></React.Fragment> : null}
                    <span className="task-k">目录</span><span className="task-v task-dir" title={t.work_dir}>{t.work_dir || '—'}</span>
                  </div>
                  {t.diagnosis && t.diagnosis.cause && (
                    <div className="task-diag">⚠️ 推测原因：{t.diagnosis.cause}</div>
                  )}
                  {(active || t.failed) && (
                    <div className="task-actions">
                      {active && (
                        <button className="task-cancel" onClick={() => doCancel(t.job_id)}>⛔ 取消作业</button>
                      )}
                      <button className="task-explain" onClick={() => doExplain(t.job_id)} disabled={explaining[t.job_id]}>
                        {explaining[t.job_id] ? '⏳ AI 分析中...' : (t.state === 'PENDING' ? '🔍 AI 查排队原因' : '🔍 AI 看进度')}
                      </button>
                    </div>
                  )}
                  {(explain[t.job_id] || explaining[t.job_id]) && (
                    <div className="task-explain-result">
                      {explaining[t.job_id] && !explain[t.job_id] ? (
                        <div className="task-explain-loading">monitor 正在检查作业状态与日志…</div>
                      ) : (
                        <div className="task-explain-text">{explain[t.job_id]}</div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const NODE_STATUS = {
  draft: ['方案草案', '○'],
  completed: ['已完成', '✓'], succeeded: ['已完成', '✓'], prefinish: ['结果检查中', '◐'],
  failed: ['需要处理', '!'], running: ['正在执行', '▶'], submitted: ['计算进行中', '▶'],
  pending: ['等待前一步', '○'], waiting_jobs: ['等待计算', '◌'], waiting_prerequisite: ['等待依赖', '○'],
  needs_resources: ['准备计算资源', '◌'], uncertain: ['状态待核对', '?'], cancelled: ['已取消', '⊘'],
};

function roleView(role) {
  if (!role) return { label: '已配置', tone: 'idle', detail: '心跳待取得' };
  const age = role.service_heartbeat ? Date.now() / 1000 - role.service_heartbeat : null;
  if (age !== null && age > 30) return { label: '待复核', tone: 'warn', detail: '心跳超过 30 秒' };
  if (role.busy_event_id || role.last_action === 'processing') {
    return { label: '处理中', tone: 'busy', detail: role.last_action || '正在处理事件' };
  }
  return { label: '在线', tone: 'ok', detail: role.last_action || '待命' };
}

function agentDisplayName(agent) {
  const names = {
    'lead-orchestrator': '主智能体',
    supervisor: '监督智能体',
    adsorption: '吸附计算智能体',
    resources: '资源调度智能体',
    resource: '资源调度智能体',
    structure: '结构处理智能体',
    analysis: '数据分析智能体',
  };
  return names[agent] || agent || '主智能体';
}

const TOOL_LABELS = {
  inspect_path: '检查输入', inspect_run: '检查作业', grep_search: '定位产物', read_file: '读取结果',
  generate_structure: '生成结构', run_pacman_charge: '框架赋电荷', validate_framework_charges: '电荷验收',
  run_cdft: 'cDFT 计算', run_gcmc_isotherm: 'GCMC 等温线', run_gcmc_batch: 'GCMC 批量筛选',
  run_pore_analysis: '孔结构分析', run_henry: 'Henry 系数', query_literature: '文献核验',
};

function nodeLabel(node) {
  return node.description || TOOL_LABELS[node.tool || node.contract?.tool] || node.step_id;
}

function compactValue(value) {
  if (Array.isArray(value)) return value.join(' / ');
  if (value && typeof value === 'object') return Object.entries(value).map(([k, v]) => k + '=' + compactValue(v)).join(' · ');
  return String(value);
}

function nodeDependencies(node) {
  return node.depends_on || node.contract?.depends_on || [];
}

function nodeStatus(node) {
  if (node.status === true || node.done === true) return 'succeeded';
  if (node.status === false) return 'pending';
  return node.status || 'pending';
}

function workflowLevels(nodes) {
  const known = new Map(nodes.map(node => [node.step_id, node]));
  const levels = new Map();
  let pending = nodes.slice();
  let guard = nodes.length + 2;
  while (pending.length && guard-- > 0) {
    const next = [];
    let changed = false;
    pending.forEach(node => {
      const deps = nodeDependencies(node).filter(dep => known.has(dep));
      if (deps.every(dep => levels.has(dep))) {
        levels.set(node.step_id, deps.length ? Math.max(...deps.map(dep => levels.get(dep))) + 1 : 0);
        changed = true;
      } else next.push(node);
    });
    if (!changed) {
      const fallback = Math.max(-1, ...levels.values()) + 1;
      next.forEach(node => levels.set(node.step_id, fallback));
      break;
    }
    pending = next;
  }
  const grouped = [];
  nodes.forEach(node => {
    const level = levels.get(node.step_id) || 0;
    if (!grouped[level]) grouped[level] = [];
    grouped[level].push(node);
  });
  return grouped.filter(Boolean);
}

function nodeTasks(node) {
  const explicit = node.tasks || node.sub_tasks || node.contract?.tasks || node.contract?.sub_tasks;
  if (Array.isArray(explicit) && explicit.length) return explicit.map((task, index) => ({
    task_id: task.task_id || task.id || task.name || ('task-' + (index + 1)),
    label: task.description || task.label || task.name || task.task_id || ('任务 ' + (index + 1)),
    status: task.status || nodeStatus(node),
    job_id: task.job_id || '',
    path: task.result_path || task.output_path || '',
  }));
  if ((node.job_ids || []).length) return node.job_ids.map(jobId => ({
    task_id: String(jobId), label: '作业 ' + jobId, status: nodeStatus(node), job_id: jobId, path: '',
  }));
  return [{ task_id: node.step_id, label: node.description || node.step_id, status: nodeStatus(node), job_id: '', path: '' }];
}

function WorkflowNode({ data, selected }) {
  const status = data.status;
  const meta = NODE_STATUS[status] || [status, '○'];
  return (
    <div className={'workflow-node status-' + status + (selected ? ' selected' : '')}
         title={'原始状态: ' + status + '\n依赖: ' + (data.dependencies.join(', ') || '无')}>
      <Handle type="target" position={Position.Left} className="workflow-handle" />
      <div className="workflow-node-index">{String(data.stage).padStart(2, '0')}</div>
      <div className="workflow-node-copy">
        <b>{data.label}</b>
        <small>{data.agent} · {data.taskCount} 项</small>
      </div>
      <div className="workflow-node-state" title={meta[0]}>{meta[1]}</div>
      <Handle type="source" position={Position.Right} className="workflow-handle" />
    </div>
  );
}

const WORKFLOW_NODE_TYPES = { workflow: WorkflowNode };

function graphGeometry(levels) {
  const nodeWidth = 152, nodeHeight = 54, columnGap = 76, rowGap = 30, padding = 22;
  const tallest = Math.max(1, ...levels.map(level => level.length));
  const height = Math.max(164, padding * 2 + tallest * nodeHeight + (tallest - 1) * rowGap);
  const width = Math.max(320, padding * 2 + levels.length * nodeWidth + Math.max(0, levels.length - 1) * columnGap);
  const positions = {};
  levels.forEach((level, column) => {
    const stackHeight = level.length * nodeHeight + Math.max(0, level.length - 1) * rowGap;
    const startY = (height - stackHeight) / 2;
    level.forEach((node, row) => {
      positions[node.step_id] = {
        x: padding + column * (nodeWidth + columnGap),
        y: startY + row * (nodeHeight + rowGap),
      };
    });
  });
  return { width, height, nodeWidth, nodeHeight, positions };
}

function timestampMs(value) {
  if (!value) return null;
  if (typeof value === 'number') return value > 1e12 ? value : value * 1000;
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : null;
}

function nodeElapsed(node) {
  const start = timestampMs(node.started_at || node.start_time || node.created_at);
  if (!start) return '尚未开始';
  const status = nodeStatus(node);
  const active = ['running', 'submitted', 'waiting_jobs', 'prefinish', 'uncertain'].includes(status);
  const end = timestampMs(node.finished_at || node.completed_at || node.updated_at) || (active ? Date.now() : start);
  return fmtElapsed(Math.max(0, end - start));
}

function OperationsDrawer({ workflow, partial, processing }) {
  const [open, setOpen] = useState(() => localStorage.getItem('runtime_drawer_open') !== '0');
  const [selectedStepId, setSelectedStepId] = useState('');
  useEffect(() => { localStorage.setItem('runtime_drawer_open', open ? '1' : '0'); }, [open]);
  workflow = workflow || {};
  const goal = workflow.goal_contract || {};
  const lines = workflow.lines || [];
  const steps = lines.flatMap(l => (l.steps || []).map(s => ({ ...s, line_id: l.line_id })));
  const branches = workflow.error_branches || [];
  const recoveries = Object.values(workflow.recovery || {});
  const lifetime = workflow.lifecycle || {};
  const patch = workflow.pending_workflow_patch;
  const runtimeNodes = Object.entries(workflow.parallel_runtime?.nodes || {}).map(([step_id, node]) => ({ step_id, ...node }));
  const stepById = new Map(steps.map(step => [step.step_id, step]));
  const approvedNodes = (goal.approved_nodes || []).map(contract => {
    const recorded = stepById.get(contract.step_id) || {};
    return { ...contract, ...recorded, step_id: contract.step_id,
      depends_on: contract.depends_on || recorded.depends_on || [] };
  });
  const projectedNodes = ['persisted_legacy_chain', 'workflow_planning_state'].includes(workflow.graph_projection?.source)
    ? (workflow.graph_projection.nodes || []) : [];
  // A historical TaskLine also contains flat diagnostic/tool logs.  Those
  // records have no dependency contract and must not be drawn as a fake column
  // of graph nodes.  Only executor runtime or user-approved DAG contracts are
  // authoritative graph data; flat calls stay in the tool/event sections.
  const displayGraph = workflow.display_graph || selectGraph(workflow);
  const nodes = displayGraph.nodes || [];
  const unstructuredHistoryCount = nodes.length ? 0 : steps.length;
  const graphLevels = workflowLevels(nodes);
  const graph = graphGeometry(graphLevels);
  const activeStates = new Set(['running', 'submitted', 'waiting_jobs', 'waiting_prerequisite', 'uncertain', 'prefinish', 'needs_resources']);
  const activeNodes = nodes.filter(n => activeStates.has(nodeStatus(n)));
  const operations = (workflow.recent_operations || []).slice(-10).reverse();
  const activity = workflow.current_activity || {};
  const openBranches = branches.filter(b => !['resolved', 'closed'].includes(b.status));
  const failedOperations = operations.filter(o => o.failed);
  const warningCount = openBranches.length + failedOperations.length;
  const roleEntries = [
    ['主智能体', (lifetime.agents || {}).main_chat],
    ['监督智能体', (lifetime.agents || {}).supervisor],
  ];
  const pathRows = [];
  nodes.forEach(node => {
    const manifest = node.path_manifest || {};
    ['calculation_dir', 'result_dir', 'checkpoint_path', 'evidence_path'].forEach(key => {
      if (manifest[key]) pathRows.push([node.step_id, key, manifest[key]]);
    });
    (node.output_files || []).forEach(path => pathRows.push([node.step_id, 'output', path]));
  });
  // The lead agent owns the conversation, but it is not necessarily the
  // worker currently doing the scientific operation. Prefer authoritative
  // activity/runtime-node ownership so delegation is visible to the user.
  const liveAgentId = activity.agent || activeNodes[0]?.agent || activeNodes[0]?.contract?.agent
    || partial?.current_agent || 'lead-orchestrator';
  const liveAgent = agentDisplayName(liveAgentId);
  const liveStatus = activity.status || (processing ? '正在执行' : '待命');
  useEffect(() => {
    if (!nodes.some(node => node.step_id === selectedStepId)) {
      const preferred = activeNodes[0] || nodes[0];
      setSelectedStepId(preferred?.step_id || '');
    }
  }, [workflow.scope?.conv_id, nodes.map(node => node.step_id).join('|'), selectedStepId]);
  const selectedNode = nodes.find(node => node.step_id === selectedStepId);
  const selectedStatus = selectedNode ? nodeStatus(selectedNode) : '';
  const selectedMeta = NODE_STATUS[selectedStatus] || [selectedStatus, '○'];
  const selectedBranches = selectedNode ? branches.filter(branch =>
    [branch.failed_step, branch.step_id, branch.node_id].includes(selectedNode.step_id)) : [];
  const selectedRecoveries = selectedNode ? recoveries.filter(item =>
    [item.step_id, item.node_id, item.task_id].includes(selectedNode.step_id)
    || (item.tool && item.tool === (selectedNode.tool || selectedNode.contract?.tool))) : [];
  const selectedArgs = selectedNode?.arguments || selectedNode?.contract?.arguments || {};
  const selectedPaths = selectedNode?.path_manifest || {};
  const selectedTasks = selectedNode ? nodeTasks(selectedNode) : [];
  const flowNodes = nodes.map(node => ({
    id: node.step_id,
    type: 'workflow',
    position: graph.positions[node.step_id] || { x: 0, y: 0 },
    data: {
      label: nodeLabel(node),
      status: nodeStatus(node),
      dependencies: nodeDependencies(node),
      taskCount: nodeTasks(node).length,
      agent: agentDisplayName(node.agent || node.contract?.agent || '待分配'),
      stage: nodes.findIndex(item => item.step_id === node.step_id) + 1,
    },
    selected: selectedStepId === node.step_id,
    draggable: false,
  }));
  const flowEdges = nodes.flatMap(node => nodeDependencies(node).map(dep => {
    if (!graph.positions[dep] || !graph.positions[node.step_id]) return null;
    const status = nodeStatus(node);
    const active = ['running', 'submitted', 'waiting_jobs'].includes(status);
    const color = status === 'failed' ? '#ef4444'
      : status === 'prefinish' ? '#f59e0b'
      : ['succeeded', 'completed'].includes(status) ? '#22c55e'
      : active ? '#22d3ee' : '#64748b';
    return {
      id: dep + '->' + node.step_id,
      source: dep,
      target: node.step_id,
      type: 'smoothstep',
      animated: active,
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
      style: { stroke: color, strokeWidth: active ? 1.8 : 1.35, opacity: status === 'succeeded' ? 0.65 : 0.85 },
    };
  }).filter(Boolean));
  const graphHeight = Math.min(510, Math.max(250, graph.height + 42));

  return (
    <aside className={'runtime-drawer' + (open ? ' runtime-drawer-open' : ' runtime-drawer-closed')} aria-label="智能体运行面板">
      <button className="runtime-drawer-toggle" onClick={() => setOpen(v => !v)}
              title={open ? '收起运行面板' : '展开运行面板'}>
        <span className={'runtime-live-dot' + (processing ? ' is-live' : '')} />
        {open ? <React.Fragment><span>运行面板</span><span className="runtime-chevron">›</span></React.Fragment> : <span className="runtime-vertical-label">运行</span>}
      </button>
      {open && (
        <div className="runtime-drawer-body">
          <section className="runtime-hero">
            <div className="runtime-eyebrow">当前活动</div>
            <div className="runtime-current-activity">
              <span className={'runtime-live-dot' + (processing ? ' is-live' : '')} />
              <b>{liveAgent}</b><em>{liveStatus}</em>
            </div>
            {activity.waiting_for && <div className="runtime-waiting">等待：{activity.waiting_for}</div>}
          </section>

          <section className="runtime-section">
            <div className="runtime-section-title"><span>全周期角色</span><span className="runtime-section-note">始终可见</span></div>
            <div className="runtime-role-list">
              {roleEntries.map(([name, role]) => {
                const view = roleView(role);
                return <div className="runtime-role" key={name}>
                  <span className={'runtime-role-dot ' + view.tone} />
                  <div><b>{name}</b><small>{view.detail}</small></div>
                  <span className={'runtime-role-state ' + view.tone}>{view.label}</span>
                </div>;
              })}
            </div>
          </section>

          <section className="runtime-section">
            <div className="runtime-section-title"><span>科研目标</span>{goal.version ? <span className="runtime-badge">v{goal.version}</span> : null}</div>
            <p className="runtime-goal">{goal.active_goal || goal.original_goal || '新对话尚未建立科研目标'}</p>
            <div className="runtime-chips">
              {goal.method && <span>{goal.method}</span>}
              {(goal.gases || []).map(g => <span key={g}>{g}</span>)}
              {goal.approved_plan_version ? <span>已批准 v{goal.approved_plan_version}</span> : null}
              {workflow.awaiting_plan_approval || patch ? <span className="warn">待协商</span> : null}
            </div>
            {goal.parameters && Object.keys(goal.parameters).length > 0 && <div className="runtime-kv-list">
              {Object.entries(goal.parameters).map(([key, value]) => <div key={key}><span>{key}</span><b>{compactValue(value)}</b></div>)}
            </div>}
          </section>

          <section className="runtime-section">
            <div className="runtime-section-title"><span>有向编排图</span><span className="runtime-section-note">{activeNodes.length} 活动 / {nodes.length} 节点</span></div>
            {displayGraph.source === 'workflow_draft' && <div className="runtime-graph-recovered">研究方案草案；待补齐参数并编译后执行。</div>}
            {displayGraph.retained && <div className="runtime-graph-recovered">正在同步更新，保留上次完整编排图。</div>}
            {approvedNodes.length > 0 && goal.approved_plan_version !== workflow.parallel_runtime?.plan_version &&
              <div className="runtime-graph-recovered">方案 v{goal.approved_plan_version} 已保存，执行状态待同步。</div>}
            {workflow.graph_projection?.source === 'persisted_legacy_chain' && projectedNodes.length > 0 && runtimeNodes.length === 0 && approvedNodes.length === 0 &&
              <div className="runtime-graph-recovered">已从本会话持久化执行链恢复；后续修改将进入正式 DAG</div>}
            {workflow.graph_projection?.source === 'workflow_planning_state' &&
              <div className="runtime-graph-planning">科研目标已持久化；这里显示真实编排阶段，不代表 DAG 已获批准或开始执行。</div>}
            <div className="runtime-status-legend" aria-label="节点状态图例">
              <span className="is-running">● 执行中</span><span className="is-prefinish">◐ 验收中</span>
              <span className="is-success">● 成功</span><span className="is-failed">● 异常</span>
            </div>
            <div className="runtime-graph-viewport" style={nodes.length ? { height: graphHeight } : undefined}>
              {nodes.length === 0 && <div className="runtime-empty runtime-graph-empty">
                <b>尚未形成结构化 DAG</b>
                <span>{unstructuredHistoryCount ? `检测到 ${unstructuredHistoryCount} 条历史 tool 记录，已避免把它们误画成编排节点。` : '编排批准后会在这里显示有向节点、分支与汇合。'}</span>
              </div>}
              {nodes.length > 0 && <ReactFlow
                nodes={flowNodes}
                edges={flowEdges}
                nodeTypes={WORKFLOW_NODE_TYPES}
                onNodeClick={(_, node) => setSelectedStepId(node.id)}
                fitView
                fitViewOptions={{ padding: 0.24, duration: 360 }}
                minZoom={0.35}
                maxZoom={1.8}
                nodesDraggable={false}
                nodesConnectable={false}
                panOnScroll
                zoomOnScroll
                proOptions={{ hideAttribution: true }}
                aria-label="可缩放的有向编排图"
              >
                <Background gap={18} size={1} color="rgba(148,163,184,.10)" />
                <Controls showInteractive={false} position="bottom-left" />
                {nodes.length > 10 && <MiniMap
                  pannable zoomable position="bottom-right"
                  nodeColor={node => {
                    const status = node.data?.status;
                    if (status === 'failed') return '#ef4444';
                    if (status === 'prefinish') return '#f59e0b';
                    if (['completed', 'succeeded'].includes(status)) return '#22c55e';
                    if (['running', 'submitted', 'waiting_jobs'].includes(status)) return '#22d3ee';
                    return '#64748b';
                  }}
                />}
              </ReactFlow>}
            </div>
            {selectedNode && <div className={'runtime-node-inspector status-' + selectedStatus}>
              <div className="runtime-inspector-head">
                <div><span>当前选中</span><b>{nodeLabel(selectedNode)}</b></div>
                <span className="runtime-inspector-state">{selectedMeta[1]} {selectedMeta[0]}</span>
              </div>
              <div className="runtime-inspector-grid">
                <div><span>Agent</span><b>{agentDisplayName(selectedNode.agent || selectedNode.contract?.agent || '待分配')}</b></div>
                <div><span>Tool</span><b>{selectedNode.tool || selectedNode.contract?.tool || '—'}</b></div>
                <div><span>耗时</span><b>{nodeElapsed(selectedNode)}</b></div>
                <div><span>原始状态</span><b>{selectedStatus}</b></div>
                <div><span>错误分支</span><b className={selectedBranches.length ? 'danger' : ''}>{selectedBranches.length}</b></div>
                <div><span>恢复/重试记录</span><b>{selectedRecoveries.length}</b></div>
              </div>
              <div className="runtime-inspector-line"><span>依赖</span><b>{nodeDependencies(selectedNode).join(' ← ') || '无（起点）'}</b></div>
              {(selectedNode.job_ids || []).length > 0 && <div className="runtime-inspector-line"><span>关联作业</span><b>{selectedNode.job_ids.join(', ')}</b></div>}
              <div className="runtime-inspector-tasks">
                <div className="runtime-inspector-subtitle">节点内任务 <span>{selectedTasks.length}</span></div>
                {selectedTasks.map((task, index) => <div className="runtime-inspector-task" key={task.task_id || index}>
                  <i className={'status-' + task.status} />
                  <div><b>{task.label}</b><small>{task.job_id ? 'job ' + task.job_id : task.task_id}</small></div>
                  <span>{(NODE_STATUS[task.status] || [task.status])[0]}</span>
                </div>)}
              </div>
              {selectedBranches.length > 0 && <div className="runtime-inspector-error">
                {selectedBranches.slice(-1).map((branch, i) => <React.Fragment key={branch.branch_id || i}>
                  <b>{branch.error || branch.diagnosis || '该节点有待处理错误'}</b>
                  {branch.resolution && <span>{branch.resolution}</span>}
                </React.Fragment>)}
              </div>}
              {(Object.keys(selectedPaths).length > 0 || Object.keys(selectedArgs).length > 0) && <details className="runtime-inspector-details">
                <summary>参数与路径</summary>
                {Object.keys(selectedPaths).length > 0 && <pre>{JSON.stringify(selectedPaths, null, 2)}</pre>}
                {Object.keys(selectedArgs).length > 0 && <pre>{JSON.stringify(selectedArgs, null, 2)}</pre>}
              </details>}
            </div>}
          </section>

          <section className="runtime-section">
            <div className="runtime-section-title"><span>最近工具</span><span className="runtime-section-note">{operations.length} 条</span></div>
            <div className="runtime-operation-list">
              {operations.length === 0 && <div className="runtime-empty">暂无工具调用</div>}
              {operations.map((op, i) => <div className={'runtime-operation' + (op.failed ? ' failed' : '')} key={op.call_id || i}>
                <span>{op.failed ? '!' : '✓'}</span>
                <div><b>{op.tool || '未知工具'}</b><small>{op.agent || 'agent'} · {op.time ? relativeTime(op.time) : '时间待取得'}</small></div>
              </div>)}
            </div>
          </section>

          {(warningCount > 0 || patch) && <section className="runtime-section runtime-alert-section">
            <div className="runtime-section-title"><span>需要关注</span><span className="runtime-badge warn">{warningCount + (patch ? 1 : 0)}</span></div>
            {patch && <div className="runtime-alert">编排有待处理修改：v{patch.base_version} → v{patch.new_version}<small>{patch.reason}</small></div>}
            {openBranches.slice(-4).map((b, i) => <div className="runtime-alert" key={b.branch_id || i}>{b.failed_step || b.branch_id || '错误分支'}<small>{b.resolution || b.diagnosis || b.error || '待诊断'}</small></div>)}
          </section>}

          {(pathRows.length > 0 || Object.keys(workflow.memory_files || {}).length > 0) && <section className="runtime-section">
            <details className="runtime-details">
              <summary>文件与证据路径</summary>
              {pathRows.slice(0, 16).map(([node, kind, path], i) => <div className="runtime-path" key={node + kind + i}><span>{node} · {kind}</span><code title={path}>{path}</code></div>)}
              {Object.entries(workflow.memory_files || {}).map(([kind, path]) => <div className="runtime-path" key={kind}><span>记忆 · {kind}</span><code title={path}>{path}</code></div>)}
            </details>
            <details className="runtime-details technical">
              <summary>技术详情</summary>
              <pre>{JSON.stringify({ nodes: nodes.slice(0, 12), recent_operations: operations, recovery: recoveries }, null, 2)}</pre>
            </details>
          </section>}
        </div>
      )}
    </aside>
  );
}

// ── Auth form ──
function AuthForm({ onLogin }) {
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setErr('');
    setLoading(true);
    try {
      const endpoint = mode === 'login' ? API + '/login' : API + '/register';
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const data = await readJsonResponse(res);
      if (!res.ok) { setErr(data.detail || 'Error'); return; }
      if (mode === 'login') {
        setToken(data.token);
        onLogin(data.username || username);
      } else {
        setErr('注册成功，请登录');
        setMode('login');
        setPassword('');
      }
    } catch (e2) { setErr(e2.message || '网络错误'); }
    finally { setLoading(false); }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-logo">🧪</div>
        <div className="auth-subtitle">BiMemAgent MOF · AI Research Platform</div>
        <h2>{mode === 'login' ? '欢迎回来' : '创建账户'}</h2>
        <form onSubmit={submit}>
          <div className="auth-field">
            <label>用户名</label>
            <input value={username} onChange={e => setUsername(e.target.value)} required autoFocus />
          </div>
          <div className="auth-field">
            <label>密码</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required />
          </div>
          {err && <div className="auth-error">{err}</div>}
          <button className="auth-btn" type="submit" disabled={loading}>{loading ? '...' : mode === 'login' ? '登录' : '注册'}</button>
        </form>
        <div className="auth-toggle">
          {mode === 'login' ? <span>没有账户？ <button onClick={() => { setMode('signup'); setErr(''); }}>注册</button></span>
            : <span>已有账户？ <button onClick={() => { setMode('login'); setErr(''); }}>登录</button></span>}
        </div>
      </div>
    </div>
  );
}

// ── Per-user model connection ──
function ModelConnectionDialog({ onClose }) {
  const [form, setForm] = useState({ mode: 'default', base_url: '', model: '', api_key: '', api_key_hint: '' });
  const [busy, setBusy] = useState(true);
  const [message, setMessage] = useState('');
  useEffect(() => {
    fetch(API + '/model-connection', { headers: authHeaders() })
      .then(r => r.ok ? readJsonResponse(r) : Promise.reject(new Error('读取配置失败')))
      .then(data => setForm({ ...data, api_key: '' }))
      .catch(error => setMessage(error.message))
      .finally(() => setBusy(false));
  }, []);
  const save = async () => {
    setBusy(true); setMessage('');
    try {
      const response = await fetch(API + '/model-connection', {
        method: 'PUT', headers: authHeaders(), body: JSON.stringify(form),
      });
      const data = await readJsonResponse(response);
      if (!response.ok) throw new Error(data.detail || '保存失败');
      setForm({ ...data, api_key: '' });
      setMessage('已安全保存；下一次模型调用生效');
    } catch (error) { setMessage(error.message); }
    finally { setBusy(false); }
  };
  return <div className="model-dialog-backdrop" onMouseDown={event => event.target === event.currentTarget && onClose()}>
    <section className="model-dialog" role="dialog" aria-modal="true" aria-label="模型连接设置">
      <header><div><span>用户设置</span><h3>模型连接</h3></div><button onClick={onClose}>×</button></header>
      <div className="model-mode-tabs">
        <button className={form.mode === 'default' ? 'active' : ''} onClick={() => setForm(value => ({ ...value, mode: 'default' }))}>平台默认</button>
        <button className={form.mode === 'custom' ? 'active' : ''} onClick={() => setForm(value => ({ ...value, mode: 'custom' }))}>我的 API</button>
      </div>
      {form.mode === 'default' ? <div className="model-default-card">
        <b>使用平台托管连接</b><span>模型：{form.model || '由服务端环境配置'}</span><small>平台 API key 不会发送到浏览器。</small>
      </div> : <div className="model-fields">
        <label>接口地址<input value={form.base_url || ''} onChange={event => setForm(value => ({ ...value, base_url: event.target.value }))} placeholder="https://api.example.com" /></label>
        <label>模型名<input value={form.model || ''} onChange={event => setForm(value => ({ ...value, model: event.target.value }))} placeholder="model-name" /></label>
        <label>API key<input type="password" autoComplete="new-password" value={form.api_key || ''} onChange={event => setForm(value => ({ ...value, api_key: event.target.value }))} placeholder={form.api_key_hint ? '已保存 ' + form.api_key_hint + '；留空保持不变' : '首次配置必须填写'} /></label>
        <small>密钥加密保存，不写入对话、编排链或日志。公共地址建议使用 HTTPS。</small>
      </div>}
      {message && <div className={'model-dialog-message' + (message.startsWith('已') ? ' ok' : '')}>{message}</div>}
      <footer><button className="secondary" onClick={onClose}>关闭</button><button className="primary" disabled={busy} onClick={save}>{busy ? '处理中…' : '保存连接'}</button></footer>
    </section>
  </div>;
}

// ── Sidebar ──
function Sidebar({ conversations, activeId, onSelect, onNew, onLogout, onSettings, user, profile, tasks }) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h3>对话</h3>
      </div>
      <button className="new-conv-btn" onClick={onNew}>
        <span>＋</span> 新对话
      </button>
      <div className="conv-list">
        {conversations.length === 0 && (
          <div className="conv-empty">暂无对话</div>
        )}
        {conversations.map((c, idx) => {
          const title = c.title || ('会话 ' + (idx + 1));
          const active = c.conv_id === activeId;
          return (
            <div
              key={c.conv_id}
              className={'conv-item' + (active ? ' conv-item-active' : '')}
              onClick={() => onSelect(c.conv_id)}
            >
              <span className="conv-title">
                {title}
                {c.is_processing && <span className="conv-running">● 运行中</span>}
              </span>
              <span className="conv-meta">
                {relativeTime(c.created_at)} · {c.message_count || 0} 条消息
              </span>
            </div>
          );
        })}
      </div>
      <TaskPanel tasks={tasks} activeConvId={activeId} />
      <div className="sidebar-footer">
        <div className="user-identity" title={profile?.display_name || user}>
          {profile?.avatar_url
            ? <img className="user-avatar" src={profile.avatar_url} alt="" />
            : <span className="user-avatar user-avatar-fallback">{String(user || '?').slice(0, 1).toUpperCase()}</span>}
          <span className="user-badge">{profile?.display_name || user}</span>
        </div>
        <button className="btn-settings" onClick={onSettings}>模型连接</button>
        <button className="btn-logout" onClick={onLogout}>退出</button>
      </div>
    </div>
  );
}

// ── Main App ──
export default function App() {
  const [user, setUser] = useState(null);
  const [profile, setProfile] = useState(null);
  const [convId, setConvId] = useState('');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [processing, setProcessing] = useState(false);
  const [interruptRequested, setInterruptRequested] = useState(false);
  const [queuedRedirect, setQueuedRedirect] = useState('');
  const [partial, setPartial] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [workflow, setWorkflow] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const chatRef = useRef(null);
  const workflowSnapshotRef = useRef({});
  const workflowCacheRef = useRef({});
  const eventAnswerUidRef = useRef('');
  const acceptWorkflow = useCallback((next, expectedConvId) => {
    if (!next || next.scope?.conv_id !== expectedConvId) return;
    const revision = Number(next.snapshot_at || 0);
    const key = JSON.stringify([next.scope.username, expectedConvId]);
    const previous = Number(workflowSnapshotRef.current[key] || 0);
    if (revision < previous) return;
    workflowSnapshotRef.current[key] = revision;
    const merged = retainGraph(workflowCacheRef.current[key], next);
    workflowCacheRef.current[key] = merged;
    setWorkflow(merged);
  }, []);

  // restore session
  useEffect(() => {
    const saved = sessionStorage.getItem(USER_KEY);
    if (saved && getToken()) setUser(saved);
    else clearToken();
  }, []);

  const loadConversations = useCallback(async () => {
    if (!getToken()) return;
    try {
      const res = await fetch(API + '/conversations', { headers: authHeaders() });
      if (res.ok) {
        const d = await readJsonResponse(res);
        const available = d.conversations || [];
        setConversations(available);
        // A hard refresh reconstructs React state from scratch.  Restore the
        // backend's durable current conversation so the chat/chain never looks
        // empty until the user clicks the sidebar again.
        setConvId(previous => previous || d.current_conv_id || available[0]?.conv_id || '');
        // Auto-create a conversation if none exists
        if (available.length === 0) {
          const cr = await fetch(API + '/conversations', { method: 'POST', headers: authHeaders() });
          if (cr.ok) {
            const cd = await readJsonResponse(cr);
            setConvId(cd.conv_id);
          }
        }
      }
    } catch (e) {}
  }, []);

  const loadTasks = useCallback(async () => {
    if (!getToken()) return;
    try {
      const res = await fetch(API + '/jobs', { headers: authHeaders() });
      if (res.ok) {
        const d = await readJsonResponse(res);
        setTasks(d.jobs || []);
      }
    } catch (e) {}
  }, []);

  useEffect(() => {
    if (user) { loadConversations(); loadTasks(); }
  }, [user, loadConversations, loadTasks]);

  useEffect(() => {
    if (!user || !getToken()) { setProfile(null); return; }
    let active = true;
    fetch(API + '/me', { headers: authHeaders() })
      .then(response => response.ok ? readJsonResponse(response) : null)
      .then(data => { if (active && data) setProfile(data); })
      .catch(() => {});
    return () => { active = false; };
  }, [user]);

  useEffect(() => {
    if (!user) return;
    const t1 = setInterval(loadConversations, 5000);
    const t2 = setInterval(loadTasks, 5000);
    return () => { clearInterval(t1); clearInterval(t2); };
  }, [user, loadConversations, loadTasks]);

  // auto-scroll
  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, partial]);

  // load messages when conversation switches
  useEffect(() => {
    if (!user || !convId) return;
    let active = true;
    fetch(API + '/conversations/' + convId, { headers: authHeaders() })
      .then(r => r.ok ? readJsonResponse(r) : null)
      .then(d => {
        if (!active || !d) return;
        setMessages((d.messages || []).map(m => ({
          role: m.role, content: m.content, agent: m.agent,
          execution_logs: m.execution_logs, timestamp: m.timestamp,
          error: m.error,
        })));
        // 切换/打开一个**正在处理**的会话时，恢复实时轮询。
        // 否则该会话看起来"任务和响应消失"——后端 agent 其实还在跑，
        // 但前端 processing=false 导致 latest_answer 永不轮询、
        // 新响应/实时进度永不出现。
        setProcessing(!!d.is_processing);
        if (!d.is_processing) setPartial(null);
        setInterruptRequested(false);
        setQueuedRedirect('');
        acceptWorkflow(d.workflow, convId);
      }).catch(() => {});
    return () => { active = false; };
  }, [user, convId, acceptWorkflow]);

  // Authenticated server-sent workflow events. Native EventSource cannot send
  // our Bearer header, so use fetch streaming and parse SSE frames. A slow
  // polling path remains below only as a reconnect/network fallback.
  useEffect(() => {
    if (!user || !convId || !getToken()) return;
    const controller = new AbortController();
    let active = true;
    const applyEvent = async data => {
      if (!active || data.conv_id !== convId) return;
      acceptWorkflow(data.workflow, convId);
      setInterruptRequested(!!data.interrupt_requested);
      setQueuedRedirect(data.queued_redirect ? '已排队' : '');
      if (!data.done) {
        setProcessing(true);
        setPartial(previous => ({
          current_agent: data.agent_name || previous?.current_agent || 'Agent',
          tool_history: previous?.tool_history || [],
          status: 'executing',
          final_response: '',
          reasoning: data.reply_preview || previous?.reasoning || '',
        }));
        return;
      }
      setProcessing(false);
      setPartial(null);
      const answerKey = convId + ':' + (data.uid || 'idle');
      if (eventAnswerUidRef.current === answerKey) return;
      eventAnswerUidRef.current = answerKey;
      try {
        const response = await fetch(API + '/conversations/' + convId, { headers: authHeaders() });
        if (!response.ok || !active) return;
        const detail = await readJsonResponse(response);
        if (!active) return;
        setMessages((detail.messages || []).map(message => ({
          role: message.role, content: message.content, agent: message.agent,
          execution_logs: message.execution_logs, timestamp: message.timestamp,
          error: message.error,
        })));
      } catch (_) {}
    };
    const connect = async () => {
      try {
        const response = await fetch(API + '/conversations/' + convId + '/events', {
          headers: authHeaders(), signal: controller.signal,
        });
        if (!response.ok || !response.body) throw new Error('event stream unavailable');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (active) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let boundary;
          while ((boundary = buffer.indexOf('\n\n')) >= 0) {
            const frame = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            const payload = frame.split('\n').filter(line => line.startsWith('data: '))
              .map(line => line.slice(6)).join('\n');
            if (payload) {
              try { await applyEvent(JSON.parse(payload)); } catch (_) {}
            }
          }
        }
      } catch (_) {
        // The low-frequency authoritative polling below covers disconnects.
      }
    };
    connect();
    return () => { active = false; controller.abort(); };
  }, [user, convId, acceptWorkflow]);

  // Main model can be idle while workers run. Keep the selected conversation
  // observable and wake the progress poll when a lifecycle event starts a turn.
  useEffect(() => {
    if (!user || !convId || processing) return;
    let active = true, inFlight = false, seenUid = null, controller = null;
    const pollIdle = async () => {
      if (!active || inFlight) return;
      inFlight = true;
      controller = new AbortController();
      const timeout = setTimeout(() => controller?.abort(), 8000);
      try {
        const res = await fetch(API + '/latest_answer?conv_id=' + convId,
          { headers: authHeaders(), signal: controller.signal });
        if (!res.ok || !active) return;
        const d = await readJsonResponse(res);
        if (!active) return;
        acceptWorkflow(d.workflow, convId);
        setInterruptRequested(!!d.interrupt_requested);
        setQueuedRedirect(d.queued_redirect || '');
        if (!d.done) {
          setProcessing(true);
          return;
        }
        if (d.uid && d.uid !== seenUid) {
          const mr = await fetch(API + '/conversations/' + convId,
            { headers: authHeaders(), signal: controller.signal });
          if (!mr.ok || !active) return;
          const md = await readJsonResponse(mr);
          if (!active) return;
          setMessages((md.messages || []).map(m => ({
            role: m.role, content: m.content, agent: m.agent,
            execution_logs: m.execution_logs, timestamp: m.timestamp, error: m.error,
          })));
          seenUid = d.uid;
          if (md.is_processing) setProcessing(true);
        }
      } catch (_) {
        // Transient network loss must not erase messages/goal or start tasks.
      } finally {
        clearTimeout(timeout);
        inFlight = false;
      }
    };
    pollIdle();
    const interval = setInterval(pollIdle, 30000);
    return () => { active = false; clearInterval(interval); controller?.abort(); };
  }, [user, convId, processing, acceptWorkflow]);

  // poll progress while processing
  useEffect(() => {
    if (!processing || !convId) return;
    let active = true;
    let reloadDone = false;  // 只执行一次完成后的消息回填，避免重复 fetch
    const poll = async () => {
      try {
        const res = await fetch(API + '/latest_answer?conv_id=' + convId, { headers: authHeaders() });
        if (!res.ok || !active) return;
        const d = await readJsonResponse(res);
        if (!active) return;
        // Show partial progress
        setPartial({
          current_agent: d.agent_name || 'Agent',
          tool_history: d.partial_results || [],
          status: d.done ? 'completed' : 'executing',
          final_response: d.done ? d.answer : '',
          reasoning: d.reasoning || '',
        });
        // 中断后解锁输入框：interrupt_requested=True 时让用户能立即发重定向
        // 消息（Esc+typing），后端会排队到当前轮停止后执行。
        setInterruptRequested(!!d.interrupt_requested);
        setQueuedRedirect(d.queued_redirect || '');
        acceptWorkflow(d.workflow, convId);
        if (d.done) {
          if (!active || reloadDone) return;
          reloadDone = true;
          // ⚠️ 关键顺序：必须先回填最终消息，再 setProcessing(false)。
          // processing 是本 effect 的依赖，setProcessing(false) 会触发 effect
          // cleanup → active=false；若在它之前发起的 await fetch 尚未返回，
          // continuation 会因 `if (!active) return` 跳过 reload——最终答复
          // 既被 setPartial(null) 清掉、又没进 messages，"响应消失"。
          // 先 reload：此时 active 仍为 true（processing 还没改）。
          const mr = await fetch(API + '/conversations/' + convId, { headers: authHeaders() });
          if (!active) return;
          if (mr.ok) {
            const md = await readJsonResponse(mr);
            setMessages((md.messages || []).map(m => ({
              role: m.role, content: m.content, agent: m.agent,
              execution_logs: m.execution_logs, timestamp: m.timestamp,
              error: m.error,
            })));
          } else {
            // reload 失败兜底：把 d.answer 直接补进 messages，保证答复不消失
            setMessages(p => [...p, { role: 'assistant', content: d.answer || '', timestamp: Date.now() / 1000 }]);
          }
          setProcessing(false);
          setInterruptRequested(false);
          setQueuedRedirect('');
          setPartial(null);
          loadConversations();
        }
      } catch (e) {}
    };
    const id = setInterval(poll, 15000);
    return () => { active = false; clearInterval(id); };
  }, [processing, convId, loadConversations, acceptWorkflow]);

  const doNew = async () => {
    setProcessing(false);
    setInterruptRequested(false);
    setQueuedRedirect('');
    setPartial(null);
    setWorkflow(null);
    try {
      const r = await fetch(API + '/conversations', { method: 'POST', headers: authHeaders() });
      if (r.ok) {
        const d = await readJsonResponse(r);
        setConvId(d.conv_id);
        setMessages([]);
      }
    } catch (e) {}
  };

  const doSwitch = (id) => {
    if (id === convId) return;
    setConvId(id);
    setMessages([]);
    setPartial(null);
    setWorkflow(null);
    setProcessing(false);
    setInterruptRequested(false);
    setQueuedRedirect('');
    fetch(API + '/conversations/' + id + '/switch', { method: 'POST', headers: authHeaders() }).catch(() => {});
  };

  const doSend = async () => {
    const text = input.trim();
    // 中断(freeze)后允许立即发消息作为重定向：processing 仍为 true 时不阻塞，
    // 后端会把这条消息排队到当前轮停止后执行（不再 429）。
    if (!text || (processing && !interruptRequested) || !convId) return;
    setInput('');
    setMessages(p => [...p, { role: 'user', content: text, timestamp: Date.now() / 1000 }]);
    setProcessing(true);
    setInterruptRequested(false);  // 这条消息就是续写指令
    setQueuedRedirect('');
    setPartial(null);
    // Keep this conversation's lifecycle roles while the next turn starts.
    // Switching conversations still explicitly clears the previous snapshot.

    try {
      const r = await fetch(API + '/query', {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ query: text, conv_id: convId }),
      });
      const data = await readJsonResponse(r);
      if (!r.ok) {
        setMessages(p => [...p, { role: 'assistant', content: data.detail || 'Error', error: true, timestamp: Date.now() / 1000 }]);
        setProcessing(false);
        setPartial(null);
      }
      // Success: /api/query returns immediately (done=False).
      // Processing continues in background; polling effect watches latest_answer
      // until done=true, then reloads messages and clears processing.
    } catch (e) {
      setMessages(p => [...p, { role: 'assistant', content: e.message || 'Network error', error: true, timestamp: Date.now() / 1000 }]);
      setProcessing(false);
      setPartial(null);
    }
  };

  const doInterrupt = async (redirectMsg) => {
    // Claude-Code-style Esc: FREEZE the agent's current thinking/delegation.
    // - Previously submitted SLURM jobs are NOT cancelled (they keep running).
    // - The agent stops further thinking & tool calls and returns an honest
    //   partial report. The user's next message becomes the new delegation.
    // - Optional redirectMsg (Esc + typing): attached as the new requirement.
    if (!convId) return;
    setInterruptRequested(true);  // 立即解锁输入框，用户可直接打重定向消息
    try {
      const body = redirectMsg ? JSON.stringify({ message: redirectMsg }) : null;
      const r = await fetch(API + '/conversations/' + convId + '/interrupt', {
        method: 'POST', headers: authHeaders(),
        body,
      });
      const d = await readJsonResponse(r);
      let msg = '⛔ 已请求中断：agent 停止本轮思考与进一步动作。';
      msg += (d.note && d.note.includes('SLURM')) ? '\n已提交的 SLURM 作业不受影响，继续在计算节点运行。' : '';
      if (redirectMsg) msg += '\n已附带需求：' + redirectMsg + '\n⏳ agent 正在用该需求继续思考…';
      else msg += '\n请发送新消息继续，agent 会从上次停下的地方接着思考。';
      setMessages(p => [...p, { role: 'assistant', content: msg, error: false, timestamp: Date.now() / 1000 }]);
      // The polling effect keeps running; when the agent aborts it will set
      // done=true and reload the partial report.
    } catch (e) {
      setInterruptRequested(false);
      setMessages(p => [...p, { role: 'assistant', content: '中断请求失败: ' + (e.message || e), error: true, timestamp: Date.now() / 1000 }]);
    }
  };

  const doLogout = async () => {
    try { await fetch(API + '/logout', { method: 'POST', headers: authHeaders() }); } catch (e) {}
    clearToken();
    sessionStorage.removeItem(USER_KEY);
    setUser(null);
    workflowCacheRef.current = {};
    workflowSnapshotRef.current = {};
    setWorkflow(null);
    setProfile(null);
    setMessages([]);
    setConversations([]);
    setConvId('');
    setProcessing(false);
    setInterruptRequested(false);
    setQueuedRedirect('');
    setPartial(null);
  };

  if (!user) return <AuthForm onLogin={u => { setUser(u); sessionStorage.setItem(USER_KEY, u); }} />;

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeId={convId}
        onSelect={doSwitch}
        onNew={doNew}
        onLogout={doLogout}
        onSettings={() => setSettingsOpen(true)}
        user={user}
        profile={profile}
        tasks={tasks}
      />

      <div className="main-area">
        <div className="chat-area">
          <div className="chat-container" ref={chatRef}>
            {messages.length === 0 && !processing && (
              <div className="welcome">
                <h2>BiMemAgent</h2>
                <p>MOF膜吸附研究 · AI科研助手</p>
                <div className="capabilities">
                  {['MOF气体吸附等温线', 'GCMC分子模拟', 'cDFT密度泛函', 'VASP DFT优化', 'PORE结构分析', 'IAST膜分离预测'].map(c => (
                    <div className="cap-item" key={c} onClick={() => setInput('请帮我做' + c + '分析')}>{c}</div>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => {
              const cls = 'message ' + (m.role === 'user' ? 'user-message' : 'agent-message') + (m.error ? ' error-message' : '');
              return (
                <div key={i} className={cls}>
                  {m.role === 'user' ? (
                    <div className="message-content">{m.content}</div>
                  ) : (
                    <React.Fragment>
                      <div className="message-header">
                        <span className="agent-name">🧬 {m.agent || 'Agent'}</span>
                        <span className="agent-status">{formatLogTime(m.timestamp)}</span>
                      </div>
                      <div className="message-content">
                        <SmartResponseRenderer text={m.content} />
                      </div>
                      <ExecutionLogs logs={m.execution_logs} />
                    </React.Fragment>
                  )}
                </div>
              );
            })}

            {processing && (
              <div className="message agent-message processing">
                <div className="message-header">
                  <span className="agent-name">⏳ {partial && partial.current_agent ? partial.current_agent : '等待'}</span>
                  <span className="agent-status">{partial ? '执行中 · ' + (partial.tool_history || []).length + ' 步' : '任务处理中'}</span>
                </div>
                <div className="message-content">
                  {partial && partial.status === 'completed' ? (
                    <span>✅ 任务完成</span>
                  ) : partial && partial.final_response ? (
                    <span style={{ color: 'var(--text)' }}>{partial.final_response}</span>
                  ) : (
                    <React.Fragment>
                      <div className="processing-indicator">
                        详细执行状态已同步到右侧运行面板 <span className="processing-dots"><span>.</span><span>.</span><span>.</span></span>
                      </div>
                    </React.Fragment>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="input-container">
            <textarea rows={1} placeholder={queuedRedirect ? '（你的消息已排队，当前轮结束后自动执行）' : (interruptRequested ? '已中断——直接输入新消息，当前轮停止后立即执行...' : '描述你的研究需求...')} value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doSend(); } }}
              disabled={processing && !interruptRequested} />
            {processing ? (
              <button className="interrupt-btn" onClick={() => {
                // 一键冻结（无弹窗，不打断操作流）：agent 立即停止本轮思考与
                // 进一步动作，已提交的 SLURM 作业继续运行。之后你在输入框
                // 直接发新消息，agent 就会从刚才停下的地方顺着新需求继续思考。
                if (window.confirm('⛔ 中断？\n\nagent 立即停止本轮思考（已提交的 SLURM 作业继续运行）。\n冻结后你在输入框发新消息，agent 会接着思考。')) {
                  doInterrupt('');
                }
              }}
              title="中断 agent 当前思考：立即停止本轮思考，已提交作业不受影响。冻结后在输入框发新消息继续">
                ⛔ 中断
              </button>
            ) : null}
            <button className="send-btn" onClick={doSend} disabled={(processing && !interruptRequested) || !input.trim() || !convId}>
              {queuedRedirect ? '⏳ 已排队' : (interruptRequested ? '继续' : (processing ? '处理中...' : '发送'))}
            </button>
          </div>
        </div>
      </div>
      <OperationsDrawer
        workflow={workflow?.scope?.conv_id === convId && workflow?.scope?.username === user
          ? workflow : workflowCacheRef.current[JSON.stringify([user, convId])] || null}
        partial={partial}
        processing={processing}
      />
      {settingsOpen && <ModelConnectionDialog onClose={() => setSettingsOpen(false)} />}
    </div>
  );
}
