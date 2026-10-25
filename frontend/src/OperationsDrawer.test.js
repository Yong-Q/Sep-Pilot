import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { OperationsDrawer } from './App';

let container, root;
const workflow = (extra = {}) => ({
  scope: { username: 'fixture', conv_id: 'research' },
  graph_projection: { source: 'persisted_chain', version: 1, nodes: [
    { step_id: 'first', tool: 'run_cdft', description: 'First calculation', status: 'pending', depends_on: [] },
    { step_id: 'second', tool: 'run_cdft', description: 'Second calculation', status: 'pending', depends_on: ['first'] },
  ] }, ...extra,
});
const render = value => act(() => root.render(<OperationsDrawer workflow={value} processing={false} />));
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  localStorage.clear();
  window.matchMedia = jest.fn(() => ({ matches: false }));
  container = document.createElement('div'); document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); container.remove(); });

test('explicit recovery ownership never leaks to another node sharing its tool', () => {
  render(workflow({ recovery: { attempt: { step_id: 'second', tool: 'run_cdft', status: 'recovering' } } }));
  const recoveryRow = [...container.querySelectorAll('.runtime-inspector-grid > div')]
    .find(row => row.textContent.includes('恢复/重试记录'));
  expect(recoveryRow.querySelector('b').textContent).toBe('0');
});

test('pending root does not claim to wait on a nonexistent dependency', () => {
  render(workflow());
  expect(container.querySelector('.runtime-inspector-state').textContent).toContain('等待调度');
});

test('Space selects a graph node without scrolling the page', () => {
  render(workflow());
  const node = container.querySelectorAll('.stable-dag-node')[1];
  const key = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
  act(() => node.dispatchEvent(key));
  expect(key.defaultPrevented).toBe(true);
  expect(container.querySelector('.runtime-inspector-head b').textContent).toBe('Second calculation');
  expect(node.getAttribute('aria-pressed')).toBe('true');
});

test('drawer communicates expanded state and Escape returns keyboard focus', () => {
  render(workflow());
  const toggle = container.querySelector('.runtime-drawer-toggle');
  expect(toggle.getAttribute('aria-expanded')).toBe('true');
  const node = container.querySelector('.stable-dag-node'); node.focus();
  act(() => node.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  expect(toggle.getAttribute('aria-expanded')).toBe('false');
  expect(document.activeElement).toBe(toggle);
});

test('first visit on a small screen leaves the conversation accessible', () => {
  window.matchMedia = jest.fn(() => ({ matches: true }));
  render(workflow());
  expect(container.querySelector('.runtime-drawer-closed')).not.toBeNull();
});

test('node selection and graph persist when live status updates arrive', () => {
  const data = workflow(); render(data);
  act(() => container.querySelectorAll('.stable-dag-node')[1].dispatchEvent(new MouseEvent('click', { bubbles: true })));
  const updated = workflow(); updated.graph_projection.nodes[1].status = 'waiting_jobs';
  render(updated);
  expect(container.querySelectorAll('.stable-dag-node')).toHaveLength(2);
  expect(container.querySelector('.runtime-inspector-head b').textContent).toBe('Second calculation');
  expect(container.querySelector('.runtime-inspector-state').textContent).toContain('作业执行中');
});

test('ambiguous legacy tool-only recovery stays unattributed', () => {
  render(workflow({ recovery: { attempt: { tool: 'run_cdft', status: 'recovering' } } }));
  const recoveryRow = [...container.querySelectorAll('.runtime-inspector-grid > div')]
    .find(row => row.textContent.includes('恢复/重试记录'));
  expect(recoveryRow.querySelector('b').textContent).toBe('0');
});

test('drawer respects an explicit saved preference on small screens', () => {
  window.matchMedia = jest.fn(() => ({ matches: true }));
  localStorage.setItem('runtime_drawer_open', '1');
  render(workflow());
  expect(container.querySelector('.runtime-drawer-open')).not.toBeNull();
});

test('graph zoom makes long chains readable and reset restores the fitted view', () => {
  render(workflow());
  const zoom = container.querySelector('[aria-label="放大编排图"]');
  expect(zoom).not.toBeNull();
  act(() => zoom.click());
  const svg = container.querySelector('.stable-dag-svg');
  expect(parseFloat(svg.style.width)).toBeGreaterThan(0);
  expect(container.querySelectorAll('.stable-dag-node')).toHaveLength(2);
  act(() => container.querySelector('[aria-label="适应面板"]').click());
  expect(svg.style.width).toBe('');
});

const technicalFailure = (extra = {}) => workflow({
  current_activity: { status: '等待用户协商编排', waiting_for: '' },
  parallel_runtime: { status: 'needs_user', pause_kind: 'failure', plan_version: 1,
    nodes: { first: { status: 'validation_failed' } } },
  ...extra,
});

test('technical validation failure without a pending decision does not ask the user to negotiate', () => {
  render(technicalFailure());
  const status = container.querySelector('.runtime-current-activity em').textContent;
  expect(status).toContain('等待智能体诊断');
  expect(status).not.toContain('用户');
  expect(container.querySelector('.runtime-inspector-state').textContent).toContain('验收未通过');
});

test.each([
  { awaiting_plan_approval: true },
  { pending_workflow_patch: { base_version: 1, new_version: 2 } },
  { current_activity: { status: '等待用户确认', waiting_for: 'user_decision' } },
])('a real pending user decision remains explicit: %j', pending => {
  render(technicalFailure(pending));
  expect(container.querySelector('.runtime-current-activity em').textContent).toContain('用户');
});

test('an explicit user pause remains visible even with a technical failure', () => {
  const data = technicalFailure(); data.parallel_runtime.user_paused = true;
  render(data);
  expect(container.querySelector('.runtime-current-activity em').textContent).toContain('已按用户要求暂停');
});

test('a lifecycle wait is not misrepresented as a pending user decision', () => {
  render(technicalFailure({ current_activity: { status: '等待用户协商编排', waiting_for: 'lifecycle_wait' } }));
  expect(container.querySelector('.runtime-current-activity em').textContent).not.toContain('用户');
});
