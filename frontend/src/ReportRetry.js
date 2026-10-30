import React, { useEffect, useRef, useState } from 'react';

const messages = {
  report_worker_claimed: '报告已开始处理。请在任务进度中查看结果。',
  worker_submitted: '报告已开始处理。请在任务进度中查看结果。',
  claimed: '请求已接收，正在确认启动状态。',
  claimed_requires_reconciliation: '请求已接收，启动状态待确认；请查询状态，避免重复提交。',
  submission_unknown: '启动状态待确认；请查询状态，避免重复提交。',
  not_started_control_or_source_changed: '报告未启动：任务或资料状态已变化，请查看当前任务状态。',
  not_started: '报告未启动：任务或资料状态已变化，请查看当前任务状态。',
};
const keyFor = (owner, convId) => 'bimem_report_retry_v1:' + JSON.stringify([owner, convId]);
function requestId() {
  if (!window.crypto?.getRandomValues) throw new Error('secure randomness unavailable');
  const bytes = new Uint8Array(16); window.crypto.getRandomValues(bytes);
  return 'report_' + Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
}
function loadPending(key) {
  try { const value = JSON.parse(sessionStorage.getItem(key));
    return value && /^[A-Za-z0-9_-]{16,128}$/.test(value.request_id) ? value : null;
  } catch (_) { return null; }
}

export default function ReportRetry({ owner, convId, authHeaders, processing, paused, revision }) {
  const scope = owner && convId ? keyFor(owner, convId) : '';
  const [options, setOptions] = useState([]);
  const [selected, setSelected] = useState(null);
  const [pending, setPending] = useState(() => scope ? loadPending(scope) : null);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const current = useRef(scope); current.current = scope;
  const inFlight = useRef(false);
  const optionsRevision = useRef(0);
  useEffect(() => { setOptions([]); setSelected(null); setStatus(''); setPending(scope ? loadPending(scope) : null); setBusy(false); inFlight.current = false; }, [scope]);
  useEffect(() => {
    if (!owner || !convId) return undefined;
    const abort = new AbortController(); const observed = scope; const sequence = ++optionsRevision.current;
    setSelected(null);
    fetch('/api/conversations/' + encodeURIComponent(convId) + '/workflow/report-retry-options',
      { headers: authHeaders(), signal: abort.signal })
      .then(r => { if (!r.ok) throw new Error('unavailable'); return r.json(); })
      .then(data => { if (current.current === observed && optionsRevision.current === sequence && Array.isArray(data.options)) setOptions(data.options); })
      .catch(() => {});
    return () => abort.abort();
  }, [owner, convId, scope, authHeaders, revision]);
  if (!owner || !convId || (!options.length && !pending)) return null;
  const remember = value => {
    try { sessionStorage.setItem(scope, JSON.stringify(value));
      if (!loadPending(scope)) return false; setPending(value); return true;
    } catch (_) { return false; }
  };
  const markUsed = stepId => setOptions(items => items.map(item => item.step_id === stepId ? { ...item, retry_state: 'used' } : item));
  const clear = () => { setPending(null); try { sessionStorage.removeItem(scope); } catch (_) {} };
  async function check() {
    if (!pending || inFlight.current) return;
    const request = pending; const observed = scope; inFlight.current = true; setBusy(true);
    try {
      const r = await fetch('/api/conversations/' + encodeURIComponent(convId) + '/workflow/report-retry-receipts/' + encodeURIComponent(request.request_id), { headers: authHeaders() });
      if (current.current !== observed) return;
      if (r.status === 404) { setStatus('暂未查到请求回执。请稍后再次查询，避免重复提交。'); return; }
      if (!r.ok) { setStatus(r.status === 401 || r.status === 403 ? '登录状态无效或无权查看该任务。' : '暂时无法查询状态，请稍后重试。'); return; }
      const data = await r.json(); if (current.current !== observed) return;
      setStatus(messages[data.status] || '请求已记录，请在任务进度中查看。');
      if (['worker_submitted', 'not_started'].includes(data.status)) { markUsed(request.step_id); clear(); }
    } catch (_) { if (current.current === observed) setStatus('连接中断，请稍后查询状态，避免重复提交。'); }
    finally { if (current.current === observed) { setBusy(false); inFlight.current = false; } }
  }
  async function submit() {
    if (!selected || pending || processing || paused || inFlight.current) return;
    const observed = scope; let id;
    try { id = requestId(); } catch (_) { setStatus('当前浏览器无法安全生成请求，请换用受支持的浏览器。'); return; }
    const ticket = { request_id: id, step_id: selected.step_id };
    if (!remember(ticket)) { setStatus('无法保存请求状态，请检查浏览器存储后再试。'); return; }
    inFlight.current = true; setBusy(true); setStatus('正在提交请求…');
    try {
      const r = await fetch('/api/conversations/' + encodeURIComponent(convId) + '/workflow/nodes/' + encodeURIComponent(selected.step_id) + '/retry-report-once', {
        method: 'POST', headers: authHeaders(), body: JSON.stringify({ plan_version: selected.plan_version,
          expected_failure_call_id: selected.expected_failure_call_id, expected_failure_token: selected.expected_failure_token,
          request_id: id, confirm_same_report_only: true }) });
      if (current.current !== observed) return;
      if (!r.ok) {
        if ([400, 401, 403, 409, 422].includes(r.status)) { clear(); setStatus(r.status === 401 || r.status === 403 ? '登录状态无效或无权操作该任务。' : '当前报告状态已变化或不满足重试条件，请刷新后查看。'); }
        else setStatus('请求状态尚未确认，请查询结果，避免重复提交。');
        return;
      }
      const data = await r.json(); if (current.current !== observed) return;
      setStatus(messages[data.status] || '请求已接收，请在任务进度中查看。'); setSelected(null);
      if (['report_worker_claimed', 'not_started_control_or_source_changed'].includes(data.status)) { markUsed(ticket.step_id); clear(); }
    } catch (_) { if (current.current === observed) setStatus('连接中断，请查询请求状态，避免重复提交。'); }
    finally { if (current.current === observed) { setBusy(false); inFlight.current = false; } }
  }
  return <section className="runtime-section report-retry" aria-label="报告重试">
    <div className="runtime-section-title"><span>报告重试</span></div>
    <p>仅重试所选报告一次；保留已有计算和结果。提交前会再次检查任务状态与来源。</p>
    {options.map(option => <div className="report-retry-row" key={option.step_id}>
      <span>{option.label || option.step_id}</span>
      <button type="button" disabled={busy || !!pending || processing || paused || option.retry_state !== 'check_required'}
        onClick={() => { setSelected(option); setStatus(''); }}>{option.retry_state === 'used' ? '本次重试已使用' : '重试此报告'}</button>
    </div>)}
    {selected && <div className="report-retry-confirm" role="group" aria-label="确认报告重试">
      <p>确认仅重试“{selected.label || selected.step_id}”？已有计算不会重新提交，也不会覆盖已有报告。</p>
      <button type="button" disabled={busy || processing || paused || !!pending} onClick={submit}>确认重试一次</button>
      <button type="button" disabled={busy} onClick={() => setSelected(null)}>取消</button>
    </div>}
    {pending && <button type="button" disabled={busy} onClick={check}>查询请求状态</button>}
    {status && <p role="status" aria-live="polite">{status}</p>}
  </section>;
}
