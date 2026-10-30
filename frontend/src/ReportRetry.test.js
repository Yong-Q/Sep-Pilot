import React from 'react';
import { act } from 'react-dom/test-utils';
import { createRoot } from 'react-dom/client';
import ReportRetry from './ReportRetry';
global.IS_REACT_ACT_ENVIRONMENT = true;
let host, root;
const headers = () => ({ Authorization: 'Bearer test-owner', 'Content-Type': 'application/json' });
const option = { step_id:'report', label:'计算报告', plan_version:3, expected_failure_call_id:'call', expected_failure_token:'a'.repeat(32), retry_state:'check_required' };
const flush = () => act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
beforeEach(() => { if (!window.crypto) Object.defineProperty(window, 'crypto', { value: {}, configurable: true }); window.crypto.getRandomValues = bytes => { bytes.fill(7); return bytes; }; sessionStorage.clear(); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); jest.restoreAllMocks(); });
async function render(scope={}) { await act(async () => root.render(<ReportRetry owner="alice" convId="case" authHeaders={headers} {...scope} />)); await flush(); }
function buttons(text) { return [...host.querySelectorAll('button')].filter(b => b.textContent === text); }

test('requires explicit confirmation; sends only exact selected report once', async () => {
 const calls=[]; global.fetch=jest.fn(async (url,init) => { calls.push([url,init]); return {ok:true,json:async()=>url.endsWith('options')?{options:[option]}:{status:'report_worker_claimed'}}; });
 await render(); expect(calls).toHaveLength(1); act(()=>buttons('重试此报告')[0].click()); expect(calls).toHaveLength(1);
 expect(host.textContent).toContain('已有计算不会重新提交'); await act(async()=>buttons('确认重试一次')[0].click());
 expect(calls).toHaveLength(2); expect(calls[1][0]).toContain('/case/workflow/nodes/report/retry-report-once');
 const body=JSON.parse(calls[1][1].body); expect(body.confirm_same_report_only).toBe(true); expect(body.expected_failure_token).toBe('a'.repeat(32));
 expect(buttons('本次重试已使用')[0].disabled).toBe(true);
});

test('used and paused states cannot submit', async () => {
 global.fetch=jest.fn(async()=>({ok:true,json:async()=>({options:[{...option,retry_state:'used'}]})})); await render();
 expect(buttons('本次重试已使用')[0].disabled).toBe(true); expect(global.fetch).toHaveBeenCalledTimes(1);
});

test('lost response retains request and checks receipt without repeating POST', async () => {
 let calls=[]; global.fetch=jest.fn(async (url,init) => { calls.push([url,init]); if(url.endsWith('options')) return {ok:true,json:async()=>({options:[option]})}; if(init.method==='POST') throw Error('offline'); return {ok:true,json:async()=>({status:'worker_submitted'})}; });
 await render(); act(()=>buttons('重试此报告')[0].click()); await act(async()=>buttons('确认重试一次')[0].click());
 expect(host.textContent).toContain('连接中断'); expect(buttons('查询请求状态')).toHaveLength(1);
 await act(async()=>buttons('查询请求状态')[0].click()); expect(calls.filter(([,init])=>init.method==='POST')).toHaveLength(1);
 expect(buttons('本次重试已使用')[0].disabled).toBe(true);
});

test('403 fails closed without raw error disclosure', async () => {
 global.fetch=jest.fn(async (url) => url.endsWith('options')?{ok:true,json:async()=>({options:[option]})}:{ok:false,status:403});
 await render(); act(()=>buttons('重试此报告')[0].click()); await act(async()=>buttons('确认重试一次')[0].click());
 expect(host.textContent).toContain('无权操作'); expect(host.textContent).not.toContain('token');
});

test('pending request belongs only to its authenticated conversation', async () => {
 sessionStorage.setItem('bimem_report_retry_v1:'+JSON.stringify(['other','case']),JSON.stringify({request_id:'report_0123456789abcdef0123456789abcdef',step_id:'report'}));
 global.fetch=jest.fn(async()=>({ok:true,json:async()=>({options:[option]})})); await render();
 expect(buttons('查询请求状态')).toHaveLength(0); expect(buttons('重试此报告')[0].disabled).toBe(false);
});

test('uncertain result retains receipt lookup and blocks duplicate requests', async () => {
 let posts=0; global.fetch=jest.fn(async (url,init) => { if(url.endsWith('options')) return {ok:true,json:async()=>({options:[option]})}; if(init.method==='POST'){ posts++;throw Error('connection'); } return {ok:true,json:async()=>({status:'submission_unknown'})}; });
 await render();act(()=>buttons('重试此报告')[0].click());await act(async()=>buttons('确认重试一次')[0].click());await act(async()=>buttons('查询请求状态')[0].click());
 expect(posts).toBe(1);expect(buttons('查询请求状态')).toHaveLength(1);expect(buttons('重试此报告')[0].disabled).toBe(true);
});

test('same conversation refreshes changed report status and ignores stale response', async () => {
 let resolveOld;
 global.fetch=jest.fn((url) => {
  if(global.fetch.mock.calls.length===1) return new Promise(resolve=>{resolveOld=resolve;});
  return Promise.resolve({ok:true,json:async()=>({options:[{...option,retry_state:'used'}]})});
 });
 await act(async()=>root.render(<ReportRetry owner="alice" convId="case" authHeaders={headers} revision="old" />));
 await act(async()=>root.render(<ReportRetry owner="alice" convId="case" authHeaders={headers} revision="new" />)); await flush();
 expect(buttons('本次重试已使用')[0].disabled).toBe(true);
 await act(async()=>resolveOld({ok:true,json:async()=>({options:[option]})})); await flush();
 expect(buttons('本次重试已使用')[0].disabled).toBe(true);
});

test('changed workflow invalidates an open confirmation', async () => {
 global.fetch=jest.fn(async()=>({ok:true,json:async()=>({options:[option]})}));
 await act(async()=>root.render(<ReportRetry owner="alice" convId="case" authHeaders={headers} revision="old" />)); await flush();
 act(()=>buttons('重试此报告')[0].click()); expect(buttons('确认重试一次')).toHaveLength(1);
 await act(async()=>root.render(<ReportRetry owner="alice" convId="case" authHeaders={headers} revision="new" />)); await flush();
 expect(buttons('确认重试一次')).toHaveLength(0);
});
