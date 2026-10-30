import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import ResultDeliverables from './ResultDeliverables';

let root, container;
const authHeaders = () => ({ Authorization: 'Bearer fixture-token' });
const item = { artifact_id: 'a'.repeat(64), name: 'result.png', kind: 'figure', visual_review: 'pending' };
const json = body => ({ ok: true, json: async () => body });
const render = (extra = {}) => act(async () => {
  root.render(<ResultDeliverables owner="owner" convId="conv" revision="v1" authHeaders={authHeaders} {...extra} />);
});
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement('div'); document.body.appendChild(container);
  root = createRoot(container);
  global.fetch = jest.fn();
  URL.createObjectURL = jest.fn(() => 'blob:verified-result');
  URL.revokeObjectURL = jest.fn();
});
afterEach(() => { act(() => root.unmount()); container.remove(); jest.restoreAllMocks(); });

test('artifact list and image preview use authenticated requests with no token in URL', async () => {
  fetch.mockResolvedValueOnce(json({ artifacts: [item] }))
    .mockResolvedValueOnce({ ok: true, blob: async () => new Blob(['png'], { type: 'image/png' }) });
  await render();
  expect(container.textContent).toContain('result.png');
  await act(async () => container.querySelector('button').click());
  expect(container.querySelector('img').getAttribute('src')).toBe('blob:verified-result');
  expect(container.textContent).toContain('尚未完成视觉审核');
  fetch.mock.calls.forEach(([url, options]) => {
    expect(url).not.toContain('fixture-token');
    expect(options.headers.Authorization).toBe('Bearer fixture-token');
  });
});

test('switching accounts immediately drops preview and revokes its blob URL', async () => {
  fetch.mockResolvedValueOnce(json({ artifacts: [item] }))
    .mockResolvedValueOnce({ ok: true, blob: async () => new Blob(['png']) })
    .mockResolvedValueOnce(json({ artifacts: [] }));
  await render(); await act(async () => container.querySelector('button').click());
  expect(container.querySelector('img')).not.toBeNull();
  await render({ owner: 'other' });
  expect(container.querySelector('img')).toBeNull();
  expect(container.textContent).not.toContain('result.png');
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:verified-result');
});

test('late previous-session listing cannot replace the current list', async () => {
  let finish;
  fetch.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }))
    .mockResolvedValueOnce(json({ artifacts: [] }));
  await render(); await render({ convId: 'other-conv' });
  await act(async () => finish(json({ artifacts: [item] })));
  expect(container.textContent).not.toContain('result.png');
  expect(fetch.mock.calls[0][1].signal.aborted).toBe(true);
});

test('late previous-session image response is discarded before making a blob URL', async () => {
  let finish;
  fetch.mockResolvedValueOnce(json({ artifacts: [item] }))
    .mockReturnValueOnce(new Promise(resolve => { finish = resolve; }))
    .mockResolvedValueOnce(json({ artifacts: [] }));
  await render(); await act(async () => container.querySelector('button').click());
  await render({ convId: 'other-conv' });
  await act(async () => finish({ ok: true, blob: async () => new Blob(['png']) }));
  expect(URL.createObjectURL).not.toHaveBeenCalled();
  expect(container.querySelector('img')).toBeNull();
});

test('report preview renders text without executing HTML or Markdown images', async () => {
  const report = { ...item, name: 'report.md', kind: 'report' };
  fetch.mockResolvedValueOnce(json({ artifacts: [report] }))
    .mockResolvedValueOnce({ ok: true, blob: async () => ({ text: async () => '<script>unsafe</script> ![x](https://example.com/x.png)' }) });
  await render(); await act(async () => container.querySelector('button').click());
  expect(container.querySelector('pre').textContent).toContain('<script>unsafe</script>');
  expect(container.querySelector('script')).toBeNull();
  expect(container.querySelector('img')).toBeNull();
});

test('native verification conflict cannot become a preview', async () => {
  fetch.mockResolvedValueOnce(json({ artifacts: [item] })).mockResolvedValueOnce({ ok: false, status: 409 });
  await render(); await act(async () => container.querySelector('button').click());
  expect(container.querySelector('[role="alert"]').textContent).toContain('校验未通过');
  expect(container.querySelector('img')).toBeNull();
});

test('temporary API loss automatically reloads the inventory without a browser refresh', async () => {
  jest.useFakeTimers();
  try {
    fetch.mockResolvedValueOnce({ ok: false, status: 503 })
      .mockResolvedValueOnce(json({ artifacts: [item] }));
    await render();
    expect(container.querySelector('[role="alert"]')).not.toBeNull();
    await act(async () => { jest.advanceTimersByTime(1000); });
    expect(container.textContent).toContain('result.png');
    expect(container.querySelector('[role="alert"]')).toBeNull();
  } finally { jest.useRealTimers(); }
});

test('an explicit result refresh recovers a stale inventory immediately', async () => {
  fetch.mockResolvedValueOnce({ ok: false, status: 409 })
    .mockResolvedValueOnce(json({ artifacts: [item] }));
  await render();
  const refresh = [...container.querySelectorAll('button')].find(button => button.textContent === '刷新结果');
  await act(async () => refresh.click());
  expect(container.textContent).toContain('result.png');
});

test('expired credentials do not create an endless metadata request loop', async () => {
  jest.useFakeTimers();
  try {
    fetch.mockResolvedValue({ ok: false, status: 401 });
    await render();
    await act(async () => { jest.advanceTimersByTime(60000); });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain('登录状态已失效');
  } finally { jest.useRealTimers(); }
});
