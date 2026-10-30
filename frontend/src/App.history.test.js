import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { TextDecoder, TextEncoder } from 'util';
import App from './App';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const response = (data, status = 200) => ({
  ok: status >= 200 && status < 300, status,
  text: async () => JSON.stringify(data),
});
function eventStream() {
  const queued = [], waiting = [];
  return {
    read: () => queued.length ? Promise.resolve(queued.shift()) : new Promise(resolve => waiting.push(resolve)),
    push(data) {
      const value = { value: new TextEncoder().encode('data: ' + JSON.stringify(data) + '\n\n'), done: false };
      if (waiting.length) waiting.shift()(value); else queued.push(value);
    },
    end() { while (waiting.length) waiting.shift()({ done: true }); },
  };
}

let container, root, server;
const flush = () => act(async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); });
const userTexts = () => [...container.querySelectorAll('.user-message .message-content')].map(node => node.textContent);
const detail = (messages = [], extra = {}) => ({ conv_id: 'one', messages, is_processing: false, ...extra });
async function mount() {
  root = createRoot(container);
  await act(async () => root.render(<App />));
  await flush();
}
async function send(text) {
  const field = container.querySelector('.input-container textarea');
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(field, text);
    field.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await act(async () => container.querySelector('.send-btn').click());
  await flush();
}
async function event(data, convId = 'one') {
  await act(async () => server.streams[convId].push({ conv_id: convId, ...data }));
  await flush();
}
async function switchTo(title) {
  await act(async () => [...container.querySelectorAll('.conv-item')].find(node => node.textContent.includes(title)).click());
  await flush();
}
async function remount() {
  await act(async () => root.unmount()); root = null;
  await mount();
}

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  global.TextDecoder = TextDecoder;
  sessionStorage.clear(); localStorage.clear();
  sessionStorage.setItem('bimem_user', 'fixture');
  sessionStorage.setItem('bimem_token', 'offline-fixture');
  window.matchMedia = () => ({ matches: false });
  container = document.createElement('div'); document.body.appendChild(container);
  server = {
    streams: {}, requests: [], detailQueue: [], posts: [], queryQueue: [], answers: {}, answerQueue: [],
    histories: { one: detail(), two: { ...detail(), conv_id: 'two' } },
  };
  global.fetch = jest.fn(async (url, options = {}) => {
    const method = options.method || 'GET';
    server.requests.push({ url, method });
    if (url === '/api/conversations' && method === 'GET') return response({ current_conv_id: 'one', conversations: [
      { conv_id: 'one', title: 'First conversation' }, { conv_id: 'two', title: 'Second conversation' },
    ] });
    if (url === '/api/jobs') return response({ jobs: [] });
    if (url === '/api/me') return response({ display_name: 'Fixture' });
    if (url.startsWith('/api/latest_answer')) {
      if (server.answerQueue.length) return server.answerQueue.shift().promise;
      const id = new URL(url, 'http://fixture').searchParams.get('conv_id');
      return response(server.answers[id] || { done: true, uid: '', conv_id: id });
    }
    if (url.endsWith('/events')) {
      const id = url.split('/')[3], stream = eventStream(); server.streams[id] = stream;
      options.signal?.addEventListener('abort', () => stream.end());
      return { ok: true, body: { getReader: () => stream } };
    }
    if (url === '/api/query') {
      const body = JSON.parse(options.body); server.posts.push(body);
      return server.queryQueue.length ? server.queryQueue.shift().promise : response({ done: false, uid: String(99 + server.posts.length), conv_id: body.conv_id, client_turn_id: body.client_turn_id });
    }
    if (/\/conversations\/(one|two)$/.test(url)) {
      return server.detailQueue.length ? server.detailQueue.shift().promise : response(server.histories[url.split('/')[3]]);
    }
    if (url.endsWith('/switch') || url === '/api/logout') return response({ status: 'ok' });
    throw new Error('Unexpected offline route: ' + method + ' ' + url);
  });
});
afterEach(async () => {
  if (root) await act(async () => root.unmount());
  Object.values(server.streams).forEach(stream => stream.end());
  container.remove(); jest.restoreAllMocks();
  jest.useRealTimers();
});

test('a delayed idle history cannot erase a user after accepted POST and busy SSE', async () => {
  await mount();
  const oldHistory = deferred(), accepted = deferred();
  server.detailQueue.push(oldHistory); server.queryQueue.push(accepted);
  await event({ done: true, uid: '', revision: 0 });
  await send('Check this structure');
  expect(userTexts()).toEqual(['Check this structure']);
  oldHistory.resolve(response(detail()));
  await flush();
  accepted.resolve(response({ done: false, uid: '100', conv_id: 'one', client_turn_id: server.posts[0].client_turn_id }));
  await flush();
  await event({ done: false, uid: '100', revision: 1 });
  expect(userTexts()).toEqual(['Check this structure']);
  expect(container.querySelector('.processing')).not.toBeNull();
  expect(server.posts).toHaveLength(1);
});

test('a stale initial history response cannot end or erase a newer accepted turn', async () => {
  const initial = deferred(); server.detailQueue.push(initial);
  await mount();
  await send('First accepted question');
  initial.resolve(response(detail())); await flush();
  expect(userTexts()).toEqual(['First accepted question']);
  expect(container.querySelector('.processing')).not.toBeNull();
  expect(server.posts).toHaveLength(1);
});

test('identical questions in distinct turns reconcile by identity without merging', async () => {
  await mount(); await send('Repeat this question');
  const first = server.posts[0];
  server.histories.one = detail([
    { role: 'user', content: first.query, client_turn_id: first.client_turn_id, turn_uid: '100' },
    { role: 'assistant', content: 'First answer' },
  ], { current_uid: '100' });
  await event({ done: true, uid: '100', revision: 1 });
  await send('Repeat this question');
  const second = server.posts[1];
  expect(second.client_turn_id).not.toBe(first.client_turn_id);
  expect(userTexts()).toEqual(['Repeat this question', 'Repeat this question']);
  server.histories.one = detail([
    ...server.histories.one.messages,
    { role: 'user', content: second.query, client_turn_id: second.client_turn_id, turn_uid: '101' },
    { role: 'assistant', content: 'Second answer' },
  ], { current_uid: '101' });
  await event({ done: true, uid: '101', revision: 2 });
  expect(userTexts()).toEqual(['Repeat this question', 'Repeat this question']);
  expect(server.posts).toHaveLength(2);
});

test('an old done SSE cannot end the newer accepted turn', async () => {
  await mount(); await send('New request');
  await event({ done: false, uid: '100', revision: 10 });
  await event({ done: true, uid: '99', revision: 11 });
  expect(container.querySelector('.processing')).not.toBeNull();
  expect(userTexts()).toEqual(['New request']);
});

test('an older SSE revision with the same uid cannot end the active turn', async () => {
  await mount(); await send('Running request');
  await event({ done: false, uid: '100', revision: 10 });
  await event({ done: true, uid: '100', revision: 9 });
  expect(container.querySelector('.processing')).not.toBeNull();
});

test('a rejected POST leaves the user visibly failed and never retries', async () => {
  const rejected = deferred(); server.queryQueue.push(rejected);
  await mount(); await send('Rejected question');
  rejected.resolve(response({ detail: 'Offline request rejected' }, 429)); await flush();
  expect(container.querySelector('.user-message').textContent).toContain('Rejected question');
  expect(container.querySelector('.user-message [role="status"]').textContent).toContain('发送失败');
  expect(container.querySelector('.processing')).toBeNull();
  await remount();
  expect(container.querySelector('.user-message [role="status"]').textContent).toContain('发送失败');
  expect(server.posts).toHaveLength(1);
});

test('a lost POST response stays uncertain until history confirms its client identity', async () => {
  const lost = deferred(); server.queryQueue.push(lost);
  await mount(); await send('Maybe accepted');
  lost.reject(new Error('Offline response lost')); await flush();
  expect(container.querySelector('.user-message [role="status"]').textContent).toContain('状态待核对');
  server.histories.one = detail([{ role: 'user', content: 'Maybe accepted', client_turn_id: server.posts[0].client_turn_id, turn_uid: '100' }], { current_uid: '100', is_processing: true });
  await remount();
  expect(userTexts()).toEqual(['Maybe accepted']);
  expect(container.querySelector('.user-message [role="status"]')).toBeNull();
  expect(server.posts).toHaveLength(1);
});

test('accepted pending history survives refresh before server history catches up', async () => {
  await mount(); await send('Accepted before refresh');
  await remount();
  await event({ done: true, uid: '', revision: 0 });
  expect(userTexts()).toEqual(['Accepted before refresh']);
  expect(container.querySelector('.processing')).not.toBeNull();
  expect(server.posts).toHaveLength(1);
});

test('a POST acknowledgement arriving after a switch cannot leak into another conversation', async () => {
  const ack = deferred(); server.queryQueue.push(ack);
  await mount(); await send('Only in first conversation');
  await switchTo('Second conversation');
  ack.resolve(response({ done: false, uid: '100', conv_id: 'one', client_turn_id: server.posts[0].client_turn_id })); await flush();
  expect(userTexts()).toEqual([]);
  expect(container.querySelector('.processing')).toBeNull();
  await switchTo('First conversation');
  expect(userTexts()).toEqual(['Only in first conversation']);
  expect(server.posts).toHaveLength(1);
});

test('history parsing completed after switching away and back cannot commit an old snapshot', async () => {
  const parsed = deferred(), delayed = deferred();
  await mount();
  server.detailQueue.push(delayed);
  await event({ done: true, uid: '', revision: 0 });
  delayed.resolve({ ok: true, status: 200, text: () => parsed.promise }); await flush();
  await switchTo('Second conversation'); await switchTo('First conversation');
  await send('New selected turn');
  parsed.resolve(JSON.stringify(detail([{ role: 'user', content: 'Obsolete server snapshot' }]))); await flush();
  expect(userTexts()).toEqual(['New selected turn']);
});

test('logout removes owner pending turns and delayed responses cannot restore them', async () => {
  const ack = deferred(); server.queryQueue.push(ack);
  await mount(); await send('Private pending question');
  await act(async () => container.querySelector('.btn-logout').click()); await flush();
  ack.resolve(response({ done: false, uid: '100', conv_id: 'one', client_turn_id: server.posts[0].client_turn_id })); await flush();
  expect(container.querySelector('.auth-container')).not.toBeNull();
  expect(sessionStorage.getItem('bimem_pending_turns_v1:fixture')).toBeNull();
  expect(container.textContent).not.toContain('Private pending question');
});

test('legacy history without identity remains visible without text deduplication', async () => {
  server.histories.one = detail([{ role: 'user', content: 'Same legacy text' }, { role: 'assistant', content: 'Old answer' }]);
  await mount(); await send('Same legacy text');
  expect(userTexts()).toEqual(['Same legacy text', 'Same legacy text']);
});

test('a failed history refresh after HTTP acceptance does not mark the accepted POST failed', async () => {
  const refresh = deferred();
  await mount(); server.detailQueue.push(refresh);
  await send('Already accepted');
  refresh.reject(new Error('Offline history unavailable')); await flush();
  expect(userTexts()).toEqual(['Already accepted']);
  expect(container.querySelector('.processing')).not.toBeNull();
  expect(container.textContent).not.toContain('Offline history unavailable');
  expect(server.posts).toHaveLength(1);
});

test('an older post-ack history cannot erase a newer confirmed answer', async () => {
  await mount();
  const postAckHistory = deferred(); server.detailQueue.push(postAckHistory);
  await send('Accepted while history catches up');
  server.histories.one = detail([
    { role: 'user', content: server.posts[0].query, client_turn_id: server.posts[0].client_turn_id, turn_uid: '100' },
    { role: 'assistant', content: 'Verified newer answer' },
  ], { current_uid: '100' });
  await event({ done: true, uid: '100', revision: 1 });
  expect(container.textContent).toContain('Verified newer answer');
  postAckHistory.resolve(response(detail())); await flush();
  expect(userTexts()).toEqual(['Accepted while history catches up']);
  expect(container.textContent).toContain('Verified newer answer');
});

test('two rapid clicks send only one request before React renders processing', async () => {
  await mount();
  const field = container.querySelector('.input-container textarea');
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(field, 'One deliberate turn');
    field.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await act(async () => {
    container.querySelector('.send-btn').click();
    container.querySelector('.send-btn').click();
  });
  await flush();
  expect(server.posts).toHaveLength(1);
  expect(userTexts()).toEqual(['One deliberate turn']);
});

test('a different authenticated owner never loads another owner pending turns', async () => {
  sessionStorage.setItem('bimem_pending_turns_v1:another-owner', JSON.stringify({ one: [{
    client_turn_id: 'other_turn', content: 'Other owner private question', submission_state: 'accepted', turn_uid: '100',
  }] }));
  await mount();
  expect(userTexts()).toEqual([]);
  expect(container.textContent).not.toContain('Other owner private question');
  expect(server.posts).toHaveLength(0);
});

test('refresh during an unresolved POST preserves an uncertain turn without resubmitting or locking forever', async () => {
  const ack = deferred(); server.queryQueue.push(ack);
  await mount(); await send('Unresolved before refresh');
  await remount();
  expect(container.querySelector('.user-message').textContent).toContain('Unresolved before refresh');
  expect(container.querySelector('.user-message [role="status"]').textContent).toContain('状态待核对');
  expect(container.querySelector('.input-container textarea').disabled).toBe(false);
  expect(server.posts).toHaveLength(1);
});

test('a late first acknowledgement cannot let its done event end a newer second turn', async () => {
  const lateAck = deferred(); server.queryQueue.push(lateAck);
  await mount(); await send('First turn, late acknowledgement');
  server.histories.one = detail([
    { role: 'user', content: server.posts[0].query, client_turn_id: server.posts[0].client_turn_id, turn_uid: '100' },
    { role: 'assistant', content: 'First completed' },
  ], { current_uid: '100' });
  await event({ done: true, uid: '100', revision: 1 });
  await send('Second active turn');
  expect(server.posts).toHaveLength(2);
  lateAck.resolve(response({ done: false, uid: '100', conv_id: 'one', client_turn_id: server.posts[0].client_turn_id })); await flush();
  await event({ done: true, uid: '100', revision: 2 });
  expect(container.querySelector('.processing')).not.toBeNull();
  expect(userTexts()).toEqual(['First turn, late acknowledgement', 'Second active turn']);
});

test('stream disconnection recovers final history through polling without resending the accepted POST', async () => {
  jest.useFakeTimers();
  await mount(); await send('Accepted before stream disconnected');
  server.streams.one.end(); await flush();
  server.histories.one = detail([
    { role: 'user', content: server.posts[0].query, client_turn_id: server.posts[0].client_turn_id, turn_uid: '100' },
    { role: 'assistant', content: 'Recovered final answer' },
  ], { current_uid: '100' });
  server.answers.one = { done: true, uid: '100', conv_id: 'one', answer: 'Recovered final answer', partial_results: [] };
  await act(async () => jest.advanceTimersByTime(15000)); await flush();
  expect(userTexts()).toEqual(['Accepted before stream disconnected']);
  expect(container.textContent).toContain('Recovered final answer');
  expect(container.querySelector('.processing')).toBeNull();
  expect(server.posts).toHaveLength(1);
});

test('a lost response cannot mark a server-confirmed completed turn uncertain', async () => {
  const lost = deferred(); server.queryQueue.push(lost);
  await mount(); await send('Confirmed independently of POST response');
  server.histories.one = detail([
    { role: 'user', content: server.posts[0].query, client_turn_id: server.posts[0].client_turn_id, turn_uid: '100' },
    { role: 'assistant', content: 'Confirmed completed answer' },
  ], { current_uid: '100' });
  await event({ done: true, uid: '100', revision: 1 });
  lost.reject(new Error('Late connection error')); await flush();
  expect(userTexts()).toEqual(['Confirmed independently of POST response']);
  expect(container.querySelector('.user-message [role="status"]')).toBeNull();
  expect(container.textContent).not.toContain('Late connection error');
  expect(server.posts).toHaveLength(1);
});

test('a late lost response from an older confirmed turn cannot stop a newer active turn', async () => {
  const lost = deferred(); server.queryQueue.push(lost);
  await mount(); await send('Older confirmed question');
  server.histories.one = detail([
    { role: 'user', content: server.posts[0].query, client_turn_id: server.posts[0].client_turn_id, turn_uid: '100' },
    { role: 'assistant', content: 'Older completed answer' },
  ], { current_uid: '100' });
  await event({ done: true, uid: '100', revision: 1 });
  await send('New question still running');
  lost.reject(new Error('Late old transport error')); await flush();
  expect(container.querySelector('.processing')).not.toBeNull();
  expect(userTexts()).toEqual(['Older confirmed question', 'New question still running']);
  expect(server.posts).toHaveLength(2);
});

test('a delayed active poll cannot revive a turn after a newer done SSE while history is pending', async () => {
  const oldActivePoll = deferred(), doneHistory = deferred();
  server.answerQueue.push(oldActivePoll);
  server.histories.one = detail([{ role: 'user', content: 'An already accepted turn', turn_uid: '100' }], { current_uid: '100' });
  await mount();
  server.detailQueue.push(doneHistory);
  await event({ done: true, uid: '100', revision: 10 });
  oldActivePoll.resolve(response({ done: false, uid: '100', conv_id: 'one' })); await flush();
  doneHistory.resolve(response(detail([
    ...server.histories.one.messages, { role: 'assistant', content: 'The terminal answer' },
  ], { current_uid: '100' }))); await flush();
  expect(container.textContent).toContain('The terminal answer');
  expect(container.querySelector('.processing')).toBeNull();
  expect(server.posts).toHaveLength(0);
});
