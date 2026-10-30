import { TextEncoder, TextDecoder } from 'util';
import { followWorkflowEvents } from './workflowEvents';

global.TextDecoder = TextDecoder;
const encoder = new TextEncoder();
const headers = () => ({ Authorization: 'Bearer fixture' });
const stream = chunks => {
  const reader = { read: jest.fn(), cancel: jest.fn(async () => {}), releaseLock: jest.fn() };
  chunks.forEach(text => reader.read.mockResolvedValueOnce({ value: encoder.encode(text), done: false }));
  reader.read.mockResolvedValue({ done: true });
  return { response: { ok: true, status: 200, body: { getReader: () => reader } }, reader };
};

test('a closed stream reconnects and delivers a later result with authentication', async () => {
  const control = new AbortController(); const received = [];
  const first = stream(['data: {"revision":1}\n\n']);
  const second = stream(['data: {"revision":2}\n\n']);
  const fetchImpl = jest.fn().mockResolvedValueOnce(first.response).mockResolvedValueOnce(second.response);
  const wait = jest.fn(async () => {});
  await followWorkflowEvents({ url: '/events', headers, signal: control.signal, fetchImpl, wait,
    onEvent: async value => { received.push(value.revision); if (value.revision === 2) control.abort(); } });
  expect(received).toEqual([1, 2]); expect(fetchImpl).toHaveBeenCalledTimes(2);
  expect(fetchImpl.mock.calls[1][1].headers).toEqual(headers());
  expect(first.reader.cancel).toHaveBeenCalled(); expect(second.reader.cancel).toHaveBeenCalled();
});

test('connection failures back off and retry without mutating conversation data', async () => {
  const control = new AbortController(); const received = [];
  const final = stream(['data: {"ready":true}\n\n']);
  const fetchImpl = jest.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(final.response);
  const wait = jest.fn(async () => {});
  await followWorkflowEvents({ url: '/events', headers, signal: control.signal, fetchImpl, wait,
    onEvent: async value => { received.push(value); control.abort(); } });
  expect(received).toEqual([{ ready: true }]); expect(wait.mock.calls[0][0]).toBe(500);
});

test('bounded backoff does not turn service loss into a tight request loop', async () => {
  const control = new AbortController(); const intervals = [];
  await followWorkflowEvents({ url: '/events', headers, signal: control.signal,
    fetchImpl: jest.fn().mockRejectedValue(new Error('offline')), onEvent: jest.fn(),
    wait: async ms => { intervals.push(ms); if (intervals.length === 6) control.abort(); } });
  expect(intervals).toEqual([500, 1000, 2000, 4000, 5000, 5000]);
});

test('expired authentication is terminal for this connection, without repeated requests', async () => {
  const control = new AbortController(); const onState = jest.fn();
  const fetchImpl = jest.fn(async () => ({ status: 401, ok: false }));
  await followWorkflowEvents({ url: '/events', headers, signal: control.signal, fetchImpl,
    onEvent: jest.fn(), onState, wait: jest.fn() });
  expect(fetchImpl).toHaveBeenCalledTimes(1); expect(onState).toHaveBeenCalledWith('authentication_required');
});

test('abort while waiting for reconnect prevents a new connection', async () => {
  const control = new AbortController(); const fetchImpl = jest.fn().mockRejectedValue(new Error('offline'));
  await followWorkflowEvents({ url: '/events', headers, signal: control.signal, fetchImpl,
    onEvent: jest.fn(), wait: async () => control.abort() });
  expect(fetchImpl).toHaveBeenCalledTimes(1);
});

test('CRLF and chunk boundaries preserve full event JSON and non-ASCII text', async () => {
  const control = new AbortController(); const values = [];
  const input = stream([': heartbeat\r\n\r\ndata: {"text":"结果"}\r', '\n\r\ndata: malformed\n\n']);
  await followWorkflowEvents({ url: '/events', headers, signal: control.signal,
    fetchImpl: async () => input.response, onEvent: async data => { values.push(data); control.abort(); } });
  expect(values).toEqual([{ text: '结果' }]);
});

test('cancelled old conversation events cannot be delivered to the new page', async () => {
  const control = new AbortController(); const onEvent = jest.fn();
  const input = stream([]);
  input.reader.read.mockImplementation(async () => { control.abort(); return { value: encoder.encode('data: {"old":true}\n\n') }; });
  await followWorkflowEvents({ url: '/events', headers, signal: control.signal,
    fetchImpl: async () => input.response, onEvent });
  expect(onEvent).not.toHaveBeenCalled(); expect(input.reader.releaseLock).toHaveBeenCalled();
});
