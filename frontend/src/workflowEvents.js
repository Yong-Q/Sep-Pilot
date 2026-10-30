// Authenticated SSE with bounded reconnects; a disconnect must not leave the
// page on the slower polling fallback until the user refreshes the browser.
function waitForRetry(ms, signal) {
  return new Promise(resolve => {
    if (signal.aborted) { resolve(); return; }
    const finish = () => { clearTimeout(timer); signal.removeEventListener('abort', finish); resolve(); };
    const timer = setTimeout(finish, ms);
    signal.addEventListener('abort', finish, { once: true });
  });
}

export async function followWorkflowEvents({ url, headers, signal, onEvent,
  onState = () => {}, fetchImpl = fetch, wait = waitForRetry,
  minRetryMs = 500, maxRetryMs = 5000 }) {
  let retryMs = minRetryMs;
  while (!signal.aborted) {
    let reader;
    try {
      const response = await fetchImpl(url, { headers: headers(), signal });
      if (signal.aborted) break;
      if ([401, 403].includes(response.status)) {
        onState('authentication_required');
        return;
      }
      if (!response.ok || !response.body) throw new Error('event stream unavailable');
      reader = response.body.getReader();
      onState('connected');
      const decoder = new TextDecoder();
      let buffer = '';
      while (!signal.aborted) {
        const { value, done } = await reader.read();
        if (done || signal.aborted) break;
        buffer += decoder.decode(value, { stream: true });
        if (buffer.length > 2 * 1024 * 1024) throw new Error('event frame exceeds bound');
        let boundary;
        while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
          const frame = buffer.slice(0, boundary.index);
          buffer = buffer.slice(boundary.index + boundary[0].length);
          const payload = frame.split(/\r?\n/).filter(line => line.startsWith('data:'))
            .map(line => line.slice(5).replace(/^ /, '')).join('\n');
          if (!payload || signal.aborted) continue;
          let data;
          try { data = JSON.parse(payload); } catch (_) { continue; }
          retryMs = minRetryMs;
          await onEvent(data);
        }
      }
    } catch (error) {
      if (signal.aborted) break;
      // Connection and handler errors keep the existing history/goal intact.
    } finally {
      if (reader) {
        try { await reader.cancel(); } catch (_) {}
        try { reader.releaseLock(); } catch (_) {}
      }
    }
    if (signal.aborted) break;
    onState('reconnecting');
    await wait(retryMs, signal);
    retryMs = Math.min(maxRetryMs, retryMs * 2);
  }
}
