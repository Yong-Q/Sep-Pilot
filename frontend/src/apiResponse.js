function compactText(value) {
  return String(value || '').replace(/\s+/g, ' ').trim().slice(0, 240);
}

export async function readJsonResponse(response) {
  const body = await response.text();
  try {
    return body ? JSON.parse(body) : null;
  } catch (_) {
    const detail = compactText(body);
    const prefix = response.ok ? '服务器返回了无法解析的响应' : '服务暂时不可用';
    throw new Error(`${prefix}（HTTP ${response.status}）${detail ? `：${detail}` : ''}`);
  }
}
