import { readJsonResponse } from './apiResponse';

function response({ status = 200, contentType = 'application/json', body }) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: name => name.toLowerCase() === 'content-type' ? contentType : null },
    text: async () => body,
  };
}

test('reports a proxy text response as a readable service error', async () => {
  const result = response({
    status: 502,
    contentType: 'text/plain',
    body: 'Proxy error: could not proxy request /api/query',
  });

  await expect(readJsonResponse(result)).rejects.toThrow(
    '服务暂时不可用（HTTP 502）：Proxy error: could not proxy request /api/query'
  );
});

test('returns JSON response data unchanged', async () => {
  const result = response({ body: JSON.stringify({ done: false }) });

  await expect(readJsonResponse(result)).resolves.toEqual({ done: false });
});
