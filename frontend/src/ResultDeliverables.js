import React, { useEffect, useRef, useState } from 'react';

// Fetch with Authorization; never put account tokens or host paths in links.
export default function ResultDeliverables({ owner, convId, revision, authHeaders }) {
  const scope = JSON.stringify([owner, convId]);
  const [state, setState] = useState({ scope: '', artifacts: [], error: '', preview: null });
  const generation = useRef(0);
  const [refreshKey, setRefreshKey] = useState(0);
  const retryDelay = useRef(1000);
  const urls = useRef(new Set());
  const requests = useRef(new Set());
  const previewUrl = useRef(null);
  const timers = useRef(new Set());
  const current = state.scope === scope ? state : { artifacts: [], error: '', preview: null };
  const base = '/api/conversations/' + encodeURIComponent(convId || '') + '/artifacts';
  const clear = () => {
    requests.current.forEach(request => request.abort()); requests.current.clear();
    urls.current.forEach(url => URL.revokeObjectURL(url)); urls.current.clear();
    timers.current.forEach(timer => clearTimeout(timer)); timers.current.clear();
    previewUrl.current = null;
  };
  const releasePreview = () => {
    if (previewUrl.current) {
      URL.revokeObjectURL(previewUrl.current); urls.current.delete(previewUrl.current);
      previewUrl.current = null;
    }
  };

  useEffect(() => {
    const version = ++generation.current;
    let retryTimer;
    clear();
    setState(previous => ({ scope, artifacts: [], error: previous.scope === scope ? previous.error : '', preview: null }));
    const headers = authHeaders();
    if (!owner || !convId || !headers.Authorization) return undefined;
    const controller = new AbortController(); requests.current.add(controller);
    fetch(base, { headers, signal: controller.signal })
      .then(async response => {
        if (!response.ok) {
          const error = new Error(response.status === 401 ? '登录状态已失效，请重新登录。' : '结果文件暂时无法读取，正在自动重试。');
          error.retryable = ![401, 403].includes(response.status);
          throw error;
        }
        return response.json();
      }).then(body => {
        if (generation.current === version && !controller.signal.aborted) {
          retryDelay.current = 1000;
          setState({ scope, artifacts: body.artifacts || [], error: '', preview: null });
        }
      }).catch(error => {
        if (generation.current === version && error.name !== 'AbortError') {
          setState({ scope, artifacts: [], error: error.message, preview: null });
          if (error.retryable !== false) {
            retryTimer = setTimeout(() => {
              if (generation.current === version) setRefreshKey(value => value + 1);
            }, retryDelay.current);
            retryDelay.current = Math.min(30000, retryDelay.current * 2);
          }
        }
      }).finally(() => requests.current.delete(controller));
    return () => { ++generation.current; clearTimeout(retryTimer); clear(); };
  }, [scope, revision, base, owner, convId, authHeaders, refreshKey]);

  const open = async (artifact, preview) => {
    if (artifact.kind === 'source_document' || /\.pdf$/i.test(artifact.name || '')) preview = false;
    const version = generation.current;
    const controller = new AbortController(); requests.current.add(controller);
    try {
      const response = await fetch(base + '/' + encodeURIComponent(artifact.artifact_id), {
        headers: authHeaders(), signal: controller.signal,
      });
      if (!response.ok) throw new Error('文件校验未通过或已更新，请刷新结果列表。');
      const blob = await response.blob();
      if (generation.current !== version || controller.signal.aborted) return;
      if (preview && artifact.kind !== 'figure') {
        const text = await blob.text();
        if (generation.current !== version || controller.signal.aborted) return;
        releasePreview();
        setState(previous => ({ ...previous, error: '', preview: { name: artifact.name, text } }));
      } else {
        const url = URL.createObjectURL(blob); urls.current.add(url);
        if (preview) {
          releasePreview(); previewUrl.current = url;
          setState(previous => ({ ...previous, error: '', preview: {
            name: artifact.name, url, visualReview: artifact.visual_review,
          } }));
        } else {
          const link = document.createElement('a'); link.href = url; link.download = artifact.name;
          document.body.appendChild(link); link.click(); link.remove();
          const timer = setTimeout(() => {
            URL.revokeObjectURL(url); urls.current.delete(url); timers.current.delete(timer);
          }, 60000);
          timers.current.add(timer);
        }
      }
    } catch (error) {
      if (generation.current === version && error.name !== 'AbortError') {
        setState(previous => ({ ...previous, error: error.message }));
      }
    } finally { requests.current.delete(controller); }
  };

  if (!current.artifacts.length && !current.error) return null;
  return <section className="runtime-section result-deliverables" aria-label="结果文件">
    <div className="runtime-section-title"><span>结果文件</span><span>{current.artifacts.length} 个</span></div>
    {current.error && <p role="alert">{current.error}</p>}
    {current.artifacts.map(artifact => <div className="result-deliverable" key={artifact.artifact_id}>
      <div><b>{artifact.name}</b><small>{artifact.kind === 'figure' ? '图片' : artifact.kind === 'report' ? '报告' : artifact.kind === 'source_document' ? '原文' : '源数据'}{artifact.kind === 'report' && artifact.report_validation_mode === 'source_numeric' ? ' · 仅来源与数字校验，未独立审核' : ''}</small></div>
      {artifact.kind !== 'source_document' && !/\.pdf$/i.test(artifact.name || '') && <button type="button" onClick={() => open(artifact, true)}>预览</button>}
      <button type="button" onClick={() => open(artifact, false)}>下载</button>
    </div>)}
    <button type="button" onClick={() => {
      retryDelay.current = 1000; setRefreshKey(value => value + 1);
    }}>刷新结果</button>
    {current.preview && <div className="result-preview">
      <div><b>{current.preview.name}</b><button type="button" onClick={() => {
        releasePreview(); setState(previous => ({ ...previous, preview: null }));
      }}>关闭预览</button></div>
      {current.preview.url ? <>
        <img src={current.preview.url} alt={current.preview.name} style={{ maxWidth: '100%', height: 'auto' }} />
        {current.preview.visualReview === 'pending' && <p>图片已生成，尚未完成视觉审核。</p>}
      </> : <pre style={{ whiteSpace: 'pre-wrap', maxHeight: '28rem', overflow: 'auto' }}>{current.preview.text}</pre>}
    </div>}
  </section>;
}
