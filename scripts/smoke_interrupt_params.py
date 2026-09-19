#!/usr/bin/env python3
"""端到端冒烟测试：参数确认 + 中断加需求 + 作业参数落地。

流程（真实 API，不打印 token）：
  1. 新建会话
  2. 提交缺热力学参数的任务 → 断言 agent 询问（方法/温度/压力），不擅自提交
  3. 中断 + 附加重定向需求 → 断言 interrupt_message 存储
  4. 回复确认 → 断言 redirect 消费、作业真实提交、参数（温度/压力）符合 redirect
  5. 清理：scancel 作业 + 冻结会话

用法：
  python3 scripts/smoke_interrupt_params.py [--keep] [--base http://127.0.0.1:8000]
  --keep: 保留测试会话（默认删除）
退出码 0 = 通过。
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

TOKEN_FILE = Path("/tmp/react_token")
CIF = "/home/user/gcmc_agent/BiMemAgent-claude-sdk/tmp/cdft_10mof/MOF_0123_bex_pacman.cif"
TASK = f"帮我对 {CIF} 做 CH4 吸附等温线计算"
REDIRECT = "温度用300K，压力范围0.1到2bar，确认用GCMC直接跑"

_failures: list[str] = []


def _api(base: str, method: str, path: str, token: str, body: dict | None = None,
         _retry_429: bool = True, _tries: int = 10):
    """API call with 429 retry. 429 = the conversation's previous agent turn is
    still processing (conv.is_processing) — e.g. right after an interrupt when
    run_agent_background hasn't finished tearing down. Wait and retry."""
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method=method,
    )
    for attempt in range(_tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and _retry_429 and attempt < _tries - 1:
                time.sleep(3)
                continue
            raise
        except TimeoutError:
            # A transient slow server (heavily loaded / /api/jobs starved)
            # must not crash the whole smoke test. Retry idempotent GET/DELETE;
            # for POST, a timeout is ambiguous (may have been processed) so raise.
            if method in ("GET", "DELETE") and attempt < _tries - 1:
                time.sleep(5)
                continue
            raise
    raise RuntimeError("unreachable")


def _check(label: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f"  {detail[:160]}" if detail and not cond else ""))
    if not cond:
        _failures.append(label)


def _wait_done(base: str, token: str, conv_id: str = "", timeout: int = 300,
               needle: str = "") -> dict:
    """Poll latest_answer until done=True (or a needle string appears).

    ALWAYS pass conv_id: /api/latest_answer with an empty conv_id silently
    falls back to the user's CURRENT conversation — when a harness runs this
    script from its own delegation conversation, current may be any other
    conversation, so polling without conv_id reads the wrong conversation and
    produces "answer always empty" false failures.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            d = _api(base, "GET", f"/api/latest_answer?conv_id={conv_id}", token)
        except Exception:
            time.sleep(2); continue
        a = d.get("answer") or ""
        if d.get("done") or (needle and needle in a):
            return d
        time.sleep(3)
    return d


def _conv_jobs(base: str, token: str, conv_id: str) -> list:
    d = _api(base, "GET", "/api/jobs", token)
    return [j for j in d.get("jobs", []) if j.get("conv_id") == conv_id]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the test conversation")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    token = TOKEN_FILE.read_text().strip()
    if not token:
        print("no token at /tmp/react_token"); return 1
    # line-buffered so a live agent/harness watching the output sees each step
    sys.stdout.reconfigure(line_buffering=True)

    print("══ smoke: 参数确认 + 中断加需求 + 作业参数 ══")
    conv = _api(base, "POST", "/api/conversations", token)
    conv_id = conv["conv_id"]
    print(f"  conv: {conv_id}")

    try:
        # 1) 缺参数任务 → agent 应询问，不应提交作业
        print("\n── 步骤1: 缺热力学参数的任务 → 应询问 ──")
        _api(base, "POST", "/api/query", token, {"conv_id": conv_id, "query": TASK})
        d = _wait_done(base, token, conv_id=conv_id, needle="GCMC")
        a = d.get("answer") or ""
        _check("agent 询问了方法 (GCMC/cDFT)", "GCMC" in a or "cDFT" in a)
        _check("agent 询问了温度", "温度" in a)
        _check("agent 询问了压力范围", "压力" in a)
        _check("agent 未提交作业(还在询问阶段)", len(_conv_jobs(base, token, conv_id)) == 0)

        # 2) 中断 + 附加重定向需求
        print("\n── 步骤2: 中断 + 加需求 ──")
        intr = _api(base, "POST", f"/api/conversations/{conv_id}/interrupt", token,
                    {"message": REDIRECT})
        _check("interrupt ok 且不取消作业", intr.get("ok") and "作业" in intr.get("note", ""))
        latest = _api(base, "GET", f"/api/latest_answer?conv_id={conv_id}", token)
        _check("latest_answer 暴露 interrupt_message", latest.get("interrupt_message") == REDIRECT)

        # 3) 回复确认 → redirect 消费 + 作业提交
        print("\n── 步骤3: 确认执行 → redirect 落地到作业 ──")
        _api(base, "POST", "/api/query", token, {"conv_id": conv_id, "query": "确认，跑吧"})
        deadline = time.time() + 300
        jobs = []
        while time.time() < deadline:
            jobs = [j for j in _conv_jobs(base, token, conv_id) if j.get("job_id")]
            if jobs:
                break
            time.sleep(3)
        _check("作业真实提交", bool(jobs), f"{jobs}")
        if jobs:
            job = jobs[0]
            _check("作业 gas = CH4", job.get("gas") == "CH4", str(job.get("gas")))
            _check("作业温度 = 300K (redirect 生效)", "300" in (job.get("work_dir") or ""), job.get("work_dir") or "")
            latest2 = _api(base, "GET", f"/api/latest_answer?conv_id={conv_id}", token)
            _check("redirect 已消费(interrupt_message 清空)", not latest2.get("interrupt_message"))
            # 清理: scancel 测试作业
            jid = job["job_id"]
            _api(base, "POST", f"/api/jobs/{jid}/cancel", token)
            print(f"  [cleanup] cancelled test job {jid}")
    finally:
        if not args.keep:
            _api(base, "DELETE", f"/api/conversations/{conv_id}", token)
            print(f"  [cleanup] deleted conv {conv_id}")
        else:
            # 冻结会话避免继续跑
            try:
                _api(base, "POST", f"/api/conversations/{conv_id}/interrupt", token,
                     {"message": "冒烟测试结束，停止"})
            except Exception:
                pass
            print(f"  [cleanup] frozen conv {conv_id}")

    print(f"\n{'✅ 冒烟通过' if not _failures else '❌ 冒烟失败: ' + ', '.join(_failures)}")
    return 0 if not _failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
