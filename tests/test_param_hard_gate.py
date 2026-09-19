"""Regression tests for typed thermodynamic negotiation:

- start(): lexical hints are advisory; the main model sees the request so a
  read-only audit mentioning GCMC/cDFT is not misclassified as computation.
- real compute is blocked later by the typed DAG-node contract when protected
  scientific choices are absent.
- reply(): "确认，跑吧" without actual params must re-ask, not silently submit.
- reply(): a redirect carrying the params (interrupt + 加需求) is honored.
- User authorization ("你推荐" / "你自己定") bypasses the gate.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.session import Session
from agents.config import get_config
from agents.defns import ORCHESTRATOR
from agents.goal_contract import GoalContract


def _sess():
    s = Session(config=get_config())
    s.current_agent = ORCHESTRATOR
    s._pending_param_question = ""
    return s


CIF = "/home/user/gcmc_agent/BiMemAgent-claude-sdk/tmp/cdft_10mof/MOF_0123_bex_pacman.cif"
TASK = f"帮我对 {CIF} 做 CH4 吸附等温线计算"


def test_start_missing_thermo_reaches_model_with_structured_advisory():
    """The model determines intent; a word-list no longer answers for it."""
    s = _sess()
    class _B:
        type = "text"
        text = "需要先确认科学条件，尚未创建任何计算节点。"
    class _Resp:
        content = [_B()]
    from unittest.mock import patch
    with patch.object(s, "_call_api", return_value=_Resp()) as call:
        out = s.start(TASK, agent=ORCHESTRATOR, max_rounds=3)
    assert call.called
    assert "尚未创建任何计算节点" in out
    assert not s._pending_param_question


def test_reply_vague_confirmation_reasks_not_submits():
    """A reply that only confirms ('确认，跑吧') without supplying params must
    NOT proceed to the LLM (agent would self-supply 298K and submit)."""
    s = _sess()
    s._pending_param_question = "计算方法(GCMC 或 cDFT)、温度(如298K)、压力范围(如0.1-10 bar)"
    from unittest.mock import patch
    with patch.object(s, "_call_api",
                      side_effect=AssertionError("LLM must not run on a vague confirmation")):
        out = s.reply("确认，跑吧", max_rounds=3)
    assert "确认" in out and "温度" in out
    assert s._pending_param_question             # still pending → agent can't submit


def test_reply_full_params_releases_gate():
    """A reply supplying method+temp+pressure releases the gate and runs."""
    s = _sess()
    s._pending_param_question = "计算方法(GCMC 或 cDFT)、温度(如298K)、压力范围(如0.1-10 bar)"

    class _B:
        type = "text"
        text = "好的，开始执行。"
    class _Resp:
        content = [_B()]

    from unittest.mock import patch
    with patch.object(s, "_call_api", return_value=_Resp()):
        out = s.reply("用GCMC，温度298K，压力0.1到10bar", max_rounds=2)
    assert s._pending_param_question == ""        # gate released
    assert out


def test_redirect_with_params_honored():
    """Interrupt redirect that itself carries the params must be honored even
    when the follow-up message is only '确认，跑吧'."""
    s = _sess()
    s._pending_param_question = "计算方法(GCMC 或 cDFT)、温度(如298K)、压力范围(如0.1-10 bar)"
    s._on_interrupt_message = lambda: "温度用300K，压力范围0.1到2bar，确认用GCMC直接跑"
    s._on_interrupt_clear = lambda: None

    class _B:
        type = "text"
        text = "按重定向执行。"
    class _Resp:
        content = [_B()]

    from unittest.mock import patch
    with patch.object(s, "_call_api", return_value=_Resp()):
        out = s.reply("确认，跑吧", max_rounds=2)
    assert s._pending_param_question == ""        # redirect supplied the params
    assert out


def test_authorized_self_decision_bypasses_gate():
    """User explicitly authorizing '你推荐/你自己定' bypasses the hard gate."""
    s = _sess()
    s._pending_param_question = "计算方法(GCMC 或 cDFT)、温度(如298K)、压力范围(如0.1-10 bar)"

    class _B:
        type = "text"
        text = "好的，我查文献后推荐参数。"
    class _Resp:
        content = [_B()]

    from unittest.mock import patch
    with patch.object(s, "_call_api", return_value=_Resp()):
        out = s.reply("你推荐吧", max_rounds=2)
    assert s._pending_param_question == ""        # authorized → proceed
    assert out


def test_missing_param_gate_uses_accumulated_goal_contract_and_deduplicates():
    s = _sess()
    s.goal_contract = GoalContract.from_user_message(
        "用 GCMC 计算 MFI 中 C2H4/C2H6 在 298K、0.1-10 bar 的吸附等温线"
    )
    assert s._detect_missing_params("继续计算和分析选择性") == ""

    s.goal_contract = GoalContract.from_user_message("请计算吸附和扩散")
    missing = s._detect_missing_params("请计算吸附、电荷和扩散")
    assert missing.split("、").count("具体MOF材料名称") == 1
    assert missing.split("、").count("气体种类") == 1
