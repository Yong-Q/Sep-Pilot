# Runtime lessons from Jarvis

Reviewed upstream commit: [`e49d315`](https://github.com/skyfireitdiy/Jarvis/tree/e49d3156656900f1e20e196afbe50c38c3f68da0).
This is an implementation comparison, not a claim that either project's full product has been validated.

| Mechanism verified in source | Relevance to SEP-Pilot |
|---|---|
| [`OpenAIPlatform.chat_native_once`](https://github.com/skyfireitdiy/Jarvis/blob/e49d3156656900f1e20e196afbe50c38c3f68da0/src/jarvis/jarvis_platform/openai.py) streams content and structured tool calls. | Model transport and scientific execution should remain separate. The existing Codex CLI bridge still has whole-response latency; smaller inputs do not solve streaming by themselves. |
| [`native_tools.py`](https://github.com/skyfireitdiy/Jarvis/blob/e49d3156656900f1e20e196afbe50c38c3f68da0/src/jarvis/jarvis_platform/native_tools.py) normalizes provider messages and repairs call/result pairing. | Context compaction must preserve tool IDs and complete pairs across providers. |
| [`ToolRegistry._handle_oversized_output`](https://github.com/skyfireitdiy/Jarvis/blob/e49d3156656900f1e20e196afbe50c38c3f68da0/src/jarvis/jarvis_tools/registry.py) stores large output and returns a read location. | Avoid sending full historical state snapshots on every round; keep complete evidence available for targeted reads. |
| [`history_compressor.py`](https://github.com/skyfireitdiy/Jarvis/blob/e49d3156656900f1e20e196afbe50c38c3f68da0/src/jarvis/jarvis_agent/history_compressor.py) separates background summary preparation from normal execution. | Context management belongs in the runtime, rather than an expanding collection of task-specific prompt instructions. |
| [`task_list.py`](https://github.com/skyfireitdiy/Jarvis/blob/e49d3156656900f1e20e196afbe50c38c3f68da0/src/jarvis/jarvis_agent/task_list.py) models dependencies and tracks model-call counts. | Measure progress and cost per task. Its unrestricted task status updates must not replace SEP-Pilot's scheduler and scientific artifact validation. |

Documentation and implementation differ in places. For example, a context-management design document is explicitly marked pending review; current run-loop code uses background preparation at 75% occupancy and a full-compression trigger at 90%. Treat design proposals as proposals, not verified behavior.

## First independently implemented improvement

`agents/model_context.py` constructs a temporary model-input view. It replaces older successful identical state queries with references and previews large state snapshots only when an evidence file contains the matching full result. It preserves original history, user messages, failures, scientific outputs and call/result pairing. It does not call another model or grant execution permission.

A local saved conversation measured 87,473 history characters before and 46,659 after this transformation: 46.7% fewer history characters. This excludes system instructions and tool schemas, and is not a latency benchmark or an end-to-end acceptance result.

No upstream implementation was copied. The remaining product work includes transport latency, bounded recovery, deployment-path correctness, and independent user-facing workflow acceptance. Scientific submission and completion remain tied to actual scheduler receipts and validated artifacts.
