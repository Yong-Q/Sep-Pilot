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

## Follow-up inspection: state handling and responsiveness

On 2026-10-27, the GitHub API still reported `e49d3156656900f1e20e196afbe50c38c3f68da0`
as the upstream `main` revision. These observations refer to that implementation,
not to unimplemented design documents or a comparative performance benchmark.

| Verified implementation | Adaptation to this application |
|---|---|
| `ToolRegistry._handle_oversized_output` writes complete oversized output to `.jarvis/tool_results/` and returns its location, size and line count. | Keep large scientific evidence separately; send bounded, typed facts and exact evidence references to agents and UI consumers. Model-input compaction alone does not shrink the persistent lifecycle mailbox. |
| `TaskList` keeps tasks in a dictionary, looks up individual task IDs, and increments a version when updating state. | Separate the current task projection from historical evidence. Routine UI reads should not reconstruct every historical event. This is a proposed adaptation, not a claim that Jarvis implements our durable scheduler receipt protocol. |
| `HistoryCompressor.start_background_pre_compression` snapshots history and starts a background thread before the full-compression threshold. | Prepare model-context views outside request handling, preserve recent tool/result pairs, and invalidate prepared views when their source version changes. |
| `AgentRunLoop` returns to its main loop after native tool execution, where buffered input and interruption handling can participate in the next iteration. | Check the current user turn and pause state at dispatch and result-delivery boundaries, including after asynchronous storage waits. An older actor must not clear or overwrite a newer turn. |

The compressor is not universally nonblocking: its full-compression path can
wait up to 60 seconds for background preparation. Likewise, the task-list module
does not establish that the entire upstream service uses append-only storage or
that it satisfies scientific result verification requirements.

The first SEP-Pilot changes make unchanged lifecycle receipts idempotent and
move blocking storage work out of the API event loop, with ownership and
cancellation regression coverage. The next implementation separates small
current-state views from immutable historical evidence. Deployment latency and
complete scientific execution must be measured independently of unit tests.

## Current-state and idle-write changes (not delivery acceptance)

The isolated implementation now stores revision payloads and large named chain
histories in owner-scoped, content-addressed evidence files. A small graph
projection is published after the manifest commit and validated against its
owner and manifest identity. Migration archives exact legacy bytes and retains
the original revision hash material. Reading the UI never hydrates old revisions.

Workflow API requests share one per-conversation build, with one dedicated
builder thread and bounded serialized caching. Current contracts, input
bindings, resource locks, result references and scientific verification remain
visible; raw tool payloads do not enter the routine view. Cancellation does not
release the builder permit while its real native reader remains alive. New user
turn IDs prevent a delayed history response from erasing a newer user message.

Liveness timestamps move to a small sidecar; actual claims, receipts and overdue
role warnings still commit through the original mailbox transaction. Heartbeat
checks still read the mailbox to identify an overdue role. This removes the
periodic full rewrite, not every historical read. Further idle-scan reductions
must preserve event wakeups, scheduler reconciliation and user pause controls.

An isolated same-input benchmark on node05 used 20 transitions and 20 graph
reads. The V14 chain manifest grew to 15,622,801 bytes; the new manifest was
12,908 bytes, with exact history recovery verified. Median graph reads were
30.12 ms and 0.047 ms respectively. Separate process peak RSS was 137,004 KiB
and 59,516 KiB. This synthetic benchmark is not production response latency or
an end-to-end scientific acceptance result.

Upstream's module-level builtin tool cache is relevant, but our registry already
has a process singleton and scientific adapters load at call boundaries. Repeating
that cache would not fix the measured mailbox/chain workload. Likewise, Jarvis's
in-memory emergency session daemon is not a substitute for durable scientific
job ownership, receipt validation and prevention of duplicate submissions.
