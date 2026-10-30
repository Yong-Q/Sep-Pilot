# Codex CLI provider boundary

The application passes its conversation and tool schemas to Codex CLI as a
structured-response request. The host Session validates and dispatches returned
application tool proposals. CLI-native tools are a separate capability surface.

The launcher explicitly disables `shell_tool`. In the verified CLI version
0.163.0, this prevents registration of `exec_command` and `write_stdin`, even
while `unified_exec` remains enabled. Disabling `unified_exec` alone is ignored
by that version. See the [official registration gate](https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/core/src/tools/spec_plan.rs#L1245)
and its [regression test](https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/core/src/tools/spec_plan_tests.rs#L1013).

The launcher also disables shell snapshots, apps, plugins, hooks, subagents,
code mode, image/browser/computer capabilities, sleep and tool suggestions. It
requests an empty MCP configuration and disables web search. These settings
reduce independent CLI activity; they do not establish that every native tool
is absent. In particular, [apply_patch registration](https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/core/src/tools/spec_plan.rs#L1432)
depends separately on model metadata and may remain available.

The existing read-only sandbox, model/reasoning settings and filesystem mounts
are unchanged. The signed-in authentication file is still mounted read-only for
the CLI process. A read-only mount prevents writes; it does not establish
credential confidentiality from the process that can access it.

Command-capture tests check the actual launcher arguments, existing filesystem
mounts and environment allowlist without invoking a model or reading credentials.
Local feature/config checks establish compatibility with CLI 0.163.0; they are
not a latency benchmark or scientific workflow acceptance.
