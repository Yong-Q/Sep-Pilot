---
name: software-delivery
description: >
  Use when repairing or delivering the agent application, including UI interactions, planning, execution nodes, result validation, supervision, or model/tool integration.
metadata:
  generated: true
---

# Software Delivery Acceptance

> Generated from registry/catalog.json. Do not hand-edit generated md files.

## Use When

- Use when repairing or delivering the agent application, including UI interactions, planning, execution nodes, result validation, supervision, or model/tool integration.

## Do Not Use For

- Do not apply software release gates to routine scientific conversation or count software fixtures as physical validation.

## Inputs

- `user-visible failure or feature`
- `owned session/node/attempt evidence`
- `affected source revision`
- `authorized delivery scope`

## Automation

- Level: `metadata`
- Entry: `Use run_project_regressions and build_project_frontend for supporting checks; exercise the affected production Agent Session and actual UI interaction before declaring that user journey complete.`
- Spec format: `docs/workflow-spec.md`

## Legacy Source

- `.claude/skills/software-delivery/SKILL.md`

## Notes

- Trace the user journey from UI action through the production Agent, planner, node dispatch, detector/validator, recovery, downstream continuation, and visible result. Reproduce the reported failure before editing and fix its earliest demonstrated cause.
- Divide independent UI, executor, and scientific-review checks among agents with explicit ownership. Exchange exact workflow/node/attempt IDs, call IDs, source revisions and unresolved findings; reconcile these at a join before delivery.
- For UI changes exercise click/keyboard, selection, refresh, delayed polling and relevant mobile behavior against actual rendered components. Build success or a static screenshot does not prove interaction/state synchronization. Report DOM coverage separately when a real browser is unavailable.
- For planner/executor changes exercise the relevant success and failure transitions: schema error, stale path, invalid plan, result rejection, delayed/duplicate event, pause and restart. Preserve completed nodes and user pauses; an uncertain dispatch must not be blindly resubmitted.
- Trace detection responsiveness using event created/claimed/receipt/next-action times. An error message alone is not recovery. Verify that the responsible main/supervisor agent receives evidence, repairs within existing authority, and releases the correct successors; retain the initial failure and final evidence.
- Recoverable technical failures are owned by the agents: inspect the authoritative chain, schema and artifacts, repair and revalidate without asking the user for confirmation. Escalate only after evidence-backed feasible repairs are exhausted or required scientific intent cannot be inferred; include attempted fixes and the specific missing decision. Respect explicit user pauses and never duplicate an uncertain submission.
- Use real production Agent definitions/model sessions for end-to-end scientific workflow acceptance. Mark synthetic inputs and mocked scheduler outputs explicitly. Nonempty answers and allowed-tool checks measure transport success, not scientific accuracy or delivery completion.
- Use the agreed scientific contract and evidence for result acceptance. Do not invent universal sample counts, fit scores, cycle counts or filenames; distinguish optional-field defaults from missing scientific choices and never fabricate labels or measurements.
- Validate the fixed source revision with focused regressions and the relevant user journey; include source hashes and outcome evidence. Once relevant checks pass, broaden testing only for a changed boundary, failure or unresolved risk.
- Before an authorized release, inspect the exact staged/public tree for credentials, private paths and user/session data, and test core behavior without private site configuration. Distinguish optional scientific installations from core defects. Claim only the scopes actually verified; maintain an experimental branch while required acceptance remains incomplete.
