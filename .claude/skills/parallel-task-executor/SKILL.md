---
name: parallel-task-executor
description: >
  Use when coordinating multiple requested tasks, independent scientific branches, or a user parameter change in an existing workflow.
metadata:
  generated: true
---

# Parallel Task Orchestration

> Generated from registry/catalog.json. Do not hand-edit generated md files.

## Use When

- Use when coordinating multiple requested tasks, independent scientific branches, or a user parameter change in an existing workflow.

## Do Not Use For

- Do not replace the persistent DAG executor, submit scientific jobs directly, or configure an external OpenClaw runtime.

## Inputs

- `current user goal and explicit parameter changes`
- `owned workflow/node/attempt and scheduler evidence`
- `dependencies and resource availability`

## Automation

- Level: `metadata`
- Entry: `Use the existing structured workflow draft, versioned patch and execution tools. The native executor owns scheduling and recovery; this skill is guidance only.`
- Spec format: `docs/workflow-spec.md`

## Legacy Source

- `https://hub.cocoloop.cn/skills/1559`

## Notes

- Adapted independently from the task decomposition, dependency, result collection and progress concepts of parallel-task-executor v1.0.0 by Erichy777 at https://hub.cocoloop.cn/skills/1559. The downloaded guide contains no execution engine; its illustrative scheduler code and OpenClaw configuration are not bundled.
- Keep one authoritative task graph. Each branch records its owner, inputs, expected artifacts and actual dependencies. Run independent branches concurrently only when their resource leases and outputs are compatible; join verified evidence before delivering a combined result.
- On an explicit user parameter change, reconcile the current goal with pending typed node arguments, propose the corresponding versioned graph patch, then apply and execute within that existing user authority. Do not ask the user to repeat an unambiguous parameter instruction. Preserve unaffected verified results.
- A running or uncertain submission retains its job and attempt identity. Report whether old parameters remain active; reconcile that job before a conflicting edit. Changing a future parameter does not authorize cancelling or repeating an existing job.
- Use returned schema, contract and activation errors to choose the next concrete repair. A goal/plan/argument mismatch is a graph reconciliation issue; delegate framework repair only when execution evidence demonstrates a code defect. Repeated reads or handoffs alone are not progress.
- Report actual pending, running, blocked, validated and delivered states with job or artifact references. A model reply, tool proposal or queued task is not a completed calculation. Derive progress from verified work and disclose blocked branches.
- Use configured concurrency, scientific walltimes and native retry receipts. Generic example defaults such as four workers, thirty seconds or three retries do not override user scope, compute resource review, idempotency or scientific acceptance.
