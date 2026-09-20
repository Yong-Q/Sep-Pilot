# Changelog

This file records user-visible changes to Sep-Pilot. Implementation details and private deployment information are maintained separately.

## v3.4.76 — 2026-09-20

- Added dedicated report workers that generate evidence-scoped fragments while independent calculations continue.
- Added live report previews to the workflow panel and durable compact receipts for restart recovery.
- Added stale-attempt and cross-session protections before final report assembly.
- Supported workflow patches that add machine-learning branches and append their validated results to the final report.

## v3.4.74 — 2026-09-20

- Added one-pass, Agent-reviewed result dossiers for scheduler-backed calculations.
- Bound validation evidence to the exact DAG node, attempt, job receipts, contract, and result root.
- Added native result checks for structure generation, charge assignment, cDFT, GCMC, MD, and electronic DFT without relying on output filenames or generic error words.
- Kept recoverable result anomalies inside the Agent workflow and prevented validation from resubmitting completed jobs.

## v3.4.73 — 2026-09-20

- Automatically cleared stale cache-review pauses while an unchanged workflow node is still running or validating.
- Kept completed scheduler work moving into agent-owned result validation without another user decision.

## v3.4.72 — 2026-09-20

- Kept the complete workflow graph visible when a same-version partial snapshot arrives.
- Merged live executor status and scheduler job IDs into the persisted workflow topology.
- Included compiled frontend assets in release integrity checks.

## v3.4.71 — 2026-09-20

- Treated resource-review receipt IDs as operational evidence rather than calculation inputs.
- Added automatic migration for retry records created by earlier receipt-sensitive versions.
- Prevented equivalent resource reviews from blocking verified calculation retries.

## v3.4.70 — 2026-09-20

- Reused verified resource-review receipts when retrying unchanged calculation nodes.
- Prevented calculation tools from waiting on duplicate model-based resource reviews.
- Applied the retry behavior consistently to all scheduler-submitting tools.

## v3.4.69 — 2026-09-20

- Fixed cDFT scheduling when an optional node-compatibility policy field is absent.
- Kept task-specific batch sizes out of the global agent policy while retaining the generic success-rate rule.
- Made partial-batch contract normalization idempotent across repeated result checks.
- Improved autonomous handling of framework defaults and pre-submission failures.

## v3.4.68 — 2026-09-20

- Made batch calculations continue from scientifically valid successful outputs instead of requiring every attempted item to finish.
- Added agent-owned batch review using task-specific criteria or a generic 60% default, with autonomous parameter repair, failed-item retry, or method fallback.
- Added item-level attempted, successful, and failed charge-assignment records.
- Added agent-owned sampling of up to three real outputs before downstream tasks continue.
- Improved autonomous recovery so parameter, input, and method failures are diagnosed and repaired without interrupting the user when the scientific conditions are already clear.

## v3.4.67 — 2026-09-20

- Kept workflow graphs visible while live state moves between planning, approved, and runtime views.
- Added reliable pormake output publishing for downstream scientific tools.
- Added `all`, `large`, and `small` structure-selection modes with successful-output counting.
- Improved recovery when a completed structure-generation job returns a different directory layout.
- Reduced avoidable delays between structure generation and dependent workflow tasks.

## v3.4.66 — 2026-09-20

- Improved workflow graph restoration and session-scoped display stability.
- Improved automatic resource fallback across compatible CPU partitions.
- Added automatic repair for valid split structure-output directories.

## v3.4.65 — 2026-09-19

- Established the public Sep-Pilot release baseline.
- Added the English feature-focused project page and noncommercial license.
- Published a privacy-sanitized source distribution under the sole authorship of user.
