# Changelog

This file records user-visible changes to Sep-Pilot. Implementation details and private deployment information are maintained separately.

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
