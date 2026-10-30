# Runtime verification and recovery

The runtime records submission, scheduler completion, artifact validation and scientific acceptance separately. A job receipt or a nonempty model answer does not complete a research workflow.

## Read-only requests

Status queries, log inspection and schema discovery can use scoped read-only tools directly. They do not require a research DAG. Explicit restrictions on running or submitting calculations remain effective even when the question mentions computational methods. Preparing inputs without submission remains a separate `prepare_only` mode.

The lead agent can use `find_cif` before planning a calculation. It searches the current session's uploads, the project `cifs` directory and optional `paths.cif_dir`; it does not import a private legacy helper. Ambiguous charged or expanded variants require a concrete input choice, and incomplete searches cannot establish uniqueness.

## A failed calculation

1. `diagnose_job` reads the owned job's terminal receipt and actual logs. Durable failure records remain available after the scheduler removes its accounting entry. Both stdout and stderr can contain the failure evidence.
2. The responsible agent diagnoses and repairs the cause. `inspect_runtime` checks configured installations and adapter syntax, returning a source/configuration fingerprint. Its scope is local configuration: it does not establish compute-node availability, a successful calculation or scientific validity.
3. `prepare_retry` requires separate actual diagnostic and later verification call IDs. The retry must match one approved scientific node and its original task identity. A verified executor repair can retain the same scientific inputs; a prose claim cannot authorize a retry.
4. A single-use permit requeues the failed node while preserving the prior attempt, completed branches and original submission logs. User pauses, active leases, live or uncertain jobs and exhausted dispatch budgets prevent this transition.
5. The worker checks the executor fingerprint again before dispatch. Original reviewed CPU/RAM settings can accompany the approved scientific arguments. A changed allocation requires resource review and cannot replace the scientific contract.

Old configuration-preflight proofs are archived when a reviewed real-job retry takes over. They cannot be reused as proof that a submitted calculation never executed. Lifecycle messages from superseded attempts remain in the audit but do not direct the current model turn.

## User-installed software

The release contains adapters, not scientific engines or account credentials. Installation checks use the same site configuration as execution, including `BIMEM_RUNTIME_CONFIG` when selected. Relevant software keys include:

| Adapter | Site configuration |
|---|---|
| Structure generation | PORMAKE interpreter in `env/tools.json`; `software.pormake_database_dir` |
| Framework charges | `software.pacmof_python`, or `software.pacman_python` and `software.pacman_dir` |
| Classical cDFT | `software.cdft_python`, `software.cdft_executable`, `software.cdft_init_dir` |
| Framework MD | `software.lammps_executable`, `software.mpi_executable`, `software.lammps_interface_executable`; optional library path |
| String/TST | `software.string_tst_executable`, `software.string_tst_generate_script` |
| External potential | `software.tustrast_executable`, `software.tustrast_generate_script`, `software.tustrast_input_param` |

MD retries use a fresh work directory when earlier outputs exist. The MD wrapper performs framework minimization or relaxation; it does not establish guest diffusion or produce an unwrapped guest trajectory.

For cDFT, set `software.cdft_init_dir` to the existing installation directory containing `data_ff_UFF`, `Molecule_FF/data_ff_Coarsening_gas`, and `Molecule_FF/UFF/<gas>.def` for each requested gas. These scientific assets are not included in the public repository. The SDK uses its public input/submission/collection adapters with those configured data paths; it does not import Python adapters from the external installation or change force-field parameters. `inspect_runtime` checks the selected assets and records their execution fingerprint before recovery. This check does not generate inputs, submit a job, or establish scientific success.

## cDFT screening analysis

`analyze_cdft_screening` reads canonical `MOF` and `<gas>_density_mol_L` columns, aligns material identities, ranks a declared pair's arithmetic density ratio and stages an arbitrary TOP N with a source hash manifest. It rejects duplicate identities, incomplete gas sets, invalid densities, zero denominators, missing exact CIFs and existing output paths.

A CSV alone does not establish single-component versus mixture mode, temperature or pressure. The returned manifest explicitly leaves those checks unverified. The scientific owner must verify the source jobs before interpreting the ratio as a single-component uptake ratio. The tool does not calculate IAST or mixture selectivity.

## Delivery acceptance

Mocked scheduler tests establish state transitions and guard behavior. Installation checks establish local configuration. Delivery also requires real UI actions, actual agent-driven dispatch, terminal job tracking, validated scientific outputs, downstream continuation and a user-visible final artifact. Record latency and model/tool round trips separately. Report incomplete scientific workflows as incomplete even when regressions pass.
