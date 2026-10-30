<div align="center">

# SEP-PILOT

### An intelligent workspace for molecular simulation and materials research

Plan calculations, follow their execution, and organize traceable results from a research question.

**[Features](#features) · [Agent collaboration](#agent-collaboration) · [Quick start](#quick-start) · [Scientific software configuration](#scientific-software-configuration) · [Results and validation](#results-and-validation) · [License](#license)**

</div>

Sep-Pilot brings conversations, task graphs, and computational tools together in one workspace. Researchers can describe a goal, provide structures and parameters, review a plan, and follow jobs. The system coordinates specialist agents according to task dependencies and organizes results, figures, and reports. Available calculations depend on the installed software, resources, and configuration of the deployment environment.

![SEP-PILOT agent collaboration architecture](docs/architecture-overview.png)

The diagram shows the principal roles and data flow. A conversation does not necessarily use every role or submit a cluster job.

## Features

- **Planning and visualization**: organize a research objective into a dependency graph and inspect nodes, status, and results in the interface.
- **Scientific workflows**: connect tools for structure and CIF preparation, pore analysis, charge assignment, GCMC, Henry calculations, molecular dynamics, String/TST, classical density functional theory (cDFT), and result analysis.
- **Data and machine learning**: support feature preparation, training, prediction, and active-learning workflows, subject to available data and installed dependencies.
- **Cluster execution and recovery**: connect to Slurm through deployment configuration and track jobs, attempts, resources, and outputs. Independent branches can run concurrently; recovery checks how existing results relate to the current task.
- **Figures, reports, and literature**: produce figures and reports from task results and find and record literature sources. A routine calculation report does not need an unsolicited literature comparison.
- **Task isolation**: organize inputs, workflows, evidence, and artifacts by session. Generated code can run in a sandbox, and its outputs can be used by downstream nodes.

## Agent collaboration

The lead agent proposes a plan based on the objective and coordinates with the user. The workflow runtime records node dependencies, plan versions, resource locks, attempts, and results; specialist roles execute nodes when their dependencies are ready. Monitoring and recovery roles use status and evidence to help diagnose failures, while analysis roles use completed outputs to create figures and reports. When a user changes the conditions, the system needs to distinguish results that can still be reused from work that requires another execution.

Callers can choose existing tools for a task; data processing and plotting can also use files permitted for that task in a Python sandbox. Sandbox execution, ordinary shell access, and scientific job submission are separate capabilities: use the interfaces and environment available in the deployment. Network access, installation of third-party dependencies, and persistent environments across tasks depend on deployment configuration and isolation. **Do not assume every installation or network scenario is available.**

Completion, numerical convergence, and physical validation are distinct. A successful program exit does not by itself prove convergence; retain the result and mark convergence as unverified when evidence is missing. Reports can include figures, references, or PDFs as requested. References can use titles, DOIs, or links obtained during the task; do not invent citations or unsupported comparisons. An image that opens successfully does not by itself validate a scientific conclusion.

## Quick start

The following starts a local development environment. Configure the model service and scientific software for your own deployment. Do not commit keys, personal accounts, run results, or local configuration to the repository.

### Backend

```bash
git clone https://github.com/Yong-Q/Sep-Pilot.git
cd Sep-Pilot
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
# Set your model service configuration in .env; do not commit it
set -a
source .env
set +a
uvicorn api:app --host 127.0.0.1 --port 8000
```

A deployment can also use `env/settings.json` for model service configuration; keep it local when it contains sensitive information. Services compatible with the Anthropic client are supported, subject to the provider, model name, and credentials supplied by the deployer. Installing an SDK alone does not grant every workflow tool-execution or network access.

### Frontend

In a second terminal:

```bash
cd frontend
npm ci
PORT=5001 npm start
```

Open `http://localhost:5001`. The development server proxies API requests to `http://localhost:8000`; adjust the frontend proxy configuration if the backend uses another address. The React development server's default port can also be used, provided the backend allows that origin.

### Command line

```bash
python -m agents --list-agents
python -m agents -i
```

## Scientific software configuration

The distribution provides interfaces and workflow code. Users or deployers install the external scientific software. Configuration is read from local `config/runtime.json`, or from the file named by `BIMEM_RUNTIME_CONFIG`. Replace these example paths with the **absolute paths** for your environment:

```json
{
  "paths": {"agent_env_python": "/opt/science-env/bin/python"},
  "software": {
    "lammps_executable": "/opt/lammps/bin/lmp_mpi",
    "mpi_executable": "/opt/mpi/bin/mpirun",
    "lammps_interface_executable": "/opt/science-env/bin/lammps-interface",
    "string_tst_executable": "/opt/string/bin/GPU_string_polyatmoic",
    "string_tst_generate_script": "/opt/string/generate_string_inputs.py"
  }
}
```

**MD uses your installed LAMMPS and MPI**: set their executable locations in this configuration; LAMMPS is not bundled with the project. The MD Python environment also needs ASE. If additional dynamic libraries are required, set `software.lammps_library_path`. For cluster submissions, compute nodes must be able to access the selected programs, inputs, and working directory. Install and check other tools according to their deployment configuration.

Slurm partitions, nodes, resource budgets, and software paths vary by site. Configure them for your environment rather than treating an individual cluster's policy as a universal default. See [runtime configuration and recovery](docs/runtime-recovery.md).

## Results and validation

Workflow job states, tool receipts, and files help trace result provenance. `prefinish` means a tool has returned and the result still needs subsequent checks; a failed or pending node should not be treated as complete. Assess report and figure contents, units, sources, and unverified items against the research question. A short test does not replace convergence or physical validation.

Repository checks can be run with:

```bash
python -m pytest tests -q
cd frontend
CI=true npm test -- --watch=false --runInBand
npm run build
```

Some integration tests require external scientific programs or data and may be skipped if dependencies are unavailable. Automated tests cover code contracts and simulated cases; they do not validate every local installation, cluster condition, or scientific conclusion. See the [changelog](CHANGELOG.md) and [runtime documentation](docs/runtime-recovery.md) for changes and their scope.

## License

Copyright © 2026 **Yong-Q**.

Sep-Pilot is source-available under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**. Personal research, study, experimentation, education, and qualifying noncommercial institutional use are permitted under its terms. **Commercial use is not permitted without a separate license from the author.**

## Author

Created and maintained solely by **[Yong-Q](https://github.com/Yong-Q)**.
