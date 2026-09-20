<div align="center">

# SEP-PILOT

### AI research copilot for molecular simulation

From a scientific question to an executable workflow, live progress, and traceable results — in one workspace.

[![Release](https://img.shields.io/badge/release-v3.4.76-6C63FF?style=for-the-badge)](https://github.com/Yong-Q/Sep-Pilot/releases/tag/v3.4.76)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Research](https://img.shields.io/badge/use-research_only-00A98F?style=for-the-badge)](#license)
[![Tests](https://img.shields.io/badge/tests-529_passed-2EA44F?style=for-the-badge)](#quality)

**[Features](#features) · [Scientific workflows](#scientific-workflows) · [Changelog](CHANGELOG.md) · [Quick start](#quick-start) · [License](#license)**

</div>

---

Sep-Pilot helps researchers plan, launch, follow, and review computational materials workflows through a conversational interface. Describe the research objective and the system turns it into a visible task graph, coordinates the required scientific programs, and keeps every result connected to its source.

## Features

| | Feature | Experience |
|---|---|---|
| 🧭 | **Research planning** | Turn a natural-language objective into a clear, reviewable sequence of scientific tasks. |
| 🕸️ | **Visual workflow** | Follow dependencies, parallel branches, parameters, progress, and results in an interactive graph. |
| 🧪 | **Specialist agents** | Coordinate agents for structures, force fields, adsorption, molecular simulation, and data analysis. |
| 🖥️ | **HPC execution** | Launch and follow long-running calculations on a configured research cluster. |
| ♻️ | **Task recovery** | Continue interrupted work and handle recoverable failures without losing completed results. |
| 📝 | **Live reports** | Read evidence-backed report sections as workflow branches finish, then receive one updated final report. |
| 📦 | **Result handoff** | Feed structures, tables, trajectories, and other outputs into the next scientific task. |
| 🔎 | **Evidence tracking** | Keep calculations, files, scheduler receipts, and conclusions connected for later review. |
| 👥 | **Session isolation** | Maintain separate goals, files, workflows, and history for each user session. |

## Scientific workflows

Sep-Pilot can coordinate workflows that include:

- MOF structure generation and CIF preparation
- pore geometry and accessible-volume analysis
- framework and guest charge assignment
- force-field discovery and inspection
- GCMC adsorption calculations and isotherms
- molecular dynamics preparation and execution
- diffusion and trajectory analysis
- adsorption-heat and cDFT workflows
- batch material screening and result comparison
- evidence-grounded scientific reports

Scientific programs are configured for each deployment. Sep-Pilot checks that required tools are available before starting a workflow.

## One workspace, full visibility

The web interface combines the research conversation with a live workflow view. Researchers can:

- review the proposed task graph before execution;
- inspect the inputs and outputs of every task;
- see queued, running, completed, paused, and failed work;
- follow cluster jobs without leaving the session;
- request a change while preserving completed results;
- return after an interruption and continue the same workflow;
- receive the final answer together with its calculation evidence.

## Quick start

### Backend

```bash
git clone https://github.com/Yong-Q/Sep-Pilot.git
cd Sep-Pilot

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

cp .env.example .env
# Add your model-provider settings to .env

set -a
source .env
set +a
uvicorn api:app --host 127.0.0.1 --port 8000
```

### Web interface

```bash
cd frontend
npm ci
npm start
```

Open `http://localhost:3000` and create a research session.

### Command line

```bash
python -m agents --list-agents
python -m agents -i
```

## Quality

Release `v3.4.76` passed 172 workflow backend checks, 12 frontend checks, production build validation, and a sanitized-export privacy scan.

## License

Copyright © 2026 **Yong-Q**.

Sep-Pilot is source-available under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**. Personal research, study, experimentation, education, and qualifying noncommercial institutional use are permitted under its terms. **Commercial use is not permitted without a separate license from the author.**

## Author

Created and maintained solely by **[Yong-Q](https://github.com/Yong-Q)**.

---

<div align="center">

**Ask · Plan · Run · Review**

</div>
