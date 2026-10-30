<div align="center">

# SEP-PILOT

### 面向分子模拟与材料研究的智能工作台

从研究问题出发，规划计算、跟踪执行并整理可追溯的结果。

**[功能](#功能) · [智能体协作](#智能体协作) · [快速开始](#快速开始) · [科学软件配置](#科学软件配置) · [结果与验证](#结果与验证) · [License](#license)**

</div>

Sep-Pilot 将对话、任务图和计算工具连接到一个工作区。研究者可以描述目标、提供结构与参数，查看计划并跟踪作业；系统根据依赖关系调度专业智能体，整理结果、图件和报告。具体能运行哪些计算取决于部署环境中安装的软件、可用资源和配置。

![SEP-PILOT 智能体协作架构](docs/architecture-overview.png)

上图说明主要角色和数据流；一次对话不一定使用全部角色或提交集群作业。

## 功能

- **任务规划与可视化**：将研究目标组织为带依赖的 DAG，在界面中查看节点、状态与结果。
- **科学计算**：对接结构和 CIF 准备、孔结构分析、电荷、GCMC、Henry、分子动力学、String/TST、经典密度泛函理论（cDFT）及结果分析等工具。
- **数据与机器学习**：支持特征准备、模型训练、预测及主动学习相关流程；实际能力以可用数据和已安装依赖为准。
- **集群与恢复**：按配置对接 Slurm，跟踪作业、尝试、资源和输出；独立分支可并行，恢复时检查已有结果与当前任务的关系。
- **图件、报告与文献**：基于任务结果制作图件和报告；可查找并记录文献来源。未请求文献比较的普通计算报告不需要强行添加比较。
- **任务隔离**：按会话组织输入、工作流、证据和产物。生成代码可通过沙盒执行，产物可供后续节点使用。

## 智能体协作

主智能体根据目标提出计划并与用户协商。工作流运行时保存节点依赖、计划版本、资源锁、尝试和结果；具备依赖条件的节点由对应专业角色执行。监督和恢复角色读取状态与证据，协助诊断故障；结果分析角色将已完成输出用于图件和报告。用户修改条件时，系统需要区分仍可复用的结果和需要重新执行的部分。


调用者可根据任务选择已有工具；数据处理和绘图也可在 Python 沙盒中使用当前任务允许的文件。沙盒执行与普通 shell、科学作业提交是不同的能力，使用时以工具暴露的接口和当前环境为准。联网、安装第三方依赖以及不同任务之间的持久环境取决于部署配置与隔离环境，**不要假定所有安装与联网场景均已可用**。

计算结束、数值收敛和物理验证是不同的判断。程序正常退出并不自动证明结果收敛；缺少收敛依据时应保留结果并标注未验证。报告可以根据用户要求包含图、参考文献或 PDF；需要引用时可使用任务中取得的题名、DOI 或链接，不能虚构文献或无依据的比较。图像可打开也不等于科学结论已验证。

## 快速开始

以下示例启动本地开发环境。模型服务与科学软件需要自行配置；不要将密钥、个人账号、运行结果或本机配置提交到仓库。

### 后端

```bash
git clone https://github.com/Yong-Q/Sep-Pilot.git
cd Sep-Pilot
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
# 在 .env 中填写模型服务配置；不要提交 .env
set -a
source .env
set +a
uvicorn api:app --host 127.0.0.1 --port 8000
```

也可在部署环境使用 `env/settings.json` 配置模型服务；该文件包含敏感信息时必须保存在本地。支持与 Anthropic 客户端兼容的服务，实际可用性、模型名和凭据由部署者提供。SDK 的安装并不自动为所有工作流提供工具执行或联网权限。

### 前端

另开一个终端：

```bash
cd frontend
npm ci
PORT=5001 npm start
```

打开 `http://localhost:5001`。开发服务器将 API 请求代理到 `http://localhost:8000`；如果你使用其他后端地址，需同步调整前端代理配置。也可使用 React 开发服务器默认端口，但应确保后端允许该来源。

### 命令行

```bash
python -m agents --list-agents
python -m agents -i
```

## 科学软件配置

发行包提供接口和工作流代码；外部科学软件需由用户或部署者安装。配置从本地 `config/runtime.json` 读取，也可用 `BIMEM_RUNTIME_CONFIG` 指向配置文件。下面的路径是示例，应替换为各自环境中的**绝对路径**：

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

**MD 使用用户安装的 LAMMPS 和 MPI**：在上述配置中设置可执行文件位置即可，项目不随发行包分发 LAMMPS。MD 的 Python 环境还需要 ASE；如需额外动态库，可配置 `software.lammps_library_path`。提交到集群时，计算节点必须能够访问所选程序、输入和工作目录。其他工具也应按对应部署配置安装并检查可用性。

Slurm 分区、节点、资源预算和软件路径具有站点差异。请在自己的环境配置它们；不要将个人集群策略当作通用默认值。详情见 [运行时、恢复与软件配置](docs/runtime-recovery.md)。

## 结果与验证

工作流中记录的作业状态、工具回执和文件可以帮助追溯来源。`prefinish` 表示工具已返回、仍需后续结果检查；失败或待处理节点不应被当作最终完成。报告和图件的具体内容、单位、来源及未验证项仍需结合研究问题检查；短测试不能替代收敛或物理验证。

仓库内可运行的检查：

```bash
python -m pytest tests -q
cd frontend
CI=true npm test -- --watch=false --runInBand
npm run build
```

部分集成测试需要外部科学程序或数据，依赖不可用时可能跳过。自动化测试覆盖代码合同和模拟场景，不代表所有本地安装、集群条件或科学结论都已验证。具体变更及适用范围见 [CHANGELOG](CHANGELOG.md) 和 [运行时文档](docs/runtime-recovery.md)。

## License

Copyright © 2026 **Yong-Q**.

Sep-Pilot is source-available under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**. Personal research, study, experimentation, education, and qualifying noncommercial institutional use are permitted under its terms. **Commercial use is not permitted without a separate license from the author.**

## Author

Created and maintained solely by **[Yong-Q](https://github.com/Yong-Q)**.
