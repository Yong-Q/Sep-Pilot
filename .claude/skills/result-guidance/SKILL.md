# 结果分析、绘图与报告：按需路由

普通科学问答/作业状态更新不载入整套Nature规则，不触发写稿或后端选择门。
实际要画图或写正式报告时，使用read_result_guidance按需读取项目内文件；本文件不是图形模板。

- 默认绘图后端为用户明确选择的Python。用户显式选择R时尊重选择；当前Python沙盒工具不冒充R支持，不静默转Python。其他需要选择而尚未选择的图稿，按nature-figure后端规则处理。
- 图稿：先读nature-figure/SKILL.md和manifest.yaml；仅载入其always_load、所选backend片段、需要的QA/参考。先说明结论/证据/单位/布局与局限，由LLM按任务设计图形并编写脚本，不固定为条图/散点或现有模板。
- 正式报告：先读nature-writing/SKILL.md和manifest.yaml，再按paper_type/section/language/journal载入少量所需片段，包括manifest列出的_shared依赖。普通任务结果简报无需套整篇论文结构。
- 使用当前真实结果表，缺值/单位未知/不同方法/未收敛明确标注。没有真实来源不能生成假图；合成数据只可用于明确标记的软件测试。
- plot_result_code执行Python脚本的OS隔离，不是LLM本身的运行容器。LLM通过受控read_result_guidance读取提示，可用guidance_files把所选提示只读放入/inputs供脚本检查；代码仍由LLM自由设计。
- 独立绘图DAG节点依赖已产生的数据表，声明新output_dir和PNG/manifest输出；报告节点依赖图节点。检查源表/运行日志/PNG完整性，修改脚本后用新目录重试，保留旧失败证据。
- 渲染成功不证明科学结论或视觉质量已通过。当前工具原生验证PNG格式/尺寸与来源SHA；没有图像识别证据时，不声称模型已经看图或视觉审核通过。
- 沙盒不可用时明确报告，不得退回普通run_bash执行LLM代码。脚本可复用；环境未完整锁定时旧图默认不缓存。

参考包仅约束证据与表达，不限制用户的科学图形选择。未包含示例大图、私有R模板或MatOS源码。文件来源/哈希与许可边界见result-guidance-manifest.json。
