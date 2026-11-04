# 第一周独立任务执行器

日期：2026-10-16，Asia/Shanghai。

## 执行协议

固定方法为 RAG、Mem0-OSS、MemoryBank 和当前方法。题单保持原始 LoCoMo 20 题与 LongMemEval-S 12 题，14 份完整历史对应 56 个建库任务和 128 个问答任务。三个 baseline 直接执行原先固定提交的作者源码；公共 Add/Search、模型、Embedding 1024 维、16k 证据预算及共享 18/20 美元门禁保持。

删除首题全成功门槛。按历史首次出现顺序、固定方法顺序串行运行，历史内按冻结题序问答。建库失败只阻塞对应方法、历史的题目；检索、回答、judge 或评分失败只结束该题，其他任务继续。内容失败不自动重试，不将空缺结果计零，不用不完整历史评分。

当前方法每次抽取最多两个原有 512/64 chunk，按完整原消息分组、源序遍历。后续批次附该消息首 chunk 的有界日期、说话者和指代参考；参考没有引用单位，事实只能引用当前批次。保留所有严格范围、逐字引文、日期、父子结构与输出完成检查，不裁剪历史或修补失败输出。

## 检查点与恢复

`tasks.sqlite3` 是任务和阶段结果的权威记录。建库保存固定 20 消息 Add 单元进度；RAG、MemoryBank 保留作者整体建库流程。问答分别保存 retrieval、answer、judge、score，评分有唯一键且依赖完整建库收据。

每个可变操作先复制已提交索引，在工作副本中运行并关闭存储句柄，发布新版本后通过 SQLite 事务更新索引指针及阶段结果。未提交副本不能被后续题目使用。MemoryBank 按用户选择保留作者连续强化；成功检索与强化状态共同提交，失败检索丢弃工作副本。成功检索后的回答、评分失败不会触发第二次检索。

Gateway 使用共享 `api-call-checkpoints.sqlite3`，按冻结运行、任务、单元、阶段及调用序号保存请求哈希、响应和预算事件。成功响应先持久化再交给调用者，重启重放已有响应；请求变化和未确认响应仅阻塞原任务，不能盲目重发。缓存读取先于预算停止检查，已有答案和 judge 可继续完成本地评分。新任务只读取响应提供的 usage，不另查账户或 generation 费用。

同一运行只有一个执行进程。恢复要求源码、提示词、配置、数据、依赖及题单与冻结快照严格一致，协议改变必须生成新冻结运行；旧运行不支持例外修改。原始开发运行保留作历史记录，不再使用旧首题 driver。

## 继承范围

继承母本为 `20261016T004000Z`。RAG、Mem0 的两个完整历史、四题评分通过作者实现、全文、Add 收据、检索与官方评分的一致性检查后导入。

审查发现 MemoryBank 作者检索会修改 FAISS docstore，原适配器只持久化了 memory.json。新适配器在成功原生检索后调用已有 FAISS `save_local`，使同一作者实例的连续状态在关闭、重启后保留；作者源码未修改。由于旧索引缺少此前检索的 docstore 状态，MemoryBank 旧两题不导入新协议，相关完整历史重新建库。旧两题及当前方法 v12 的单题结果留在原运行，不能混入新协议覆盖率。

## 离线验证与运行记录

全历史审计：14 份历史、650 个 Add 单元、6959 个抽取请求，原文字符区间完整覆盖，所有 chunk 按源序处理一次，每次最多两个 chunk。最大请求连同 4096 输出和 1024 token 余量为 14858 token，低于 128k。审计没有真实模型调用。

审计文件：`artifacts/week1/task-executor-history-audit-2026-10-16.json`；可重跑：

```bash
.venv/bin/python scripts/week1_history_audit.py --output artifacts/week1/task-executor-history-audit-2026-10-16.json
.venv/bin/python -m src.week1_runner preflight
```

最终完整回归 **802 项通过、1 条既有依赖弃用警告**，记录在 `artifacts/week1/task-executor-tests-final-2026-10-16.txt`。测试涵盖跨方法及跨历史失败隔离、每阶段恢复、调用重放、不确定调用、索引提交边界、MemoryBank 连续强化与回滚、用户中断不被清理异常覆盖、预算门禁、完整历史禁评以及公平交集汇总。

新冻结与续跑：

```bash
.venv/bin/python -m src.week1_runner run --budget-usd 20 --inherit-run artifacts/week1/20261016T004000Z
.venv/bin/python -m src.week1_runner run --budget-usd 20 --run-dir artifacts/week1/20261016T011008129863Z
.venv/bin/python -m src.week1_runner report --run-dir artifacts/week1/20261016T011008129863Z
```

运行入口写入 `artifacts/week1/active-run.json` 和运行目录中的 `runtime-status.json`，包括 PID、任务、阶段、Add 进度及更新时间。进程状态需结合系统 PID 核对；这些文件是带时间的快照。实时报告包含每个方法的完成率、完整建库数、失败与依赖阻塞任务、未执行题目及四方法共同完成题集。只有四方法各完成 32 题才标记完整 pilot 完成。

真实续跑已于北京时间 **2026-10-16 09:10:05** 启动：运行 `20261016T011008129863Z`，Python PID `80744`，防休眠 PID `80745`，日志为 `artifacts/week1/task-executor-live-2026-10-16.log`，启动收据为 `artifacts/week1/task-executor-launch-2026-10-16.json`。初始导入 4 个完整建库任务和 4 条评分；52 个建库任务、124 个问答任务尚待执行。进程存活期间不重复启动，停止后使用同一 `--run-dir` 恢复。

09:11:36 核查：Python 进程存活，任务正在 MemoryBank/LoCoMo/conv-41 的完整建库阶段；有效评分已到 22/128（RAG、Mem0 各 11/32），无失败或依赖阻塞，另两方法在本版尚无评分。本次已新增真实评分 18 条；快照保存在本版 `task-executor-progress-check.json`，后续以实时报告为准。
