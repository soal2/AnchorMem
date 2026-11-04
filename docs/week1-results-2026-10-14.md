# 第一周实验进展

## 10 月 18 日网络恢复与续跑

已验证通过当前 VPN 的执行器传输连续三次返回 200，并完成安全恢复及 **64 项相关回归**。原冻结协议不变：51 个连接失败建库任务及其 96 项问答已恢复，**93 个成功响应和 22 条评分均未重跑、内容未变**。1 个 MemoryBank/conv-41 读取超时任务无法确认服务端是否完成，保持隔离，不自动重发。

北京时间 **2026-10-18 10:28:51** 已从同一运行 `20261016T011008129863Z` 实际续跑，Python PID **7115**。10:29:20 核查进程存活，当前方法正在完整建库，已取得 5 个新成功响应；已评分仍为 **22/128**。日志切换为 [10 月 18 日续跑日志](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/task-executor-resume-2026-10-18.log)。[恢复说明及审计收据](/Users/eversse/Documents/科研/llm_mem/AnchorMem/docs/week1-network-recovery-2026-10-18.md)。以下 10 月 16 日状态均为历史快照，不代表当前进程状态。

**10:30:54 更新：进程正常，已新增 65 个成功请求，无新失败；当前方法已提交 1/46 个 Add 检查点，正在第 2 批。完整历史完成前不进行问答或评分，原有 22 条评分继续保留。**

## 当前独立任务运行（2026-10-16）

新执行器全量回归 **802 项通过**；14 份完整历史、650 个 Add 单元的抽取输入审计通过。当前方法固定每次最多两个 chunk，不截断历史；建库和每题各阶段都有 SQLite 检查点，失败只阻塞对应任务，取消所有首题全成功门槛。

北京时间 **09:10:05** 已实际启动运行 `20261016T011008129863Z`，Python PID **80744**，日志为 `artifacts/week1/task-executor-live-2026-10-16.log`。初始继承 RAG、Mem0 各两个完整历史及两题成绩，共 **4/128**；MemoryBank 因补齐作者连续检索的 FAISS 状态持久化而重建受影响历史，其旧两题及当前方法旧协议成绩保留在原运行，不混入本版统计。

本版目标为 **56 个完整建库任务、128 个逐题问答任务**，尚未完成完整 pilot。实时覆盖、失败和共同题集见[本版报告](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261016T011008129863Z/report.md)，[执行状态](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/active-run.json)、[运行日志](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/task-executor-live-2026-10-16.log)与系统进程共同确认是否正在运行。[执行器与恢复说明](/Users/eversse/Documents/科研/llm_mem/AnchorMem/docs/week1-task-executor-2026-10-16.md)。

**09:11:36 现场快照：进程存活，已完成 22/128，失败及依赖阻塞均为 0。** RAG、Mem0 各完成 LoCoMo 10 题和 LongMemEval 1 题，共各 11/32；MemoryBank 正在构建 `conv-41` 完整历史，当前方法尚待调度。新运行有效完成 4/56 个完整建库任务，另 1 个进行中。相比初始继承的 4 题，已新增真实问答评分 18 题；四系统共同完成题集仍为空，不据此比较四方法优劣。[快照收据](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261016T011008129863Z/task-executor-progress-check.json)。

## 历史运行记录（以下为旧协议状态）

2026-10-15 用户授权以 **MemoryBank（AAAI 2024 作者公开代码）替换 A-Mem**，并明确要求删除 A-Mem 结果。A-Mem 的索引、预测、答案、检索、建库及失败记录已删除，不另存结果归档；其他方法原始 JSONL 逐字保留。删除收据见[记录](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261014T085344Z/user-requested-amem-result-deletion-2026-10-15.json)。实际已发生调用的预算占用保留在同一账本中。

当前选定的三个 baseline 为 **RAG、Mem0-OSS (2025)、MemoryBank（作者公开代码）**；两个 benchmark 仍为原始 LoCoMo 与 LongMemEval-S。MemoryBank 两套首题的完整历史建库、检索、回答与评分均成功，基线运行于 2026-10-15 20:55:19 结束。

**2026-10-16 08:26 现场核查：实验已停止，当前没有后台实验进程。** 运行 `20261015T163458Z` 于北京时间 **01:04:08** 标为 `initial-validation-incomplete` 并退出，PID `67458` 已不存在，也未发现其他第一周执行进程。LongMemEval 首题已提交 **232/487 条消息、153 条事实、22 个批次**；随后抽取输出达到原定 **4096 token** 上限，返回 `finish_reason=length`，原始响应被截断，严格校验拒绝并停止该链路。未产生 LongMemEval 当前方法评分，剩余题单未自动扩展。

本版累计 **7/128 份评分**：三个 baseline 各两份、当前方法 LoCoMo 一份。LoCoMo 663 条消息、255 条事实、46 个批次的完整建库、检索、回答及官方评分已完成，单题 F1=0。全套 **749 项测试通过**属于离线验证，不表示完整 pilot 已完成。此前 00:57 的“运行中、LongMemEval 104 条消息”是当时现场快照，本次已更新实际结束状态。

[本次进程与落库检查](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T163458Z/repair-validation-checkpoint.json)、[结束状态](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T163458Z/runtime-status.json)、[原始失败](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T163458Z/failures.jsonl)、[运行日志](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/ours-global-source-range-validation-2026-10-16.log)。本次只核查状态并更新记录，未发起模型调用、未重试失败输出、未核查费用。

v12 将 `first..last` 明确规定为当前批次展示词 ID 的完整闭区间，全部中间单位均被显式选中；跨 chunk 分别保留独立的精确原文引文，不拼造跨消息字符串。不同范围之间不补入未选的实质文本，逆序、未知 ID、重复对和无法表达的 occurrence 继续严格拒绝；ISO 格式、日历和精度约束保持。原文坐标有效仍不等于事实获得完整语义支持。

本版继承三个作者源码 baseline 的 6 份评分，新增当前方法 LoCoMo 首题 1 份；LongMemEval 首题链路未完成，因此 driver 未进入其余冻结题单。旧版本当前方法索引、事实和评分不继承。[实时日志](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/ours-global-source-range-validation-2026-10-16.log)、[状态快照](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T163458Z/runtime-status.json)、[现场检查点](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T163458Z/repair-validation-checkpoint.json)、[冻结协议](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T163458Z/manifest.json)、[修复记录](/Users/eversse/Documents/科研/llm_mem/AnchorMem/docs/week1-source-span-repair-2026-10-15.md)。PID 和计数是带时间快照，最终结果以本版预测、结束状态和进程检查为准。

本版已通过旧捐车错绑位置回归。独立抽查快照为 00:47:57，前 10 条事实及已知用餐位置全部 7 条事实共 17 条，17 段引文的原文位置和注册源 SHA 均正确；发现 1 条引用未覆盖完整表述，以及 1 条把 Maria 对 John 说的话归于 John 的模型说话者错误。只作定性质量记录，不代表总体发生率，未据此修改冻结配置。[本版错绑位置复测](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T163458Z/known-source-regression.json)、[本版来源抽查](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T163458Z/source-support-spotcheck.json)。

v11 两套首题均在跨 chunk 范围处失败，于 00:28:56 结束。LoCoMo 提交 327 条消息、120 条事实、23 个批次后，一条联合多消息的引用被同 chunk 约束拒绝；LongMemEval 的跨 chunk 引用来自同一长消息被 512/64 窗口拆分。v12 通过新明示协议处理这类区间，并从完整历史重建，不把旧失败响应换规则补入数据库。[v11 原始失败](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T161854Z/failures.jsonl)、[结束状态](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T161854Z/runtime-status.json)。

v10 在提交 LoCoMo 248 条消息、104 条事实、18 个批次后，模型写出 `mentioned_at="2023-05-04T15:18:00Z."`，尾部句点被原有解析器严格拒绝。该轮未产生当前方法评分，已于 00:14:05 停止；v11 不继承它的当前方法索引。时间约束只排除格式不合法的字符串，不代替原有日历、精度及语义检查。[v10 原始失败](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T160510Z/failures.jsonl)、[结束状态](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T160510Z/runtime-status.json)。

v10 已在建库中重新通过 `conv-41/session_2/messages21-28` 的旧捐车错绑回归，相关事实来自用餐及跆拳道讨论。00:08:33 的固定开发位置抽查覆盖 14 条事实、14 段引文，原文字符位置和注册源 SHA 全部一致，未见身份互换；其中 1 条事实的引文未覆盖全部表述，另 1 条解释性措辞的支持需谨慎。只作定性记录，不作为总体质量比例，不据此改参数或筛选事实。[本版故障位置复测](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T160510Z/known-source-regression.json)、[本版来源抽查](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T160510Z/source-support-spotcheck.json)。

v9 运行 `20261015T155602Z` 的两套首题均因来源范围倒序而严格失败并退出。冻结源码回放确认：旧候选先列句段、再从消息开头列词片段，导致数字递增时原文位置回到开头。v10 用单层、按原位排列的候选消除了这一结构；不修复旧 raw、不降低校验、不补写旧失败结果。[原始失败](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T155602Z/failures.jsonl)、[离线故障回放](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T160510Z/source-unit-v9-failure-replay-2026-10-16.json)。

上一版 `20261015T150009Z` 于 23:36 完成当前方法 LoCoMo 首题：663 条消息、286 条事实，检索、回答及官方单题 F1=0；随后 LongMemEval 在 44 条消息后，模型枚举上千个引用编号并重复尾部编号，触及 4096 token 输出上限，23:40 停止。该次共有 7 份评分，但当前方法的 1 份只保留为 **v8 开发结果**，不继承到 v12，也不据此评价整体效果。[旧运行](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T150009Z/manifest.json)。

v8 的已知旧上下文错绑位置复测通过；15 条事实、30 段引文的定性抽查发现 6 条事实的选中引文未覆盖全部子句。这些记录仍是 v8 的开发证据，不能作为 v10 的验证结论。v10 在同一故障位置重新检查后才允许生成首题回答。[v8 检查收据](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T150009Z/known-source-regression.json)、[v8 抽查](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T150009Z/source-support-spotcheck.json)。

Jev 上游 429 的状态码透传已修复，只有 429 允许一次有记录、等待 60 秒、走原预算门禁的传输重试；模型内容错误不重试。费用按用户指示不逐笔核查，未知费用继续按完整上界占额。传输修复未改变科学请求参数。[传输恢复记录](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T150009Z/transport-recovery-20261015T153232Z/receipt.json)。

## 真实完成范围

本版从 `20261015T123818Z` 继承三个 baseline 已完成的 6 份评分；v8 当前方法的 1 份开发评分另行保留，不计入本版。2026-10-16 08:26 核查范围如下：

| 系统 | LoCoMo / 20 | LongMemEval / 12 | 状态 |
|---|---:|---:|---|
| RAG | 1 | 1 | 两份首题完整历史建库、回答与评分完成 |
| Mem0-OSS | 1 | 1 | 两份首题完整历史建库、回答与评分完成 |
| MemoryBank | 1 | 1 | 两套首题完整链路完成，LoCoMo 单题 F1=0，LongMemEval 单题 judge=1 |
| 当前方法 | 1 | 0 | LoCoMo 首题完整链路完成，单题 F1=0；LongMemEval 在 232/487 条消息后输出截断，已停止 |

旧运行已标为 `superseded-baseline-replacement`，不再继续 A-Mem，也不将其得分计入当前实验。[旧 manifest](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261014T085344Z/manifest.json)、[保留预测](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261014T085344Z/predictions.jsonl)。新 protocol 单独冻结并记录继承来源，不重复生成 RAG/Mem0 已有答案。完整目标仍为四系统各 32 题，共 128 次问答；v12 启动时两个 benchmark 的四系统共同完成题数均为 0；00:52 时 LoCoMo 四系统共同完成 1 题，LongMemEval 共同完成 0 题。表中运行状态为带时间的快照，最终范围以当前运行的逐题预测和结束状态为准。

之前后台于 2026-10-15 14:02:11 结束，而手动状态快照停留在 13:36 的“运行中”，已现场核查并纠正。后续以进程检查、driver 的结束状态和逐题结果共同确认进度，不将离线测试通过数量当作真实问答数量。

## 当前方法最初的独立阻碍

最初验证使用已批准的提示词追加段落，SHA256 为 `2df7ad7a1a05dffb3b80acf148a74c7c7654e76a774786a0c2f14b564c505c4c`。该段属于旧运行；后续用户指示“那先修复”“继续”，新运行已重新冻结输入与引用协议，不叠加旧段落。

- LoCoMo：模型仍将上一批的捐车句误标为当前 `new_chunks[0]` 的引文。该句不在当前批次中，严格来源校验拒绝。
- LongMemEval：首批请求实际返回 HTTP 200，第二段不同引文在对应 chunk 中只出现一次，模型却填写 `occurrence=1`，而合法值应为 0；严格校验拒绝，成功持久化 0/487 条消息。

原始输出和诊断保留在[失败记录](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261014T085344Z/failures.jsonl)。这两项是内容错误，不再同参重试。v12 只读当前批次原文，由起止片段 ID 显式选择完整原文范围，在代码中生成逐字引文及出现次序；不放松来源校验，不把旧失败输出补写成成功结果。更换 baseline 不能消除当前方法的这项独立问题。

## 源码与公平性

| 系统 | 固定源码 | 实际路径 |
|---|---|---|
| RAG | Mem0 evaluation `ec1d7a45d3d0a11894a16cd7d05d6ba0257141f6` | 作者 `RAGManager.create_chunks/search`，500 token、无重叠、top-k=20 |
| Mem0-OSS | `07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4` | 原 `Memory.add(infer=True)/search`，本地 Qdrant on_disk，graph 关闭 |
| MemoryBank | `cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9` | 作者 `summarize_memory.py` 与 `memory_retrieval/forget_memory.py` |

MemoryBank 的选型依据是 [Mem0 原论文 §3.3](https://arxiv.org/html/2504.19413v1#S3.SS3) 将它列入经典记忆对照；其[作者仓库](https://github.com/zhongwanjun/MemoryBank-SiliconFriend/tree/cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9)公开了摘要、FAISS 检索、遗忘及强化实现。直接执行这些代码，不依据论文另写算法。保留作者公开代码实际的 `exp(-t / 5 * S)` 公式和 summary 不强化的行为，不能声称该代码与论文全部数学行为一致。

MemoryBank 使用独立 Python 3.11 CPU 环境；只适配统一模型与远程向量接口，禁用作者传输层的隐藏重试和超限截断。保留作者摘要 prompt、max_tokens=400、top-k=6 和 chunk_size=200。使用题目日期（LoCoMo 无题目日期则使用最后历史会话日期）作为冻结时钟，遗忘 seed=42。四系统读取同样完整合法历史，原角色、日期、说话者及 LoCoMo 图片描述保留；QA、人工摘要、答案、证据标注均不进入 builder，模型答案不写回记忆。

统一生成/回答模型为 `openai/gpt-4o-mini`、temperature=0，向量为 `openai/text-embedding-3-small`、1024 维，reader 证据上限 16k token。LoCoMo 使用固定官方 F1（含词干化、多跳和开放域处理）；LongMemEval 使用官方分类/拒答提示及 `openai/gpt-4o-2024-08-06` judge，按作者 label 表达式解析并保存 raw judge。

## 数据和题单

- LoCoMo：commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`，原文件 SHA256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`。
- LongMemEval-S cleaned：HF revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`，SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`。
- 冻结题单 SHA256 `f87197293993ef41943ce70a45f93a12a6249599710d0039d4f7b5c920346191`：LoCoMo 20 题、LME 六类各两题共 12 题，seed=42，共 14 份完整历史。题单和开发曝光记录保持不变。

所有请求继续串行经过统一网关，总 API 上限仍为 20 美元。按用户指示不再逐笔核查费用；未知费用保留完整上界占额，记录为 null。第二周全量预算要待新四系统取得足够实测数据后估算。

## 替换后的验证

新增 MemoryBank 源码校验、完整历史转换、日期冻结、原生 FAISS/遗忘/强化、持久化重开、隐藏重试阻断与数据隔离检查。两套完整历史均已通过本地 HTTP mock（只模拟模型响应，作者算法真实执行），并对全部 32 道冻结题完成合法历史与日期转换检查。

全套 **615 项测试通过**，83.50 秒，保留一条既有 Starlette/httpx 弃用提示。见[测试输出](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T123818Z/test-results.txt)和[验证记录](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261015T123818Z/verification.json)。新运行复用已核对来源/配置/reader 一致的 RAG 与 Mem0 四份评分，不继承 A-Mem 结果，不重复当前方法的已知同参内容失败。driver 在结束时更新 manifest、runtime-status 与 active-run；替换 baseline 的那次运行未自动启动剩余完整 pilot。

## 首批真实验证结果：2026-10-15 21:47 核查

当时核查：MemoryBank 于北京时间 20:55:19 完成限定的两套首题验证并退出，没有对应实验进程。LoCoMo 663 条消息完整建库耗时 400.64 秒；LongMemEval 487 条消息完整建库耗时 605.38 秒。此次 MemoryBank 没有失败记录，最后一次模型响应为 HTTP 200。

累计评分为 RAG 2 份、Mem0 2 份、MemoryBank 2 份、当前方法 0 份。单题分数只用于链路验证，不作为整体方法效果结论。剩余 122 次评分未启动，两个 benchmark 的四系统共同完成题数仍为 0。停止原因是既定的首批验证边界及当前方法已知的严格来源校验失败，并非本次 MemoryBank 异常。下一步先解决当前方法抽取输出协议，再从同样完整历史重建并继续 pilot。
