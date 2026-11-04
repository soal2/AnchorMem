# AnchorMem 长期记忆实验计划

本计划固定为 **3 个 baseline、2 个主流 benchmark、当前方法**。目标是验证当前记忆组织、增量归纳和原文回溯机制的有效性，形成可用于 EI 收录会议投稿的实验材料。预计由一名研究者用约四周完成；实际工期与费用由第一周的小样本建库结果校准。录用取决于方法贡献与实验结果，不能由 baseline 数量或某个分数保证。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-14
- Verification Status: UNVERIFIED
- Version Label: experiment_plan_v2_memorybank
- Repro Lock: null

原计划已进入 pilot。2026-10-15 用户授权以 MemoryBank 替换 A-Mem，并删除 A-Mem 实验结果；新协议单独冻结，不将旧方法成绩转给新方法。

## 1. 固定研究问题

**核心问题：在相同生成模型、向量模型和回答证据预算下，当前方法能否更好地保存并检索跨会话事实，特别是信息更新、时间推理和多跳问题需要的证据？**

| 假设 | 实验依据 | 主要观察量 |
|---|---|---|
| H1：增量归纳改善跨会话信息更新 | 完整方法与移除归纳版本比较 | LongMemEval 的 knowledge-update、multi-session 分项 |
| H2：分类约束减少不相关事实的错误归并 | 取消归纳候选的同 topic 限制，保留其他流程 | 分项效果、错误归并案例及来源证据 |
| H3：命中记忆后回溯原文有助于可靠作答 | 同一冻结索引和命中条目，分别提供原文与提炼文本 | 两套 QA 指标、事实遗漏与失真案例 |
| H4：收益来自方法设计，而不只是额外的 Jev 模型 | Jev 关系判断替换为统一生成模型 | 同模型条件下的效果与成本 |

论文优先围绕 H1、H3 和 H4 建立论证，H2 用于解释机制。六类 topic 或目录层级本身不直接作为已证明的新颖性。相关工作仍需解释与 OpenViking 等目录记忆设计的区别，但不增加主实验 baseline。

## 2. 只复现三个 baseline

| Baseline | 选择理由 | 实施边界 |
|---|---|---|
| **标准向量 RAG** | 长期记忆论文常见的基础对照；检验结构化记忆是否优于直接检索历史 | 对完整合法历史分块、向量召回、统一模型作答；不做事实提炼或记忆更新 |
| **Mem0，2025 论文同期 OSS** | 代表事实抽取、相似记忆检索及记忆更新机制 | 固定历史 OSS，实现 ADD / UPDATE / DELETE / NONE；不采用 2026 改版算法或 Mem0g |
| **MemoryBank，AAAI 2024 作者公开代码** | 经典长期记忆方法，代表会话摘要、用户画像、遗忘和召回强化 | 直接调用作者摘要与 FAISS 记忆检索/遗忘模块，保留公开实现的实际行为 |

选择依据是论文已有对照：Mem0 的 2025 论文 §3.3 将 MemoryBank 列为 established memory approaches，并使用 RAG 对照。这一组合覆盖原文检索、事实更新、摘要与遗忘三类机制，使用较早公开方法。[Mem0 原论文 §3.3](https://arxiv.org/html/2504.19413v1#S3.SS3)、[MemoryBank 论文](https://arxiv.org/abs/2305.10250)、[官方源码](https://github.com/zhongwanjun/MemoryBank-SiliconFriend)。

2026-10-15，用户明确提出“实在不行不要用 A-Mem 了，换一个 baseline”，随后要求“把 A-Mem 的结果删了吧”。据此替换 baseline 并删除其结果；两套 benchmark、冻结题单、模型、reader 和评分保持原设置。替换依据是复现执行成本和稳定性，不依据某道 QA 分数。A-Mem 不进入后续主表，也不继续消耗调用。

**版本与复现口径：**

- Mem0 固定 `v0.1.94`，commit `07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4`。该版本在 2025-04 发布，公开实现具备抽取与更新流程。[发行版本](https://github.com/mem0ai/mem0/releases/tag/v0.1.94)、[固定源码](https://github.com/mem0ai/mem0/blob/07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4/mem0/memory/main.py)。
- Mem0 原论文的作者评测调用托管 API，不能通过 OSS commit 冻结当年服务端实现。同期 OSS 也未完整公开论文描述的周期会话摘要路径。因此主表准确命名为 **Mem0-OSS (2025)**，保留公开实现，不自行补造摘要模块，不宣称复现作者原始分数；论文方法与 OSS 的差异写入复现说明。[作者评测脚本](https://github.com/mem0ai/mem0/blob/ec1d7a45d3d0a11894a16cd7d05d6ba0257141f6/evaluation/src/memzero/add.py)。
- MemoryBank 固定 `zhongwanjun/MemoryBank-SiliconFriend` commit `cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9`。直接调用 `memory_bank/summarize_memory.py` 与 `memory_bank/memory_retrieval/forget_memory.py`；不根据论文重写摘要、检索、遗忘或强化。原始文件保持不变，仅适配统一模型接口和合法历史数据。[固定作者源码](https://github.com/zhongwanjun/MemoryBank-SiliconFriend/tree/cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9)。
- 该 MemoryBank 公开代码的遗忘曲线实际为 `exp(-t / 5 * S)`，其 summary 强化分支也与原始对话不同。保留作者代码行为，在复现说明中披露，不自行按论文公式修正并声称完全一致。作者摘要客户端内置的重试和超限截断由统一无重试、无截断模型接口替代，摘要 prompt 保持原文。
- MemoryBank 使用隔离 Python 3.11 CPU 环境和原 FAISS 后端，向量接口适配为统一的 1024 维模型。以题目日期作为冻结的 as-of 日期；LoCoMo 无题目日期时使用完整历史的最后会话日期。保留完整历史后执行作者遗忘机制，随机种子 42，不以机器日期计算历史年龄，不根据答案选择保留记忆。
- 三个方法都在统一评测框架中重新运行。论文表格中的历史成绩只用于背景，不填入本次实测主表。

## 3. 两个主流 benchmark

| Benchmark | 正式来源 | 主实验范围 | 主指标 |
|---|---|---|---|
| **LoCoMo 原始公开版** | ACL 2024 | 10 段长期对话，类别 1–4 共 1,540 道回答题 | 官方逐题 F1 的平均及分类 F1 |
| **LongMemEval-S，v1 官方 cleaned 数据** | ICLR 2025 | 全部 500 道题，包含 30 道拒答题 | 官方 judge accuracy 及能力分项 |

LoCoMo 已被 A-Mem 等后续记忆论文采用；LongMemEval 是专门评估长期用户—助手记忆的正式会议 benchmark。使用原有任务的官方清洗数据，不改用新一代 benchmark。[LoCoMo 论文](https://aclanthology.org/2024.acl-long.747/)、[LongMemEval 论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html)、[LongMemEval 官方数据说明](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/README.md)。

LoCoMo 分类按官方代码映射：`4=单跳，1=多跳，2=时间，3=开放域，5=对抗`，不能照论文的文字枚举顺序映射。类别 5 的 446 道题不并入回答题 F1；本轮优先利用 LongMemEval 的 30 道拒答题检验拒答，控制工作量。[LoCoMo 官方评分](https://github.com/snap-research/LoCoMo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py)。

LongMemEval-S 平均历史规模约 115k token，覆盖 user、assistant、preference、multi-session、temporal-reasoning 和 knowledge-update 六种题型。拒答题已包含在 500 道题中。使用官方分类判分提示与 `gpt-4o-2024-08-06` judge；若服务无法提供该固定模型，应统一替换四个系统并明确评分差异。[官方评价实现](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)。

**数据冻结建议：**

- LoCoMo 仓库：`3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`；`locomo10.json` 核查到的 SHA256 为 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`。
- LongMemEval evaluator：`9e0b455f4ef0e2ab8f2e582289761153549043fc`；[cleaned 数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) revision：`98d7416c24c778c2fee6e6f3006e7a073259d48f`。S 文件下载后计算并冻结 SHA256；本计划不把 oracle 文件哈希冒充 S 文件哈希。

## 4. 公平对比配置

| 项目 | 统一规则 |
|---|---|
| 生成与回答模型 | 沿用当前 `gpt-4o-mini`，固定同一可用版本与供应商；temperature=0 |
| 向量模型 | `text-embedding-3-small`，1024 维；MemoryBank 只适配 encoder，不替换 FAISS 检索或遗忘/强化算法 |
| 最终证据预算 | 主实验最大 16,000 token；所有方法用同一 tokenizer 计数 |
| 回答流程 | 一次检索得到各方法原生证据，再由统一回答提示和模型作答 |
| 输入信息 | 同样的历史、日期、说话者、会话边界与问题日期；不提供答案或人工证据标记 |
| 写入顺序 | 固定原始时间顺序；每个方法从空索引构建 |
| 作用域 | 每份 benchmark 历史对应独立记忆空间；不同问题的回答不写回索引 |
| 超参 | 开发阶段有限调参，正式评测前冻结；每个系统公布实际配置 |

RAG 初始采用 512 token 分块、64 token overlap；开发阶段比较 512/1024 分块和 `top-k=5/10/20/40`，只选择一次最终配置。Mem0 保留固定 OSS 的更新候选逻辑；各方法的原生提示词、表示和存储机制保持各自设计。

统一 token 上限，不强行统一记忆条数。MemoryBank 保留作者检索返回及顺序，在原生检索后按统一规则裁剪；不能仅因为 retrieve-k 相同就宣称预算相同。Mem0/MemoryBank 返回原生记忆文本，当前方法返回回溯原文；不替它们增加当前方法的原文回溯模块。

当前系统额外调用 Jev，主表如实保留该模型，并用第 6 节的同模型消融检验归因。16k 是最终回答证据预算，不代表建库与检索总 token 或总成本相同。

## 5. 主实验与重复运行

**主表只有四行：RAG、Mem0-OSS (2025)、MemoryBank（作者公开代码）、当前方法。** 四个系统在两个 benchmark 上各完成一次预先登记的完整运行，共 `4 × (1540 + 500) = 8,160` 次问答。

| 实验 | 执行范围 | 交付结果 |
|---|---|---|
| 主对比 | 四系统 × 两 benchmark，全量一次 | LoCoMo F1、LongMemEval accuracy、各能力分项 |
| 鲁棒性 | 当前方法与预先指定的 Mem0，各额外完整运行两次 | 三次均值、标准差、每次成对差值 |
| 预算敏感性 | 当前方法与 Mem0，在固定分层诊断题上比较 4k/8k/16k 证据 | 质量与证据长度曲线 |
| 失败分析 | 对错误与方法分歧案例分层抽样约 50 例 | 遗漏、检索、错误归并、过期信息和生成错误分类 |

鲁棒性实验从空索引重新建库，不只重答问题；固定历史顺序。API 不支持 seed 时记录独立运行编号，不声称可控随机种子。主表四系统使用同一次登记运行，重复运行结果另表展示，避免拿当前方法三次均值与其他方法单次结果直接比较。

LoCoMo 可附统一 LLM judge accuracy，与 F1 分开报告。两套 benchmark 的分数不求平均做单一总分。

## 6. 四项核心消融

| 版本 | 唯一变化 | 验证目的 |
|---|---|---|
| 无增量归纳 | 保留事实、原文和召回流程，移除 observation 的生成及对应召回 | 归纳是否改善更新与多会话题 |
| 无 topic 候选约束 | 保留 topic 标签、主辅条目、来源和其他候选规则，仅取消归纳候选的同 topic 限制 | 分类约束是否减少错误归并 |
| 无原文回溯输出 | 在相同冻结索引及命中条目上，向回答模型提供对应提炼事实/归纳而非原文 | 原文证据是否改善作答可靠性 |
| Jev 换成统一 LLM | 关系判断使用 `gpt-4o-mini`，保持候选对、关系定义和后续处理一致 | 排除额外模型能力的混杂 |

四项优先做完整 LongMemEval-S，再在 LoCoMo 固定诊断题集检查方向一致性。论文若主张某项机制在两套数据上普遍有效，再补该项 LoCoMo 全量结果。改变写入机制的消融必须重建索引；仅改变证据输出的消融复用经过校验的同一索引。

## 7. 目录路由的补充实验

当前配置以有效向量条数为单位，只有超过 300,000 才启用 topic 路由。常规两套 benchmark 若触发率为零，就不能用主表证明路由效率。

先做当前方法开/关路由的冻结索引对照，用现有 benchmark 的固定查询构成规模子实验，不增加第三个 benchmark 或第四个 baseline。建议先测 10k、100k、300k、320k、500k 有效向量，保留 300k 边界点。记录路由触发率、检索质量、p50/p95 Search 耗时、实际读取向量量和峰值内存。

增加的记忆应是有来源的不同内容，处于相同合法检索作用域，不能靠复制同一向量制造“真实长期记忆”成绩。若只能生成索引夹具，该实验只报告检索实现的工作量与耗时，不报告真实记忆 QA 泛化。

若论文最终主张大规模检索优于标准 RAG，沿用同一个 RAG baseline 的成熟 ANN 后端进行对比，避免只比较朴素全扫。规模实验不是本轮核心问答贡献的前提；只有实际达到规模且端到端结果支持，才把路由效率写成主要贡献。减少向量读取量不等于提速。

## 8. 评价、统计与成本

**效果指标：** LoCoMo 官方 F1；LongMemEval 官方 accuracy；按题型分项，拒答单列。预先将当前方法相对 Mem0 的 LongMemEval 总 accuracy 差值设为主要比较；LoCoMo 和其余 baseline 比较提供跨数据与机制证据。

**诊断指标：** 在有合法来源标注的可回答题上检查最终交给 reader 的证据覆盖情况。不能把未输出给 reader 的全部 provenance ID 当成召回。Mem0/MemoryBank 缺少精确来源映射时，不虚构统一 Recall；以盲审的事实支持率和错误案例补充，明确可比较范围。

**不确定性：** 报告成对差值与 95% CI，bootstrap 10,000 次。LoCoMo 以 conversation 为群组，附每对话结果；只有 10 个对话群组，不把 1,540 道题当成独立用户样本。LongMemEval 按审计后的共享答案证据与原题/拒答题家族分组。数据抽样 CI 与三次运行的标准差分别报告。多个正式假设检验如使用 p 值，采用 Holm 校正；不以 p 值单独代替效果量和机制分析。

**效率指标：** 建库总时间、每百万原文 token 建库费用、Search p50/p95、Search+Answer p50/p95、平均证据 token、调用数量、存储大小和峰值内存。固定机器与并发条件，记录冷/热缓存；不同方法交错测量，降低 API 时段变化影响。

**成本分开记录：**

`总费用 = 建库费用 + 检索费用 + 回答费用 + Judge 费用 + 失败重试费用`

`单查询摊销费用(Q) = 建库费用 / Q + 检索费用 + 回答费用`

明确 Q 是该历史实际服务的查询次数。LoCoMo 一个历史对应多题，LongMemEval 很多题各自拥有不同历史，两者不能使用同一种隐藏摊销口径。优先记录供应商实际返回费用；费用缺失保留 null，credits 与 USD 不混算。计划预算可以使用公开单价与 pilot token 做估算，实测报告不能把估算当账单。

## 9. 开发集与数据曝光

先恢复或整理已用开发题、历史和错误分析清单。原来的 LoCoMo-Refined 与原始 LoCoMo 同源，换文件版本不能视为未见数据；原 8/2 或 80/20 划分也不能通过重新 shuffle 消除曝光。

- 开发与 pilot 优先使用已暴露题，新增开发题提前标记。正式测试冻结后不再据其结果修改该轮方法。
- 全量结果称为公开 benchmark 评测；同时标记已用于调参的题，并额外报告未用于调参子集及实际规模。曝光无法确认时明确记为未知，不宣称严格盲测。
- LongMemEval 用答案证据 session 的规范化内容 hash 与原题/拒答题家族分组，不能只查 session ID。仅无关 distractor 重复记录为上下文重叠，不直接认定答案泄漏。
- builder 只接收合法历史、角色和时间；排除 LoCoMo 的 QA/人工事件摘要/证据标记，以及 LongMemEval 的 answer、answer_session_ids、has_answer 等标注。检索不得根据 gold 时间范围、gold topic 或答案证据决定路由。

## 10. 四周执行安排

| 阶段 | 工作 | 验收条件 |
|---|---|---|
| 第 1 周：接入与冻结 | 下载并核验两套数据；实现原始 LoCoMo loader/F1；接入三个 baseline；LoCoMo 选两段开发对话共约 20 题，LongMemEval 选约 12 题覆盖六类能力 | 四系统均能完成 Add→Search→Answer→评分；历史完整；无标注泄漏；得到单位建库成本和耗时 |
| 第 2 周：完整主实验 | 冻结模型、源码、超参和题单；运行四系统全量；处理传输失败并保留全部记录 | 主表覆盖 8,160 次问答；逐题结果与失败原因齐全；不得仅统计成功题而隐去失败 |
| 第 3 周：消融与重复 | LongMemEval 四项消融；LoCoMo 分层诊断；当前方法与 Mem0 各补两次独立建库运行 | 效果归因、运行波动、成对差值齐全；必要的 LoCoMo 全量消融按论文主张补齐 |
| 第 4 周：分析与写作材料 | CI、成本、预算敏感性、错误分析；按实际可达规模完成路由诊断 | 完成主要结果表、分项表、消融表、成本表、质量成本图和失败案例；形成可复现说明 |

先完成四系统的小样本端到端链路，再扩展到全量。第一周的实测预算必须包含建库；不能仅按 8,160 次回答调用推算总费用。上述三次鲁棒性方案额外增加约 8,160 次问答，四项 LongMemEval 消融增加约 2,000 次问答，尚不含诊断、Judge 或建库调用。

## 11. 项目实施起点与交付物

当前方法以 `vikingjev-generic-topics-2` 配置及六类 topic 版本为起点。现有评测器使用 AML 的 LoCoMo-Refined 和 binary judge，不能直接标成原始 LoCoMo 复现；新增原始数据适配与官方指标是第一周的明确任务。

新版 topic 调整的工程验证记录明确尚未重跑 benchmark，历史五题冒烟只能说明当时链路情况。当前工作区存在未提交变更，冻结时必须保存实际源码与提示词快照，不能仅记录 HEAD commit。[当前设计](generic-topics.md)、[验证记录](generic-topics-validation.md)、[历史冒烟](vikingjev-benchmark-smoke-results.md)。

交付目录需保存数据来源与 SHA、各 baseline 原始 commit 和适配差异、依赖锁定、完整配置、源码与提示词快照、逐题检索及预测、评分与失败记录、调用费用与耗时，以及分析图表。主实验数据与各运行独立索引置于既有 data/artifacts 目录，密钥不进入快照。

**投稿准备条件：** 四系统完成一致口径的两个公开 benchmark 评测；至少一个具体机制获得主对比、消融与案例的共同支持；增益相对统计不确定性具有可解释性；成本和失败记录完整。若只有旧 smoke、单次小幅分数变化或额外 Jev 模型带来的提升，继续完善方法与证据，不能据此承诺 EI 会议录用。
