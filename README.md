# AML Memory

文本赛道 Add/Search 服务：原文和提炼事实分别索引，BM25、原文向量、事实向量召回，RRF 融合并恢复邻近原文。Search 只返回可追溯的原文证据。服务模式下四组 B0–B3 共用写入结果，仅改变检索配置；本地评测每次运行独立建索引。

## 安装与启动

需要 Python 3.12、`uv` 和两个环境变量：

```bash
export OPENAI_API_KEY='你的 OpenRouter API Key'
export AML_MEMORY_KEY='自己设置的服务密钥'
uv sync --extra test
uv run aml-memory serve --config configs/service/vikingjev.yaml
```

生成模型和向量模型通过 `https://openrouter.ai/api/v1` 调用 `openai/gpt-4o-mini` 和 `openai/text-embedding-3-small`（1024 维），事实提炼、Embedding、Answer 和 Judge 共用 `OPENAI_API_KEY`；VikingJev 的关系判断与大规模查询路由使用 OpenRouter Decisions API 的 `typesafe/jev-1.13`。模型标识和 Embedding 参数见 [OpenRouter 文档](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings)。当前为本地测试，`competition_mode: false`；新数据库目录为 `data/vikingjev-generic-topics-v5/`，不与历史索引混用。

模型名、地址、密钥变量名、超时和重试可在 YAML 中修改。密钥值只从环境变量读取；若填写在项目根目录的 `.env`，使用 `uv run --env-file .env aml-memory serve --config configs/service/vikingjev.yaml` 启动，评测命令同样可添加 `--env-file .env`。`/health` 无需鉴权；`/add`、`/search` 接受 `Authorization: Bearer <AML_MEMORY_KEY>` 或 `X-Api-Key: <AML_MEMORY_KEY>`。

```bash
curl -X POST http://localhost:8000/add -H 'Content-Type: application/json' -H "Authorization: Bearer $AML_MEMORY_KEY" -d '{"request_id":"r1","user_id":"u1","session_id":"s1","messages":[{"role":"user","content":"Paris is in France"}]}'
curl -X POST http://localhost:8000/search -H 'Content-Type: application/json' -H "Authorization: Bearer $AML_MEMORY_KEY" -d '{"query":"Where is Paris?","user_id":"u1","top_k":10}'
```

Add 是同步操作。相同 `request_id` 与相同请求体直接返回原响应；相同 ID 不同请求体返回 409。写入的事实必须引用当次新增原文片段，模型或向量调用失败时不写入半成品。每个 `user_id` 对应独立 SQLite 文件。B0–B3 的数据库目录相同，索引配置指纹相同；更改 Embedding、分块或事实提示词后，应配置新的数据库目录。

## 可选上下文与记忆分类

Add、Search 均接受可选 `project_id`：省略、`null`、空白字符串表示无项目，仅使用当前用户的全局记忆；非空字符串去除首尾空白，检索全局与当前项目记忆。系统不猜测项目标识、不继承会话最近的项目。请求指纹在规范化后计算，省略与显式 `null` 等价；同一请求 ID 改变项目仍返回 409。

生成模型在同一次抽取中输出六类条目：`profile`（用户档案）、`preferences`（偏好）、`entities`（实体知识）、`events`（事件与安排）、`requirements`（要求）、`methods`（方法与经验）。每条只有一个主 topic，涉及其他类别的部分拆为独立辅助条目，保留主辅关联和原文片段。用户档案、偏好默认全局；其余类别在项目中默认归当前项目，明确共享时归全局。无项目的普通条目归全局；明确属于项目但缺少项目标识的条目保存为 `unresolved`，不参加默认检索及归纳。正常条目照常提交，Add 响应增加 `unresolved_entry_count` 和 `warnings`。未确定条目不自动归属后续项目，可携带明确项目使用新请求重新写入原文。

Search 另接受可选 `event_time_range: {"start":"2026-10-01","end":"2026-11-01"}`，使用左闭右开范围与事件发生或生效区间相交筛选；只传一端表示单边范围，两端为空等同不筛选。未知时间或时区不可比较的记忆不进入显式时间筛选，写入时间不作为事实新旧或时间筛选依据。非法类型、非法日期、逆序或无法比较的范围边界返回参数错误。

证据的原文 `content` 保持可回溯，另附 `entry_ids`、`topic`、`scope`、`project_id`、`requirement_priority`；项目要求优先级为 `2`，全局要求为 `1`，其他为空。上层回答模型应执行“冲突时项目要求优先，非冲突要求并存”，不增加一次冲突判断调用。所有召回和邻居恢复遵守相同作用域、时间范围，避免混合原文泄漏其他项目内容。具体结构与规则见 [通用 topic 说明](docs/generic-topics.md)。

## 本地评测

评测配置位于 `configs/evaluation/`。将 LoCoMo-Refined 的 `conversations.jsonl`、`questions.jsonl` 放在 `data/locomo/`；将 LongMemEval-S cleaned JSON 放在 `data/longmemeval/longmemeval_s_cleaned.json`。PersonaMem 使用完整的 `data/personamem-v2/` 目录；BEAM 使用 `data/beam/100K.parquet`；CLBench 使用 `data/clbench/CL-bench.jsonl`。将 [AML 公开契约仓库](https://github.com/AML-memory/agent-memory-leaderboard)检出到 `data/aml-contract/`，固定为评测 YAML 中的 commit。把 YAML 的 `data_sha256` 改为下载文件或目录的实际哈希；哈希算法见 `evaluation.sha256`。manifest 同时记录 Git commit 和实际 Python 源码快照，未提交的代码以快照为准。

PersonaMem 数据准备：运行 `uv run python scripts/prepare_personamem.py`。脚本从 `bowen-upenn/PersonaMem-v2` 的固定 revision `ed956dea41521fc4499acbc63f966e0fd3c053ba` 下载 CSV 和它引用的 200 个 32k 历史，保留 CSV 的原始相对路径，并保存 `source.json` 来源及逐文件 SHA-256。评测前校验全部历史的 persona ID 和 `chat_history` 消息；缺文件、空历史或非法选项直接报数据错误，不会用 persona 元数据或当前问题替代历史。官方版本中的空 assistant 消息仅跳过，不修改原始数据或补充替代文本；历史消息只读取 role/content。只写入真实历史，同一历史跨题复用。

```bash
uv run aml-memory evaluate --config configs/evaluation/locomo-dev.yaml
uv run aml-memory evaluate --config configs/evaluation/longmemeval-dev.yaml
uv run --env-file .env aml-memory evaluate --config configs/evaluation/personamem-dev.yaml
uv run --env-file .env aml-memory evaluate --config configs/evaluation/beam-dev.yaml
```

评测按原始历史分组拆分开发/验证；LoCoMo 的 10 组固定为 8/2。题数由 `sample_limit` 控制，PersonaMem/BEAM 开发配置各运行 5 题。每次运行生成 `artifacts/runs/<run_id>/`，包含 manifest、实际生效的配置快照、Python 源码快照、逐题检索、预测、指标和失败记录，以及该次运行独立的 `memory-index/` 数据库。历史只在同次运行内复用，不读取旧评测索引。LoCoMo/LongMemEval 和 BEAM 的提示词从指定 AML commit 的 `pipeline.py` 通过 AST 读取，不执行该脚本。结果标记为 `local-development`，不是 AML 官方 Full 结果，BEAM 不包含 event-ordering 专用排序指标。公开数据和运行结果保存在本地忽略目录中，评测需要 API 密钥。

当前适配器为每份历史分组生成稳定的 `benchmark:<dataset>:<group>` 项目标识，Add 和 Search 使用同一标识，跨会话、跨问题保持一致。五种回答路径统一渲染原文及作用域元数据，并使用固定要求优先级策略；manifest 和提示词快照记录回答系统策略及其哈希。本轮代码调整不重复运行已有五套 benchmark，历史报告保留其原始源码和配置口径。

PersonaMem 将三个错误选项解析成独立选项，与正确答案组成四选一；根据 seed 和题目 ID 确定性打乱。四个选项同时用于检索和回答，模型只返回 `{"choice":"A"}` 等 JSON，无法选择时返回 `{"choice":"UNKNOWN"}`。只比较实际选择与正确标号；错误项、拒答、非法 JSON、多选或越界选择均为 `WRONG`，不调用 Judge。预测保存选项顺序、原始模型输出、选择标号、正确标号和历史来源。

BEAM 每题通过配置中的 Judge 一次性对全部 rubric 做语义评分，每条返回 `index`、数值 `0 / 0.5 / 1` 和非空理由。事实、同义表达和拒答都按该条 rubric 的要求判断，不使用关键词或字符串包含规则。空或非法 rubric 在 API 调用前使预检失败；Judge 无效输出记录 `judge_failure`，不产生分数。预测保存 `beam_score` 和逐条 `rubric_scores`；`mean_rubric_score` 是完成判分题目的平均分，`accuracy` 是全部 rubric 得到 1 分的题目比例。评分要求不提供给回答模型。

所有数据集的指标都报告 `requested`、`evaluated`、`failed`、`completion_rate`；`accuracy` 的分母是 `evaluated`，失败题必须结合完成比例查看。manifest 记录 `scoring_protocol` 和 `data_source`。CLBench 继续使用配置中的 Judge 结合原始对话判定 rubric，全部满足才标记 `CORRECT`，并保存逐条布尔结果及满足比例。

## Jevsight 实验

`Jevsight` 分支在 B3 上增加事实时间字段、增量归纳和 Jev 关系判断。A/B/C/D 的服务配置分别为 `configs/service/jevsight-{a,b,c,d}.yaml`：A 保留 B3 输入和检索；B 增加时间；C 使用生成模型判断归纳关系；D 只把 C 的关系判断换成 `typesafe/jev-1.13`。C/D 的改写仍用 `openai/gpt-4o-mini`。Jev 通过 OpenRouter Decisions API 调用，沿用 `OPENAI_API_KEY`。

归纳覆盖全部已提取事实，每条最多匹配五个候选，同批结果也参与匹配。重复只补来源，新增或不确定独立保留，补充、状态变化、纠错和冲突按事项合并改写；每批生成最多八条归纳。同一次 Add 对同一事项只生成一个版本，失败不提交任何新记忆。归纳向量增加一条最多 50 个候选、权重 1.0 的召回路径，沿来源事实回到原文，继续使用 20 个种子、邻居恢复和 16,000 token 证据预算。

时间保留来源的精度和时区信息，未知值为空；系统写入时间仅用于审计。版本、来源和决策记录保存在独立的新 schema 索引中。模型、提示词、时间与判断规则参与指纹；旧索引不迁移，必须从原文重建。

```bash
uv run --env-file .env python scripts/jevsight_experiments.py --prepare
uv run --env-file .env python scripts/jevsight_experiments.py --run dev --workers 2
uv run --env-file .env python scripts/jevsight_experiments.py --run validation --workers 2
uv run python scripts/jevsight_experiments.py --report
```

固定清单与源码对照位于 `artifacts/jevsight/selections/` 和 `artifacts/jevsight/baseline/`。已完成的五个 benchmark 开发样本 A/B/C/D 对照见 `artifacts/jevsight/available-results.md`、`available-results.json`；它们来自同一历史源码快照，当前源码尚未完成整套重新验证。原计划每组运行 44 道开发题和 90 道固定验证题，目前只有 LoCoMo 的 A/B 固定验证完整，因此不能据此判定最终效果。每次运行保留源码、配置、提示词、逐题预测和评分、逐调用成本与耗时；费用缺失值为空，供应商返回的 credits 与 USD 分开记录。

这是本地固定样本实验，不能称为五套完整或官方 Full 评测。LongMemEval 的源会话在开发与验证划分间可能交叉；BEAM 采用本地平均 rubric 分数；CLBench 沿用已有语义任务完成评分协议，其 Answer 可见 rubric，Judge 读取原始历史。

## VikingJev 目录记忆实验

`VikingJev` 在 Jevsight 上参考 OpenViking 源码增加目录内的 L0 摘要、L1 概览和 L2 原始消息，向量与来源关系独立存放于 SQLite。当前写入由既有生成模型抽取六类通用条目，删除单独的 Jev 写入分类调用；逻辑目录为“作用域 → topic → 事项”，仅事件与安排设置事项层。Jev 在同用户、同作用域、同 topic 内判断归纳关系，同一事项的新侧面可保留独立归纳并共享事项标识。

查询先按用户、项目和时间限定可检索范围，按有效向量条数切换：不超过可配置阈值 300,000 时直接检索，超过时判断六个固定主题描述并限定目录范围，不增加逐层递归判断。原文片段、事实和归纳分别召回，回溯来源证据并保留现有 RRF、20 个种子和 16,000 token 预算。使用 `configs/service/vikingjev.yaml` 和独立的 schema 5 新索引；旧库必须从原文重建，不迁移。新指纹纳入抽取结构、主题定义和作用域规则；供应商费用缺失记为空，不推算节省比例。

```bash
uv run --env-file .env python -m scripts.vikingjev_smoke
```

源码参考、目录树、原文完整性和实验边界见 [VikingJev 说明](docs/vikingjev.md)。

以下冒烟报告来自调整前冻结源码，保留历史结果，本轮不重跑。合成数据 OpenRouter 冒烟结果：3 次写入、2/2 问答通过，当时 330 项工程测试通过；费用与耗时见 [结果报告](docs/vikingjev-smoke-results.md)。

五个真实 benchmark 的冒烟配置位于 `configs/evaluation/vikingjev-smoke/`，各取既有开发清单的第一题，保留完整历史、数据哈希和原生评分流程。CLBench 使用 `vikingjev-clbench-smoke.yaml`，把抽取/归纳模型输出上限提高到 16,384，避免长输入的事实抽取被截断。冒烟验证完整链路，答案得分单独记录。

五套均已完成 `requested=evaluated=1, failed=0`；其中 BEAM 样本回答错误，结果保留。逐套评分、完整历史验证、判断耗时和实际调用费用见 [五套 benchmark 冒烟报告](docs/vikingjev-benchmark-smoke-results.md)。

## Docker

```bash
docker build -t aml-memory .
docker run -p 8000:8000 -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" \
  -e OPENAI_API_KEY -e AML_MEMORY_KEY \
  -e AML_CONFIG_PATH=configs/service/vikingjev.yaml aml-memory
```

部署时持久化 `/app/data` 并在反向代理提供 HTTPS。当前服务使用单 worker；同一用户的 Add 在进程内串行，Search 使用 SQLite 已提交读取快照。系统仅实现文本 Add/Search；多模态输入不属于此版本。

## 第一周论文实验

论文第一周实验入口使用冻结作者源码的 RAG、Mem0-OSS、MemoryBank，与当前方法在原始 LoCoMo 和 LongMemEval-S 上对比。数据、依赖、题单、官方评分及预算门禁见[第一周报告](docs/week1-results-2026-10-14.md)。

另可将 MemoryBank 论文自带评测集（15 虚拟用户 × 10 天对话，中英各 100 道 probing 题）作为独立 benchmark 接入：数据在 `data/week1/datasets/memorybank/`，加载/归一化/全量题单与 LME 兼容的二元 LLM-judge 见 `src/week1_memorybank_benchmark.py`，离线说明见 [MemoryBank benchmark](docs/memorybank-benchmark.md)。官方参考答案未发布；当前 judge 在无金标时使用对话证据。未改动原 $20 预算上限，也未自动并入付费 runner。

```bash
uv run python -m scripts.week1_experiments prepare
uv run python -m scripts.week1_experiments preflight
uv run python -m scripts.week1_experiments run --budget-usd 20
uv run python -m scripts.week1_experiments run --run-dir artifacts/week1/20261016T011008129863Z
uv run python -m scripts.week1_experiments report --run-dir artifacts/week1/20261016T011008129863Z
```

2026-10-18 已验证当前 VPN 网络并恢复 51 个连接失败建库任务及其 96 项问答，原 93 个成功响应与 22 条评分逐项保持不变。原运行 `20261016T011008129863Z` 于北京时间 10:28:51 续跑，Python PID `7115`，日志为 `artifacts/week1/task-executor-resume-2026-10-18.log`；状态需结合 `active-run.json` 和系统进程核对。1 个 MemoryBank/conv-41 读取超时任务保持隔离，避免重复可能已完成的调用。新增恢复及相关回归 64 项通过，原执行器全量回归 802 项通过；作者源码、模型、题单和共享预算不变。详见[网络恢复记录](docs/week1-network-recovery-2026-10-18.md)和[第一周结果](docs/week1-results-2026-10-14.md)。

## 测试

```bash
uv run pytest -q
```

## 官方服务部署

官方开源文本赛道使用 `configs/service/official-c.yaml`：Jevsight C 的时间字段和增量归纳流程，事实提炼、关系判断及改写均使用 `openai/gpt-4o-mini`，向量使用 `openai/text-embedding-3-small`（1024 维），不调用 Jev 模型。`competition_mode` 校验生成模型；向量模型在申请时如实披露。

Docker 镜像通过 `uv.lock` 固定 Python 依赖，以非 root 用户运行，并预置 tokenizer 缓存。源码、提示词和服务配置进入镜像，密钥与评测数据保存在镜像外。Compose 默认使用独立的 `memory-data`、`memory-artifacts` 持久卷并仅绑定 `127.0.0.1:8000`，公网由 HTTPS 反向代理转发。Add/Search 的代理超时设为 1800 秒，与平台的单次请求上限一致。

```bash
docker compose --env-file /opt/anchormem/.env \
  --project-name anchormem \
  -f /opt/anchormem/source/compose.yaml \
  -f /opt/anchormem/compose.override.yaml up -d --build
```

密钥文件 `/opt/anchormem/.env` 仅供服务端读取。服务端源码目录为 `/opt/anchormem/source`，后续构建、测试和调试在该目录执行。运行官方评测前须先提交自托管接口申请，由 AML 审核签发 Eval Key，再运行 Smoke 和 Full；本地接口检查不构成官方评测结果。
