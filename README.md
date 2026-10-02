# AML Memory

文本赛道 Add/Search 服务：原文和提炼事实分别索引，BM25、原文向量、事实向量召回，RRF 融合并恢复邻近原文。Search 只返回可追溯的原文证据。服务模式下四组 B0–B3 共用写入结果，仅改变检索配置；本地评测每次运行独立建索引。

## 安装与启动

需要 Python 3.12、`uv` 和两个环境变量：

```bash
export OPENAI_API_KEY='你的 OpenRouter API Key'
export AML_MEMORY_KEY='自己设置的服务密钥'
uv sync --extra test
uv run aml-memory serve --config configs/service/v1-b3.yaml
```

原有配置已改为通过 `https://openrouter.ai/api/v1` 调用 `openai/gpt-4o-mini` 和 `openai/text-embedding-3-small`（1024 维），事实提炼、Embedding、Answer 和 Judge 都共用 `OPENAI_API_KEY`。模型标识和 Embedding 参数见 [OpenRouter 文档](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings)。当前为本地测试，`competition_mode: false`；数据库目录为 `data/openrouter-index/`，避免混用之前模型的向量。

模型名、地址、密钥变量名、超时和重试可在原有 YAML 中修改。密钥值只从环境变量读取；若填写在项目根目录的 `.env`，使用 `uv run --env-file .env aml-memory serve --config configs/service/v1-b3.yaml` 启动，评测命令同样可添加 `--env-file .env`。`/health` 无需鉴权；`/add`、`/search` 接受 `Authorization: Bearer <AML_MEMORY_KEY>` 或 `X-Api-Key: <AML_MEMORY_KEY>`。

```bash
curl -X POST http://localhost:8000/add -H 'Content-Type: application/json' -H "Authorization: Bearer $AML_MEMORY_KEY" -d '{"request_id":"r1","user_id":"u1","session_id":"s1","messages":[{"role":"user","content":"Paris is in France"}]}'
curl -X POST http://localhost:8000/search -H 'Content-Type: application/json' -H "Authorization: Bearer $AML_MEMORY_KEY" -d '{"query":"Where is Paris?","user_id":"u1","top_k":10}'
```

Add 是同步操作。相同 `request_id` 与相同请求体直接返回原响应；相同 ID 不同请求体返回 409。写入的事实必须引用当次新增原文片段，模型或向量调用失败时不写入半成品。每个 `user_id` 对应独立 SQLite 文件。B0–B3 的数据库目录相同，索引配置指纹相同；更改 Embedding、分块或事实提示词后，应配置新的数据库目录。

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

`VikingJev` 在 Jevsight 上参考 OpenViking 源码增加目录内的 L0 摘要、L1 概览和 L2 原始消息，向量与来源关系独立存放于 SQLite。向量召回后由 Jev 批量判断候选目录，再读取原文证据交给 GPT。使用 `configs/service/vikingjev.yaml` 和独立的 schema 3 新索引；旧库不迁移。

```bash
uv run --env-file .env python -m scripts.vikingjev_smoke
```

源码参考、目录树、原文完整性和实验边界见 [VikingJev 说明](docs/vikingjev.md)。

合成数据 OpenRouter 冒烟结果：3 次写入、2/2 问答通过，330 项工程测试通过；费用与耗时见 [结果报告](docs/vikingjev-smoke-results.md)。

五个真实 benchmark 的冒烟配置位于 `configs/evaluation/vikingjev-smoke/`，各取既有开发清单的第一题，保留完整历史、数据哈希和原生评分流程。CLBench 使用 `vikingjev-clbench-smoke.yaml`，把抽取/归纳模型输出上限提高到 16,384，避免长输入的事实抽取被截断。冒烟验证完整链路，答案得分单独记录。

五套均已完成 `requested=evaluated=1, failed=0`；其中 BEAM 样本回答错误，结果保留。逐套评分、完整历史验证、判断耗时和实际调用费用见 [五套 benchmark 冒烟报告](docs/vikingjev-benchmark-smoke-results.md)。

## Docker

```bash
docker build -t aml-memory .
docker run -p 8000:8000 -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" \
  -e OPENAI_API_KEY -e AML_MEMORY_KEY \
  -e AML_CONFIG_PATH=configs/service/v1-b3.yaml aml-memory
```

部署时持久化 `/app/data` 并在反向代理提供 HTTPS。当前服务使用单 worker；同一用户的 Add 在进程内串行，Search 使用 SQLite 已提交读取快照。系统仅实现文本 Add/Search；多模态输入不属于此版本。

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
