# AML Memory

文本赛道 Add/Search 服务：原文和提炼事实分别索引，BM25、原文向量、事实向量召回，RRF 融合并恢复邻近原文。Search 只返回可追溯的原文证据。四组 B0–B3 共用写入结果，仅改变检索配置。

## 安装与启动

需要 Python 3.12、`uv` 和三个环境变量：

```bash
export OPENAI_API_KEY='...'
export DASHSCOPE_API_KEY='...'
export AML_MEMORY_KEY='自己设置的服务密钥'
uv sync --extra test
uv run aml-memory serve --config configs/service/v1-b3.yaml
```

默认使用 OpenAI 的 `gpt-4o-mini` 提炼事实，以及 DashScope 兼容接口的 `text-embedding-v4` 生成 1024 维向量。模型名、地址、密钥变量名、超时和重试可在 YAML 中修改。真正的密钥值只从环境变量读取。`/health` 无需鉴权；`/add`、`/search` 接受 `Authorization: Bearer <AML_MEMORY_KEY>` 或 `X-Api-Key: <AML_MEMORY_KEY>`。

```bash
curl -X POST http://localhost:8000/add -H 'Content-Type: application/json' -H "Authorization: Bearer $AML_MEMORY_KEY" -d '{"request_id":"r1","user_id":"u1","session_id":"s1","messages":[{"role":"user","content":"Paris is in France"}]}'
curl -X POST http://localhost:8000/search -H 'Content-Type: application/json' -H "Authorization: Bearer $AML_MEMORY_KEY" -d '{"query":"Where is Paris?","user_id":"u1","top_k":10}'
```

Add 是同步操作。相同 `request_id` 与相同请求体直接返回原响应；相同 ID 不同请求体返回 409。写入的事实必须引用当次新增原文片段，模型或向量调用失败时不写入半成品。每个 `user_id` 对应独立 SQLite 文件。B0–B3 的数据库目录相同，索引配置指纹相同；更改 Embedding、分块或事实提示词后，应配置新的数据库目录。

## 本地评测

评测配置位于 `configs/evaluation/`。将 LoCoMo-Refined 的 `conversations.jsonl`、`questions.jsonl` 放在 `data/locomo/`；将 LongMemEval-S cleaned JSON 放在 `data/longmemeval/longmemeval_s_cleaned.json`。将 [AML 公开契约仓库](https://github.com/AML-memory/agent-memory-leaderboard)检出到 `data/aml-contract/`，固定为评测 YAML 中的 commit。把 YAML 的 `data_sha256` 改为下载文件或目录的实际哈希；哈希算法见 `evaluation.sha256`。评测前应先提交本仓库的代码，以便 manifest 记录 commit。

```bash
uv run aml-memory evaluate --config configs/evaluation/locomo-dev.yaml
uv run aml-memory evaluate --config configs/evaluation/longmemeval-dev.yaml
```

评测按原始历史分组拆分开发/验证；LoCoMo 的 10 组固定为 8/2。开发配置先抽 30 题检查链路。每次运行生成 `artifacts/runs/<run_id>/`，包含 manifest、配置快照、逐题检索、预测、指标和失败记录。Answer/Judge 提示词从指定 AML commit 的 `pipeline.py` 读取，评分标记为 `local-development`；它不是 AML 官方 Full 结果。评测需要真实公开数据和 API 密钥，仓库内不附带数据或运行分数。

## Docker

```bash
docker build -t aml-memory .
docker run -p 8000:8000 -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" \
  -e OPENAI_API_KEY -e DASHSCOPE_API_KEY -e AML_MEMORY_KEY \
  -e AML_CONFIG_PATH=configs/service/v1-b3.yaml aml-memory
```

部署时持久化 `/app/data` 并在反向代理提供 HTTPS。当前服务使用单 worker；同一用户的 Add 在进程内串行，Search 使用 SQLite 已提交读取快照。系统仅实现文本 Add/Search；多模态输入不属于此版本。

## 测试

```bash
uv run pytest -q
```
