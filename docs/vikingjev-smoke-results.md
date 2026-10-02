# VikingJev OpenRouter 冒烟结果

2026-10-02，在 `VikingJev` 分支使用 `configs/service/vikingjev.yaml` 和独立新索引完成一次真实 OpenRouter 冒烟。没有重跑 benchmark。

运行命令：

```bash
uv run --no-sync --env-file .env python -m scripts.vikingjev_smoke \
  --output-dir artifacts/vikingjev/openrouter-smoke-2026-10-02
```

结果：**3 次首次写入成功，2/2 问答成功，0 失败**。20 次逻辑模型调用，共耗时 **26.182 秒**。

| 阶段 | 调用数 | 模型调用累计耗时 | 供应商返回费用（credits） |
| --- | ---: | ---: | ---: |
| 写入：抽取、向量、归纳判断与改写 | 14 | 20.492 秒 | 0.000835626 |
| 检索：问题向量与 Jev 目录判断 | 4 | 2.265 秒 | 0.000198204 |
| GPT 回答 | 2 | 3.076 秒 | 0.000135300 |
| 合计 | 20 | 25.833 秒 | **0.001169130** |

费用直接来自各次 API 响应的 `usage.cost`，20 次均有该字段；未用 token 数估算费用，未把 credits 转成 USD。SDK 最多重试两次，记录口径为逻辑调用的响应费用，而非账单对账。系统总耗时另外包含存储与 HTTP 测试客户端开销。

单独看目录判断：Jev 两次请求合计 **0.851 秒**，平均每次 **0.425 秒**，费用合计 **0.000197484 credits**；每次同时判断三个候选目录。

## 实际行为

| 问题 | Jev 选择 | GPT 答案 |
| --- | --- | --- |
| 2026 年 3 月当前饮品，以及 1 月的旧偏好 | 保留两条饮品记忆，排除项目目录 | 当前 green tea；1 月 coffee |
| 天文软件项目的代号 | 保留项目目录，排除两条饮品记忆 | ORION |

GPT 输入的是经过源文件 SHA256 校验后恢复的原始记忆，输出的证据 ID 均来自本次 Search 返回值。

额外检查通过：目录三个层级文件存在、独立 SQLite 向量存在、消息和 chunk 表没有存原文副本、HTTP 结构不变、重启前后重复 Add 均为零额外模型调用、其他用户查询为空且不调用模型。

实际返回的模型标识：`openai/gpt-4o-mini`、`text-embedding-3-small`、`typesafe/jev-1.13-20260917`。

工程测试：`330 passed`。其中新增 20 项 Jev 选择器测试、21 项目录记忆测试；覆盖无关目录不能经邻居恢复回流、原文缺失/损坏、事务失败与提交前后中断。保留一条现有 FastAPI/Starlette TestClient 依赖弃用警告。

完整逐调用记录、目录判断、答案、配置、源码/提示词哈希和报告位于本地 `artifacts/vikingjev/openrouter-smoke-2026-10-02/`。这是合成数据的链路验证，不能据此推断 benchmark 准确率、检索速度或费用相对 Jevsight 已提升。已成功的冒烟不重复运行。
