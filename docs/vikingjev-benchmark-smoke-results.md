# VikingJev 五个真实 Benchmark 冒烟记录

执行日期：2026-10-02。源代码提交：94dd453809e5bfac118cc3b42c53b32dfee09b1c。

五套流程均完成：每套 requested=evaluated=1、failed=0。
每套只用一题冻结的真实样本，写入该题完整历史，再执行 Add → Search/Jev → Answer → 原生评分 → 保存结果。没有截断历史，没有重跑已成功的样本。这是完整历史冒烟，不是 Full 评测或效果提升对照。

## 逐套结果

| Benchmark | 完整历史消息 | Add 批次 | requested / evaluated / failed | 原生主指标 | 结果 |
|---|---:|---:|---|---|---|
| LoCoMo-Refined | 663 | 46 | 1 / 1 / 0 | Binary accuracy = 1 | 链路通过；CORRECT |
| LongMemEval-S | 550 | 53 | 1 / 1 / 0 | Binary accuracy = 1 | 链路通过；CORRECT |
| PersonaMem | 189 | 10 | 1 / 1 / 0 | 四选一准确率 = 1 | 链路通过；CORRECT |
| BEAM 100K | 194 | 10 | 1 / 1 / 0 | rubric 均分 = 0 | 链路通过；WRONG |
| CLBench | 6 | 1 | 1 / 1 / 0 | 全部 rubric 满足题目比例 = 1 | 链路通过；CORRECT |

不同 benchmark 的原生主指标不合并成总准确率；单题分数不能用于估计总体效果。

## 时间与实际调用费用

费用直接累加各调用返回的 usage.cost，单位为 provider credits。未按 token 数估算，也没有换算成美元。Add 按首次实际写入计时；Search 含 Jev 目录判断。

| Benchmark | Add 秒 | Search 秒 | Answer 秒 | Judge 秒 | 总运行秒 | 成功运行费用 credits | 费用缺失调用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo-Refined | 931.444 | 1.985 | 1.444 | 1.121 | 936.613 | 0.114711490 | 0 |
| LongMemEval-S | 1060.524 | 2.977 | 1.008 | 1.292 | 1068.964 | 0.129203610 | 0 |
| PersonaMem | 237.826 | 1.693 | 1.531 | 0.000068 | 242.157 | 0.033239134 | 0 |
| BEAM 100K | 277.689 | 1.750 | 4.385 | 2.998 | 287.536 | 0.052073720 | 0 |
| CLBench | 56.650 | 1.354 | 15.162 | 5.045 | 79.081 | 0.013550648 | 0 |

PersonaMem 的 Judge 为本地选项比对，未调用评分模型。各任务并行执行，总运行秒不应相加解释为实际等待时间。

| Benchmark | Jev 目录判断调用 | 判断秒 | 判断费用 credits | 总逻辑模型调用 |
|---|---:|---:|---:|---:|
| LoCoMo-Refined | 1 | 0.663 | 0.000575106 | 569 |
| LongMemEval-S | 1 | 1.660 | 0.000526176 | 660 |
| PersonaMem | 1 | 0.690 | 0.000621894 | 154 |
| BEAM 100K | 1 | 0.730 | 0.000523950 | 155 |
| CLBench | 1 | 0.759 | 0.000090426 | 44 |

| Benchmark | Add credits | Search credits | Answer credits | Judge credits |
|---|---:|---:|---:|---:|
| LoCoMo-Refined | 0.113833234 | 0.000575406 | 0.000199500 | 0.000103350 |
| LongMemEval-S | 0.127884994 | 0.000526316 | 0.000691050 | 0.000101250 |
| PersonaMem | 0.032127120 | 0.000627214 | 0.000484800 | 0.000000000 |
| BEAM 100K | 0.049014840 | 0.000524330 | 0.002348850 | 0.000185700 |
| CLBench | 0.007883652 | 0.000091046 | 0.002825850 | 0.002750100 |

已完成成功运行费用：**0.342778602 credits**。
失败尝试费用：**0.005532000 credits**。
本轮全部尝试合计：**0.348310602 credits**；费用缺失调用 0。

## 保留的失败记录与限制

- **BEAM 回答错误但链路完整。** beam-3-abstention-0 得分 0：原文没有确立 Trello 任务排序的具体标准，模型却回答了 Must-Have / Should-Have 等通用分类，未正确拒答。预测及 rubric 理由完整保留。
- **CLBench 首次执行失败。** 20261002T070143612466Z-vikingjev-smoke-clbench：requested=1、evaluated=0、failed=1；费用 0.005532000 credits，耗时 68.090 秒。
- CLBench 首次抽取返回 finish_reason=length。仅在专用服务配置中将生成输出上限 4096 → 16384 后重新执行失败项；完整历史、模型名称、Answer 与评分协议未改变。输出上限是故障修复参数，不作为费用估算依据。
- CLBench 采用本地语义 rubric 全部满足的题目比例；BEAM 为本地 rubric 分数。均沿用现有评分流程，不能解释为官方 Full 成绩。
- 成功运行已核验：SQLite mode=ro 读取完整历史消息计数；原文目录文件与 SHA-256 完整；原文 messages/chunks 字段为空；每题 prediction、retrieval、评分、ingestion 与 model-calls 齐全；实际 directory_selection_jev 存在，所有模型调用成功。
- 调用记录是逻辑请求口径；SDK 若发生传输重试，不会作为独立调用计入。费用缺失时保留 null，不能把已知部分合计视作完整总额。

## 结果位置

可随 Git 跟踪的机器汇总：[vikingjev-benchmark-smoke.json](results/vikingjev-benchmark-smoke.json)，只含题目 ID、哈希、计数、指标、费用、耗时、相对路径和验证状态。完整本地汇总：[summary.json](../artifacts/vikingjev/benchmark-smoke/summary.json)。以下 artifacts 原始链接仅在本地存在，不随 Git 推送；原始运行包含源码、提示词、解析后配置、冻结选题与数据哈希快照。

- LoCoMo-Refined：[conv-41#q0000](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070140999621Z-vikingjev-smoke-locomo/metrics.json)；[预测与评分](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070140999621Z-vikingjev-smoke-locomo/predictions.jsonl)，[原文检索](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070140999621Z-vikingjev-smoke-locomo/retrieval.jsonl)，[逐调用费用](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070140999621Z-vikingjev-smoke-locomo/model-calls.jsonl)。
- LongMemEval-S：[e47becba](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070138257935Z-vikingjev-smoke-longmemeval/metrics.json)；[预测与评分](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070138257935Z-vikingjev-smoke-longmemeval/predictions.jsonl)，[原文检索](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070138257935Z-vikingjev-smoke-longmemeval/retrieval.jsonl)，[逐调用费用](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070138257935Z-vikingjev-smoke-longmemeval/model-calls.jsonl)。
- PersonaMem：[pm-521-Health-0](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070251845423Z-vikingjev-smoke-personamem/metrics.json)；[预测与评分](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070251845423Z-vikingjev-smoke-personamem/predictions.jsonl)，[原文检索](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070251845423Z-vikingjev-smoke-personamem/retrieval.jsonl)，[逐调用费用](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070251845423Z-vikingjev-smoke-personamem/model-calls.jsonl)。
- BEAM 100K：[beam-3-abstention-0](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070114183500Z-vikingjev-smoke-beam/metrics.json)；[预测与评分](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070114183500Z-vikingjev-smoke-beam/predictions.jsonl)，[原文检索](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070114183500Z-vikingjev-smoke-beam/retrieval.jsonl)，[逐调用费用](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070114183500Z-vikingjev-smoke-beam/model-calls.jsonl)。
- CLBench：[cl-be9cc79d-258c-4eaf-b594-ba72ed51b530](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070401256543Z-vikingjev-smoke-clbench/metrics.json)；[预测与评分](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070401256543Z-vikingjev-smoke-clbench/predictions.jsonl)，[原文检索](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070401256543Z-vikingjev-smoke-clbench/retrieval.jsonl)，[逐调用费用](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070401256543Z-vikingjev-smoke-clbench/model-calls.jsonl)。
- CLBench 失败尝试：[诊断](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070143612466Z-vikingjev-smoke-clbench/failures.jsonl)，[逐调用费用](../artifacts/vikingjev/benchmark-smoke/runs/20261002T070143612466Z-vikingjev-smoke-clbench/model-calls.jsonl)。
