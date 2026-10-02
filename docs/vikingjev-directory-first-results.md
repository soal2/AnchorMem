# VikingJev：目录前置检索对照（2026-10-02）

已实现：目录摘要向量召回 → Jev 判断最多 10 个候选目录 → 入选目录内 BM25、chunk、fact、observation 召回 → 原文证据。拒绝和未入选目录的向量不会送入相似度计算；事实和归纳回溯也限制在入选目录。移除了目录到全部 chunk 的额外 RRF 路线。

五个 benchmark 各使用原来的同一题、完整历史与相同冻结索引。复制索引后校验哈希、索引指纹和 Add 请求载荷，再执行真实 Search → Answer → 原评分流程。原始索引未改变；未做全量评测。生成模型、判断模型、提示词和证据预算保持一致。

费用直接累加供应商 usage.cost，单位 provider credits，不从 token 数推算。下面的查询费用包含 Search、Answer 和 Judge。写入没有重新调用模型，原首次写入成本单列，不能把本次跳过写入算成提速。

## 耗时、调用费用和结果

| Benchmark | Search 秒：之前 → 现在 | 查询全链路秒：之前 → 现在 | 查询费用 credits：之前 → 现在 | 原生单题分数：之前 → 现在 |
|---|---:|---:|---:|---:|
| LoCoMo-Refined | 1.985 → 3.018 | 4.551 → 7.171 | 0.000878256 → 0.000892704 | 1 → 1 |
| LongMemEval-S | 2.977 → 4.768 | 5.278 → 7.927 | 0.001318616 → 0.000704786 | 1 → 0 |
| PersonaMem-v2 | 1.693 → 5.937 | 3.224 → 8.696 | 0.001112014 → 0.001398814 | 1 → 1 |
| BEAM 100K | 1.750 → 3.207 | 9.133 → 8.978 | 0.003058880 → 0.002681330 | 0 → 0 |
| CLBench | 1.354 → 3.970 | 21.560 → 26.154 | 0.005666996 → 0.005631746 | 1 → 1 |

BEAM 为本地 rubric 平均分，其余为各自原生准确性指标。每套只有一道题，不合并为总准确率。五个最终运行均 requested=evaluated=1、failed=0；答案错误不等于链路失败。

## 实际向量工作量

旧值由相同索引各类型向量数量与查询视角数计算，旧代码逐视角全量读取这四类向量；新值由本次真实 SQL fetchall 返回行数计数。包含目录向量；PersonaMem 有问题和问题加选项两个查询视角。

| Benchmark | 向量比较数：之前 → 现在 | 减少比例 | 接受目录 / 候选目录 | 本地向量阶段毫秒 |
|---|---:|---:|---:|---:|
| LoCoMo-Refined | 1149 → 80 | 93.0% | 1 / 10 | 2.512 |
| LongMemEval-S | 1295 → 53 | 95.9% | 0 / 10 | 8.915 |
| PersonaMem-v2 | 680 → 160 | 76.5% | 2 / 10 | 4.158 |
| BEAM 100K | 470 → 66 | 86.0% | 1 / 10 | 0.753 |
| CLBench | 89 → 89 | 0.0% | 1 / 1 | 1.584 |

**LongMemEval 的工作量下降伴随漏召回，不能视为有效提效。** 问题询问毕业专业，正确答案为 Business Administration。原文及目录 overview 均包含答案，但目录 `53b3c249e44323b5775c19c4c882abaadfb271a9a017be05717a96e8ab73111b` 未进入目录向量排名的前 10 个候选，后续候选全部被 Jev 拒绝，返回空证据，答案从正确变为错误。

其余四个样本分数保持原值，BEAM 仍错误。CLBench 只有一个目录，所以不能减少向量数量。该结果支持“前置目录能减少候选计算”，不支持“已经实现 O(log n)、普遍提速或保持准确率”。

本轮 Search 耗时均增加。向量阶段只有毫秒级，外部 Embedding/Jev 请求占据大部分 Search 时间；本轮还发生 SSL 连接中断。基线与本次不是同时采样，单题网络耗时不能用来断言算法本身变慢。

## 原首次写入与失败尝试

| Benchmark | 原首次写入秒 | 原首次写入 credits | 本次新写入模型调用 |
|---|---:|---:|---:|
| LoCoMo-Refined | 931.444 | 0.113833234 | 0 |
| LongMemEval-S | 1060.524 | 0.127884994 | 0 |
| PersonaMem-v2 | 237.826 | 0.032127120 | 0 |
| BEAM 100K | 277.689 | 0.049014840 | 0 |
| CLBench | 56.650 | 0.007883652 | 0 |

五次成功查询费用：原 0.012034762 → 新 0.011309380 credits。费用下降包含 LongMemEval 漏召回后输入变少的影响，不能全部归因为有效节省。

初次并发运行的 LoCoMo、LongMemEval 和 PersonaMem 遇到 SSL EOF 连接错误；仅重试这三个失败项，各一次。BEAM 和 CLBench 成功后未重跑。所有失败尝试保留。
全部尝试已知费用 0.011904554 credits，另有 3 次调用没有返回费用，完整总费用为 null；不把缺失值当作零。

## 工程验证与记录

334 项测试通过，包括目录判断先于局部召回、拒绝目录损坏向量不被读取、BM25 范围过滤先于候选截断、跨目录邻居隔离、原文恢复、冻结索引复用和来源哈希校验。

[机器报告](results/vikingjev-directory-first.json)。逐题结果、真实模型调用、候选目录与实际向量读取计数保存在以下本地产物中：

- LoCoMo-Refined：[运行记录](../artifacts/vikingjev/directory-first/runs/20261002T080545859227Z-directory-first-locomo)；[向量工作量](../artifacts/vikingjev/directory-first/runs/20261002T080545859227Z-directory-first-locomo/retrieval-work.json)。
- LongMemEval-S：[运行记录](../artifacts/vikingjev/directory-first/runs/20261002T080620825588Z-directory-first-longmemeval)；[向量工作量](../artifacts/vikingjev/directory-first/runs/20261002T080620825588Z-directory-first-longmemeval/retrieval-work.json)。
- PersonaMem-v2：[运行记录](../artifacts/vikingjev/directory-first/runs/20261002T080659098691Z-directory-first-personamem)；[向量工作量](../artifacts/vikingjev/directory-first/runs/20261002T080659098691Z-directory-first-personamem/retrieval-work.json)。
- BEAM 100K：[运行记录](../artifacts/vikingjev/directory-first/runs/20261002T080429328826Z-directory-first-beam)；[向量工作量](../artifacts/vikingjev/directory-first/runs/20261002T080429328826Z-directory-first-beam/retrieval-work.json)。
- CLBench：[运行记录](../artifacts/vikingjev/directory-first/runs/20261002T080429752103Z-directory-first-clbench)；[向量工作量](../artifacts/vikingjev/directory-first/runs/20261002T080429752103Z-directory-first-clbench/retrieval-work.json)。

复现命令：`uv run --no-sync --env-file .env python -m scripts.vikingjev_directory_first_comparison <dataset>`。已运行的数据集受锁保护；`--retry-failed` 只允许重试未完成项。
