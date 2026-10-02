# 全目录 Jev 概率 Top-4 冒烟实验

每个 benchmark 使用与上一轮相同的一道题、完整冻结索引、模型、回答与评分提示词；只改变目录选择策略。Add 为幂等验证，没有重新抽取或计入首次写入费用。全部通过 requested=evaluated=1、failed=0 验证。每类仅一题，不代表完整 benchmark 准确率。

策略：读取全部用户目录摘要，按 ID 稳定分批，每批最多 10 个；以 relevant 概率为第一排序依据，uncertain 概率及 ID 打破平局，选前 4 个。类别不作硬过滤。各批概率未经跨批校准。

查询耗时为 Search + Answer + Judge；费用为供应商实际 usage.cost credits。API 延迟包含网络、排队和推理，不能单独归因为算法。

| Benchmark | 分数：上轮→本轮 | 查询秒：上轮→本轮 | credits：上轮→本轮 | 目录候选→选中 | 读取向量：上轮→本轮 |
|---|---:|---:|---:|---:|---:|
| locomo | 1 → 1 | 8.072 → 8.282 | 0.000806304 → 0.002977128 | 46 → 4 | 80 → 136 |
| longmemeval | 0 → 1 | 6.515 → 7.882 | 0.000704786 → 0.004434788 | 53 → 4 | 53 → 83 |
| personamem | 1 → 1 | 5.012 → 6.250 | 0.001398814 → 0.001697764 | 10 → 4 | 160 → 298 |
| beam | 0 → 0 | 8.345 → 8.313 | 0.003060080 → 0.003033230 | 10 → 4 | 113 → 195 |
| clbench | 1 → 1 | 20.382 → 16.762 | 0.004516046 → 0.003175496 | 1 → 1 | 89 → 88 |

## 分阶段调用

| Benchmark | 阶段 | 调用数 | 秒：上轮→本轮 | credits：上轮→本轮 | 本轮命中 prompt cache |
|---|---|---:|---:|---:|---|
| locomo | embedding | 1 | 2.778 → 1.861 | 0.000000300 → 0.000000300 | False |
| locomo | directory_selection_jev | 5 | 0.794 → 2.785 | 0.000589554 → 0.002569728 | False |
| locomo | answer | 1 | 2.136 → 1.865 | 0.000113100 → 0.000304950 | False |
| locomo | judge | 1 | 2.339 → 1.732 | 0.000103350 → 0.000102150 | False |
| longmemeval | embedding | 1 | 1.444 → 1.651 | 0.000000140 → 0.000000140 | False |
| longmemeval | directory_selection_jev | 6 | 2.372 → 3.192 | 0.000547596 → 0.003173898 | False |
| longmemeval | answer | 1 | 1.667 → 2.179 | 0.000055650 → 0.001160400 | False |
| longmemeval | judge | 1 | 0.981 → 0.805 | 0.000101400 → 0.000100350 | False |
| personamem | embedding | 1 | 1.666 → 1.585 | 0.000005320 → 0.000005320 | False |
| personamem | directory_selection_jev | 1 | 1.126 → 2.037 | 0.000621894 → 0.000621894 | False |
| personamem | answer | 1 | 2.181 → 2.579 | 0.000771600 → 0.001070550 | False |
| beam | embedding | 1 | 2.414 → 1.571 | 0.000000380 → 0.000000380 | False |
| beam | directory_selection_jev | 1 | 0.968 → 1.582 | 0.000523950 → 0.000523950 | False |
| beam | answer | 1 | 3.309 → 2.831 | 0.002348850 → 0.002322000 | False |
| beam | judge | 1 | 1.624 → 2.301 | 0.000186900 → 0.000186900 | False |
| clbench | embedding | 1 | 1.593 → 2.161 | 0.000000620 → 0.000000620 | False |
| clbench | directory_selection_jev | 1 | 1.408 → 0.644 | 0.000090426 → 0.000090426 | False |
| clbench | answer | 1 | 14.157 → 11.755 | 0.001669650 → 0.001618650 | True |
| clbench | judge | 1 | 3.187 → 2.184 | 0.002755350 → 0.001465800 | True |

## 运行记录

全部尝试已知费用 0.015318406 credits；缺失费用调用 0 次。

[机器报告](results/vikingjev-directory-all-top4.json)。各运行包含逐题结果、证据、模型调用记录和全部目录概率：

- locomo: [运行](../artifacts/vikingjev/directory-all-top4/runs/20261002T082937975820Z-directory-all-top4-locomo)
- longmemeval: [运行](../artifacts/vikingjev/directory-all-top4/runs/20261002T082941492996Z-directory-all-top4-longmemeval)
- personamem: [运行](../artifacts/vikingjev/directory-all-top4/runs/20261002T082939010957Z-directory-all-top4-personamem)
- beam: [运行](../artifacts/vikingjev/directory-all-top4/runs/20261002T082938185049Z-directory-all-top4-beam)
- clbench: [运行](../artifacts/vikingjev/directory-all-top4/runs/20261002T082939095839Z-directory-all-top4-clbench)

## 判断与限制

- 336 项工程测试通过。五类各一题全部完成，无失败、无重跑。
- LongMemEval 的正确目录此前没有进入前十候选，本次进入完整 53 目录判断集，Jev 返回 relevant=1 并入选 Top-4；最终正确回答 Business Administration。其余四题评分未变，BEAM 仍为 0。
- LoCoMo 和 LongMemEval 分别需要 5、6 次 Jev 调用，判断费用显著增加；PersonaMem 判断费用不变，但扩大证据使回答费用增加。总实际查询费用由 0.010486030 增至 0.015318406 credits，约增加 46.1%。本轮证明修复了这个漏召回案例，未证明降本。
- CLBench 只有一个目录，两种策略选取范围相同。此次 Answer 与 Judge 都命中 prompt cache，不能将费用或耗时降低归因于 Top-4 策略。
- 全目录判断消除了前置候选截断；仍可能因摘要遗漏、Jev 排序或 Top-4 截断漏掉证据。跨批概率也未经校准。每类一题不足以证明总体准确率提升。
