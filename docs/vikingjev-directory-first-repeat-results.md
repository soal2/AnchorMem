# 目录前置检索：第二轮五套冒烟对照（2026-10-02）

用户明确要求再做一轮。每套仍为同一道题，复用同一完整冻结索引，运行实际 Search → Answer → Judge；Add 只做幂等载荷验证。所有运行均校验代码快照、索引、选题和 Answer/Judge 提示词哈希与上一轮一致。每轮每套仅一个成功样本，比较对象为相同算法的两次执行。

## 全链路结果

耗时为检索＋回答＋评分，不包含首次写入、索引复制和数据加载。费用为供应商实际 usage.cost（credits），未按 token 数估算。表格比较成功尝试；失败重试单列。

| Benchmark | 查询秒：上轮 → 本轮 | Search 秒：上轮 → 本轮 | credits：上轮 → 本轮 | 原生单题分数：上轮 → 本轮 |
|---|---:|---:|---:|---:|
| LoCoMo | 7.171 → 8.072 | 3.018 → 3.595 | 0.000892704 → 0.000806304 | 1 → 1 |
| LongMemEval | 7.927 → 6.515 | 4.768 → 3.865 | 0.000704786 → 0.000704786 | 0 → 0 |
| PersonaMem | 8.696 → 5.012 | 5.937 → 2.830 | 0.001398814 → 0.001398814 | 1 → 1 |
| BEAM | 8.978 → 8.345 | 3.207 → 3.409 | 0.002681330 → 0.003060080 | 0 → 0 |
| CLBench | 26.154 → 20.382 | 3.970 → 3.034 | 0.005631746 → 0.004516046 | 1 → 1 |

## 外部调用阶段

| Benchmark | 阶段 | 秒：上轮 → 本轮 | credits：上轮 → 本轮 |
|---|---|---:|---:|
| LoCoMo | embedding | 1.450 → 2.778 | 0.000000300 → 0.000000300 |
| LoCoMo | directory_selection_jev | 1.534 → 0.794 | 0.000589554 → 0.000589554 |
| LoCoMo | answer | 2.768 → 2.136 | 0.000199500 → 0.000113100 |
| LoCoMo | judge | 1.383 → 2.339 | 0.000103350 → 0.000103350 |
| LongMemEval | embedding | 2.882 → 1.444 | 0.000000140 → 0.000000140 |
| LongMemEval | directory_selection_jev | 1.861 → 2.372 | 0.000547596 → 0.000547596 |
| LongMemEval | answer | 2.309 → 1.667 | 0.000055650 → 0.000055650 |
| LongMemEval | judge | 0.847 → 0.981 | 0.000101400 → 0.000101400 |
| PersonaMem | embedding | 1.876 → 1.666 | 0.000005320 → 0.000005320 |
| PersonaMem | directory_selection_jev | 4.015 → 1.126 | 0.000621894 → 0.000621894 |
| PersonaMem | answer | 2.758 → 2.181 | 0.000771600 → 0.000771600 |
| BEAM | embedding | 2.047 → 2.414 | 0.000000380 → 0.000000380 |
| BEAM | directory_selection_jev | 1.141 → 0.968 | 0.000523950 → 0.000523950 |
| BEAM | answer | 3.065 → 3.309 | 0.001971300 → 0.002348850 |
| BEAM | judge | 2.705 → 1.624 | 0.000185700 → 0.000186900 |
| CLBench | embedding | 1.886 → 1.593 | 0.000000620 → 0.000000620 |
| CLBench | directory_selection_jev | 2.050 → 1.408 | 0.000090426 → 0.000090426 |
| CLBench | answer | 15.963 → 14.157 | 0.002797650 → 0.001669650 |
| CLBench | judge | 6.219 → 3.187 | 0.002743050 → 0.002755350 |

阶段时间是客户端观察到的 API 请求耗时，包含网络、供应商排队、推理和可能的 SDK 内部重试。没有服务端分段计时，不能把变化直接归因于网络或模型推理。PersonaMem 使用本地选项匹配，无 Judge API。

## 上下文与模型结果是否一致

| Benchmark | 候选目录一致 | 入选目录一致 | 原文证据一致 | 回答完全一致 | 证据字符：上轮 → 本轮 | 向量读取：上轮 → 本轮 |
|---|---|---|---|---|---:|
| LoCoMo | 是 | 是 | 是 | 是 | 3567 → 3567 | 80 → 80 |
| LongMemEval | 是 | 是 | 是 | 是 | 0 → 0 | 53 → 53 |
| PersonaMem | 是 | 是 | 是 | 是 | 21718 → 21718 | 160 → 160 |
| BEAM | 是 | 否 | 否 | 否 | 54204 → 65381 | 66 → 113 |
| CLBench | 是 | 是 | 是 | 否 | 71680 → 71680 | 89 → 89 |

字符数用于解释是否给模型提供了不同上下文，不用于推算费用。Jev 的候选目录一致意味着相同存储摘要、路径、问题及选项；Answer 的证据一致且提示词哈希一致，可控制输入内容。回答变动会同时改变 Answer 输出和 Judge 输入。

## 费用与运行记录

成功查询合计：上轮 0.011309380，本轮 0.010486030 credits。本轮所有尝试已知费用 0.010486030，缺失费用调用 0 次。

[机器报告](results/vikingjev-directory-first-repeat.json)。全部逐调用、逐题、目录路由和向量工作量记录：

- LoCoMo：[本轮运行](../artifacts/vikingjev/directory-first-repeat-20261002/runs/20261002T082059052232Z-directory-first-locomo)。
- LongMemEval：[本轮运行](../artifacts/vikingjev/directory-first-repeat-20261002/runs/20261002T082103020489Z-directory-first-longmemeval)。
- PersonaMem：[本轮运行](../artifacts/vikingjev/directory-first-repeat-20261002/runs/20261002T082100154378Z-directory-first-personamem)。
- BEAM：[本轮运行](../artifacts/vikingjev/directory-first-repeat-20261002/runs/20261002T082059454697Z-directory-first-beam)。
- CLBench：[本轮运行](../artifacts/vikingjev/directory-first-repeat-20261002/runs/20261002T082100064629Z-directory-first-clbench)。
