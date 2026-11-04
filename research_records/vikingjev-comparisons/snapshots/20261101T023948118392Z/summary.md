---
title: "VikingJev 可比实验数据"
snapshot_id: "20261101T023948118392Z"
recorded_at: "2026-11-01T02:41:36.945939+00:00"
scores_captured_at: "2026-11-01T02:39:48.385849+00:00"
record_type: saved_same_question_comparison
new_evaluations: 0
---

本快照记录 **22 道题**、3 个分版本比较组。仅纳入 VikingJev、Mem0-OSS 和 MemoryBank 在同一道题上的有效保存成绩。各组按题等权汇总，指标均为越高越好。

这是已选 pilot 的完整交集，不代表完整基准。继承成绩按原始方法版本记录；各 VikingJev 版本分别列出。

## LoCoMo · vikingjev-repair-v3

N=20；指标：Token F1（0–1）。

| 方法 | 同题 N | 均值 |
| --- | ---: | ---: |
| VikingJev | 20 | 0.095532 |
| Mem0-OSS | 20 | 0.084521 |
| MemoryBank | 20 | 0.033398 |

VikingJev 相对 Mem0-OSS / MemoryBank 的同题平均差值：+0.011010 / +0.062133。

RAG 次参考：相同 20 题均值 0.160261；VikingJev 在这 20 题上为 0.095532。

历史范围：conv-26, conv-41。逐题分数见 [comparable-scores.csv](comparable-scores.csv)。

VikingJev 来源 run：`20261030T133627612820Z`。
有效协议 SHA-256：`22bf22f1d517b6bceba0d0c283696c7fb9321ad185da37311718dd40cd0dd1cc`。

## LongMemEval-S（cleaned） · vikingjev-repair-v3

N=1；指标：回答正确率（已保存的二元 judge 标签）（0–1）。

| 方法 | 同题 N | 均值 |
| --- | ---: | ---: |
| VikingJev | 1 | 1.000000 |
| Mem0-OSS | 1 | 1.000000 |
| MemoryBank | 1 | 1.000000 |

VikingJev 相对 Mem0-OSS / MemoryBank 的同题平均差值：+0.000000 / +0.000000。

RAG 次参考：相同 1 题均值 1.000000；VikingJev 在这 1 题上为 1.000000。

题目：`95bcc1c8`。这是一题的结果，不构成整体方法优势证据。

VikingJev 来源 run：`20261030T133627612820Z`。
有效协议 SHA-256：`22bf22f1d517b6bceba0d0c283696c7fb9321ad185da37311718dd40cd0dd1cc`。

## LongMemEval-S（cleaned） · vikingjev-source-ids-v1

N=1；指标：回答正确率（已保存的二元 judge 标签）（0–1）。

| 方法 | 同题 N | 均值 |
| --- | ---: | ---: |
| VikingJev | 1 | 1.000000 |
| Mem0-OSS | 1 | 1.000000 |
| MemoryBank | 1 | 0.000000 |

VikingJev 相对 Mem0-OSS / MemoryBank 的同题平均差值：+0.000000 / +1.000000。

RAG 次参考：相同 1 题均值 0.000000；VikingJev 在这 1 题上为 1.000000。

题目：`099778bb`。这是一题的结果，不构成整体方法优势证据。

VikingJev 来源 run：`20261101T014305123616Z`。
有效协议 SHA-256：`f4c8eed449f40b24fff231b0489baf61fddf78ba547ea6155cd64f2c4a003edf`。

## 选样与可比口径

LoCoMo 使用已固定的两段对话、20 道 development pilot 题。LongMemEval 的续跑子集来自已具备基线成绩且可复用进度的题目，并按剩余建库工作量选择；没有依据答案或分数替换题目。选样与恢复过程限制了代表性。不同协议分别报告，N=1 仅描述单题。

Token F1 使用已有 LoCoMo 官方评分函数结果；LongMemEval 使用已有官方 judge 提示产生并保存的 0/1 标签。这里仅聚合原分数。原始版本、评分类别处理与选样证据见版本溯源文件。

## 预算背景

全部历史实验的预算账，不是本表可比题目的成本；共享建库费用不按题均摊，缓存继承不重复记账。

账本快照：2026-11-01T02:39:51.800449+00:00；最后事件：2026-11-01T02:37:29.787479+00:00。

已确认费用 **$7.484028706**；未结算预留 **$5.432318464**；预算占用上界 **$12.916347170**。

未结算预留包括费用未知和在途请求；不是已确认扣款。账户可用余额未知。

Source-ID 阶段沿用原基线 $12.679101782 和绝对上限 $15.679101782。当前子 run 的起点 $12.770056540 是另一个账本边界，不重置阶段预算；本子 run 新增确认费用 $0.146290630，原阶段预算占用增量 $0.237245388。总预算仍为 $20。

## 记录与来源

[逐题 JSON](comparable-scores.json) · [逐题 CSV](comparable-scores.csv) · [汇总 JSON](summary.json)

每条分数保留原始 run、checkpoint、score stage、答案/评分 payload 哈希及源文件路径。基线的容器目录不替代其实际协议；历史协议哈希未保存时保留 unknown。

- comparable_scores：[文件](</Users/eversse/Documents/Codex/2026-10-18/task/experiment-records/vikingjev-comparisons/records/20261101T023948118392Z/comparable-scores.json>)；SHA-256 `bd47ecd58126210e88677f31687bce0156ded34832cd35d6acd5513f06349708`。
- costs：[文件](</Users/eversse/Documents/Codex/2026-10-18/task/experiment-records/vikingjev-comparisons/records/20261101T023948118392Z/costs.json>)；SHA-256 `bb1bc321ce6e04371c6919554a44ccb1baa466574070fa335a76b97a60a2f128`。
- version_provenance：[文件](</Users/eversse/Documents/Codex/2026-10-18/task/experiment-records/vikingjev-comparisons/records/20261101T023948118392Z/version-provenance.json>)；SHA-256 `00b34a9198d1871244bfe1e18ec183418ead9a41fd40ffa7c1f6e949a6ca7473`。
