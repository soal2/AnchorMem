---
title: "VikingJev 可比记录索引"
snapshot_id: "20261101T042649023385Z"
recorded_at: "2026-11-01T04:27:45.832213+00:00"
scores_captured_at: "2026-11-01T04:26:49.321177+00:00"
record_type: saved_same_question_comparison
new_evaluations: 0
---

本目录是一个追加式快照，只记录同题有效交集。首选阅读 [summary.md](summary.md)。

| 文件 | 内容 |
| --- | --- |
| [comparable-scores.csv](comparable-scores.csv) | 逐题逐方法分数与原始来源 |
| [paired-comparisons.csv](paired-comparisons.csv) | 每题每版本一行，三主方法与同题 RAG 宽表 |
| [comparable-scores.json](comparable-scores.json) | 同题配对、完整溯源与证据哈希 |
| [summary.json](summary.json) | 分数据集、分协议的均值与同题差值 |
| [summary.md](summary.md) | 中文摘要、选样口径和预算背景 |

主比较为 VikingJev / Mem0-OSS / MemoryBank；RAG 仅作相同题目的次参考。各 VikingJev 版本分别汇总。全部值来自已有实验产物，无新评分、无模型调用、无原实验写入。

快照不覆盖已有记录。后续新结果需生成新目录；已有结果的来源及输入 SHA-256 保存在 summary.json。
