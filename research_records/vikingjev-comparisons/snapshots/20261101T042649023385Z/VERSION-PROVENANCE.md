# 可比实验的版本与抽样来源

记录时间：2026-11-01T04:27:37.346094+00:00。
仅记录有效同题交集，共 23 组、92 条成绩；RAG 为次参考。

| 数据集 | VikingJev 版本 | 同题数 |
| --- | --- | ---: |
| locomo | vikingjev-repair-v3 | 20 |
| longmemeval | vikingjev-optional-fact-text-v4 | 1 |
| longmemeval | vikingjev-repair-v3 | 1 |
| longmemeval | vikingjev-source-ids-v1 | 1 |

既有版本、基线实现、抽样与官方评分来源保留在 version-provenance.json；各版本和指标分别汇总。

v4 继承 44/53 个有效建库单元和 8 个已付费缓存操作，并继续完成同一历史；这是混合来源的修复版本。
ord7 使用最早已付费 base 响应；4 个 fact 的 selected 来源证据已逐条审查，8 个空白 conditions/verification 字段转为 null。null 仅表示未提供有效结构化文本，不断言现实中没有条件或已完成验证。其他非空字段、事实内容、来源、时间和层级保持不变。
新请求只为这两个 nullable 字段加非空白模式；已有付费响应以原 wire/schema 免费回放，保留原始字节和已消耗重试。
继承祖先 v3 的 ord11 从原四次付费响应按顺序选择 attempt1；7 个 fact 经人工 selected 来源核验，保序去重 4 个重复 ID。两处选择均未参考 QA 答案或得分。
全部原始 ID 先校验，来源映射及不同 ID 均保留；源文本、prompt、模型、完整输入和抽取上限 16384 保持不变。恢复时的响应选择、归一化与人工源证据审查属于此版本条件，不构成原协议或 first-pass 复现。
505 为恢复选择的一题，按 v4 独立列组；不能据此推断完整基准表现。

只聚合保存的有效评分，不重评基线；完整证据路径、哈希及审查绑定见同目录 JSON。
