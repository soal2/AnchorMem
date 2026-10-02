# Jevsight 与当前目录 Top-4 检索对比

未新增任何 benchmark 调用。旧结果取 Jevsight D 组完整开发运行中的同题记录，新结果取全目录 Top-4 冒烟。两次建索引、源码版本和调用时间不同；这是历史同题对照，不是只改变目录变量的受控实验。

| Benchmark | 旧→新结果 | Search 秒旧→新 | 查询 credits 旧→新 |
|---|---|---:|---:|
| locomo | WRONG → CORRECT | 0.823 → 4.685 | 0.000551700 → 0.002977128 |
| longmemeval | CORRECT → CORRECT | 0.848 → 4.896 | 0.001390190 → 0.004434788 |
| personamem | CORRECT → CORRECT | 0.728 → 3.670 | 0.001239070 → 0.001697764 |
| beam | WRONG → WRONG | 0.592 → 3.180 | 0.002186780 → 0.003033230 |
| clbench | CORRECT → CORRECT | 0.882 → 2.821 | 0.005584070 → 0.003175496 |

查询费用包含 Search、Answer、Judge；费用来自 usage.cost。Search 时间不包含 Answer/Judge。

## 同一新索引下的向量读取工作量

全量列根据 Jevsight 全量读取逻辑与新索引的向量条目数计算，不是重跑旧分支的结果；目录 Top-4 列为本轮实际读取。PersonaMem 包含两个查询视角。

| Benchmark | 全量逻辑 | 当前目录范围 | 减少 |
|---|---:|---:|---:|
| locomo | 1103 | 136 | 87.7% |
| longmemeval | 1242 | 83 | 93.3% |
| personamem | 660 | 298 | 54.8% |
| beam | 460 | 195 | 57.6% |
| clbench | 88 | 88 | 0.0% |

## 代码差异与结论

- Jevsight 的 Jev 仅参与写入归纳关系判断；查询阶段全量读取 chunk/fact/observation 向量，并读取全部 chunk 正文恢复邻居。
- 当前版本查询全部目录摘要，分批调用 Jev，按概率选前四目录；SQL 在加载向量之前限定范围，原文仅从入选证据所属源文件读取并校验哈希。按文件读取可能包含该文件的其他消息，不是逐 chunk 随机读取。
- 两者沿用时间信息、知识归纳、多路 RRF 和原文证据；新增优势是检索范围控制与原文按需读取。
- 当前五题历史对照中 LoCoMo 错变对，其余结果相同；不能由不同建索引的同题对照推出目录导致准确率提升。
- 五题 Search 均变慢；当前数据规模下，本地向量处理原本为毫秒级，Jev API 引入秒级开销。工作量减少不能等同端到端提速。
- 目录本身不降低 API 单价；查询新增 Jev 费用，最终费用还受证据、生成输出和缓存影响。
- 当前仍判断全部目录，不是按层级树逐级剪枝，不具有 O(log n) 保证。

来源：[旧 D 组结果](../artifacts/jevsight/available-results.json)、[当前五题结果](results/vikingjev-directory-all-top4.json)。
