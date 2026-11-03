# MemoryBank 论文评测集（benchmark）

工作副本：`/Volumes/VikingJevWork/AnchorMem`（T7 sparsebundle `VikingJev-workspace.sparsebundle`）。
分支：`memorybank-benchmark`。未修改冻结归档 / `canonical.pax.tar` / readable 树。

## 数据来源

- 论文：arXiv:2305.10250
- 仓库：`zhongwanjun/MemoryBank-SiliconFriend` @ `cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9`
- 归档 SHA256：`55f95fab5c4718a68c0af1fba34856f2b3ce874a51d87b0ced2cb57ad52eb702`
- 评测文件复制到：`data/week1/datasets/memorybank/{en,cn}/`（与 `data/week1/memorybank/source/eval_data` 字节一致）
- 溯源：`data/week1/datasets/memorybank/SOURCE.json`

## 规模（离线实测）

| 项 | 值 |
|---|---:|
| 语言 | en + cn |
| 用户 / 语言 | 15 |
| 日历对话天 / 用户 | 10 |
| 历史数 | 30 |
| probing 题 | 200（en 100 + cn 100） |
| 源消息（全部历史） | 2264 |
| 源 tokens（tiktoken gpt-4o-mini） | 105819 |

论文正文写约 194 题；冻结仓库文件为每语言 100 题共 200 题。`en:John Zhang` 的 `2023-05-05` 为空对话日，归一化时跳过该 session，但仍记录在 `empty_dates`。

## 费用估算（仅询价，不授权实验）

依据 week-one `cost-basis` 的 LoCoMo 短对话费率 + 保守证据型 judge：

| 方法 | 建库 | QA | Judge（保守） | 合计 USD |
|---|---:|---:|---:|---:|
| VikingJev | 1.883755 | 0.042084 | 0.2 | 2.125839 |
| Mem0 | 0.128307 | 0.028265 | 0.2 | 0.356571 |
| MemoryBank 方法 | 0.081936 | 0.023372 | 0.2 | 0.305308 |

VikingJev fallback `$0.0000181`/source token 建库：1.915324。
机器可读：`data/week1/datasets/memorybank/cost-estimate.json`。**未改 $20 预算上限，未发起付费调用。**

## 评分

- 模块：`src/week1_memorybank_benchmark.py`
- Scorer：`data/week1/scorers/memorybank_evaluate_qa.py`
- 与 LongMemEval 相同的 yes/no 解析（`parse_memorybank_judge`）
- 官方参考答案未发布（GitHub issue #22）；缺金标时 judge 使用该用户完整对话证据
- 论文人工指标（检索准确率、回答正确性 0/0.5/1、连贯性、排名）未复现

## 开放问题

- 评分主观性：二元 LLM-judge 只是论文人工量表的代理
- zh/en 并行语料，不保证题意/答案一一对齐
- 英文 memory bank 用户名含空格，loader 已 strip 后与 probing 对齐
- 尚未接入付费 `week1_runner` / EvaluationConfig；仅离线加载与单测
