# 第一周当前方法抽取协议修正提案

执行结果（2026-10-15）：已按批准配置从完整 LoCoMo 历史重建，实际模型输入增加 103 token，确认追加段落已生效；模型仍在同一批误引旧上下文，严格来源校验正确拒绝。失败与费用保留，详见本次运行 `approved-prompt-outcome-2026-10-15.json`。按本提案停止进一步修改提示词和扩大题单，继续首批剩余独立链路验证。

状态：已获用户批准并冻结。用户明确选择“允许修正提示词并继续（推荐）”；2026-10-15 01:59:18 UTC 已创建仅供论文实验使用的提示词副本及协议收据。本记录不表示新提示词已完成真实实验。

## 已复现的问题

2026-10-15 按原模型、原提示词和原始完整历史续跑后，当前方法在 LoCoMo `conv-41` 的 `session_2` 第二批消息抽取失败。模型返回的引文为：

> I donated my old car to a homeless shelter I volunteer at yesterday.

该引文位于同一 session 的上一批历史。模型却把它标为本批 `new_chunks` 的 `index=0`，引发 `fact source quote occurrence is absent from its chunk`。输出 JSON 完整，`finish_reason=stop`。代码和第二次独立检查均确认新消息索引映射正确，严格校验正确拒绝了旧上下文引文。原文及校验信息保存在本次运行 `failures.jsonl`。

## 建议的最小修正

仅为论文 pilot 的当前方法创建独立抽取提示词副本，保留现有正文并追加以下段落：

```text
Before returning the facts array, check every source_quotes item against new_chunks[index].content. The quote must occur literally in that exact current chunk. Text found only in prior_messages cannot support a returned entry, even if it describes a useful fact. Do not carry an earlier entry forward unless a current new_chunk repeats its supporting statement. Omit an entry if no current new_chunk contains its exact supporting quote. If no entry has a valid current source, return {"facts": []}. prior_messages provide reference context only.
```

严格来源校验保留，不将无效引文自动映射到旧消息，不截断或跳过任何历史批次，不通过多次抽样挑选成功答案。生成模型、温度、输出上限、向量、题单、读取器、证据预算和三个 baseline 源码/参数不变。不新增内容重试。

批准后已冻结提示词副本和协议收据；续跑配置和源码快照由实验运行入口记录。原失败与费用继续保留。当前方法尚无答案或评分，需保留失败索引并从同一完整历史重建；三个 baseline 已完成的索引、答案及评分继续复用。本次修正依据建库格式错误提出，不参考题目答案或方法得分。

所有后续调用仍计入同一账本，实验预算上限仍为 20 美元。修正后仍须先通过两个 benchmark 各一道题的四系统链路，再扩展到完整 pilot。若再次出现无效来源，保留失败并停止扩大实验。

## 已冻结的协议修正

- 协议收据：`artifacts/week1/20261014T085344Z/extraction-prompt-amendment.json`。
- 原提示词：`artifacts/week1/20261014T085344Z/prompts/extract-temporal-facts.txt`，SHA256 为 `f1d26f38343fe86aa35cf78ad20ea33ee654aee29d6a2831c14008185df64538`，与项目根目录原提示词的 bytes 一致。
- 独立副本：`artifacts/week1/20261014T085344Z/prompt-amendment/extract-temporal-facts.txt`，SHA256 为 `2df7ad7a1a05dffb3b80acf148a74c7c7654e76a774786a0c2f14b564c505c4c`。

副本保留原文件全部 bytes，仅追加两次换行、上方英文检查段落及末尾换行。项目主提示词和三个 baseline 的作者源码不变。此修正由用户批准，依据当前 chunk 来源校验失败提出；当前方法批准前已评分数为 0，不是根据 QA 答案或评分调参。收据同时固定不新增内容重试、预算上限 20 美元。
