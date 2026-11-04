# 可比实验的版本与抽样来源

记录时间：2026-11-01T02:27:34.326488+00:00。仅读取既有 manifest、冻结源码、选题计划和结果验证记录；未读账本、查询进程、调用模型或修改实验。

本记录只关联三种主方法均已有有效同题结果的 **22 题**：LoCoMo20 题，LongMemEval-S(cleaned) 的95bcc1c8及099778bb。不同benchmark分别报告，不混合指标。没有导出失败或缺失结果表。

|可比数据|VikingJev版本及run|基线来源|
|---|---|---|
|LoCoMo，conv-41/conv-26各10题|repair-v3，20261030T133627612820Z|原实验成功结果经兼容继承，续跑未重评基线|
|LME95bcc1c8，single-session-user|repair-v3，20261030T133627612820Z|同题Mem0-OSS/MemoryBank已有结果|
|LME099778bb，multi-session|source-ids-v1，20261101T014305123616Z|同题Mem0-OSS/MemoryBank已有结果|

基线实现版本来自原冻结manifest：

- mem0：`mem0ai/mem0`，commit `07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4`，archive SHA256 `90688805131f4b811eba20d968556d9a1c88535361f0a9c3001600d8388d2183`。
- memorybank：`zhongwanjun/MemoryBank-SiliconFriend`，commit `cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9`，archive SHA256未记录（JSON为null）。
- 回答模型 `openai/gpt-4o-mini`，judge模型 `openai/gpt-4o-2024-08-06`；此处记录既有配置，不发起新评测。

## 版本关系

- 原冻结run 20261016T011008129863Z：4方法×14 histories=56个建库任务，4方法×32题=128个QA任务。数字是原计划规模，不等于本次有效比较数量。
- repair-v3（20261030T133627612820Z）：撤销本地Jev 32000/拟64000字节拒绝门槛，改由真实供应商限制及自动长上下文处理；非法输出最多额外重生成3次，严格来源验证保留。成功前缀/缓存继承，非从零重跑。
- source-ids-v1（20261101T014305123616Z）：抽取由数字区间端点改成显式词/标点来源ID集合，prompt/schema表示随版本记录。迁移证明继承67个有效单元（099为46、另一目标21）及10个成功抽取缓存（099为7、另一目标3）。可比099使用继承46单元再续建的混合来源库，不可称为全v1从零复现。
- output-v2（20261101T021307599847Z）仅作版本背景：505专用，继承24个有效单元与1个成功旧ID请求；新schema增加maxItems=N，抽取额度16384，其他complete/rewrite仍4096。旧缓存按旧schema/4096严格回放。该版本没有加入本记录的用户比较表，不据启动或HTTP成功声明结果。

## 选题与解释边界

- 原pilot使用seed42、题型分层和来源索引抽样，选择文件明确不按答案/表现选样；development_exposure=true。原样本为LoCoMo20+LME12，共14 histories，不是完整benchmark。
- LoCoMo20沿用原固定20题。只续跑VikingJev并读取相同题目的既有基线结果；两个conversation的小样本不支持整体能力结论。
- LME扩展事先从三基线共同完成的5题中，按剩余建库单元、剩余source tokens、完整history tokens、稳定题ID依次升序选3题。选题算法不读取答案或模型分数，题型覆盖是该固定规则所得。其方便抽样与已有基线覆盖筛选均有偏差。
- 后续source-ID修复按先前建库问题选中相同历史，未按评分换题；当前仅保留有效共同题比较，仍存在恢复选择和完成条件带来的偏差。LME两题跨v3/v1，合并只能标注为混合版本探索结果。
- 继承成功缓存/前缀不算新的独立重复实验；未证明随机重复次数、置信区间或完整基准外推，JSON对应字段为null。记忆更新、一致性或成本效率不能由这些问答结果推断。

## 证据文件与SHA256

逐版本protocol SHA、冻结源文件校验和完整来源列表均在同目录version-provenance.json中。以下为主要证据：

- `/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261016T011008129863Z/manifest.json`
  SHA256 `c89bf7cbeed6f2617ed3bca2691edb887fc1c9c48e198ee788d1de81b1375e65`
- `/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261016T011008129863Z/selection.json`
  SHA256 `f87197293993ef41943ce70a45f93a12a6249599710d0039d4f7b5c920346191`
- `/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261016T011008129863Z/data-manifest.json`
  SHA256 `e560c79f059560d7256af0ad03cde143830f5bfe8a79de2d110c648c5a216d85`
- `/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261030T133627612820Z/manifest.json`
  SHA256 `1824dbbf56737e6b2fd5e3af1ddc5f1f09196d1b6f10527371178bdb4a85a67f`
- `/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261101T014305123616Z/manifest.json`
  SHA256 `40ef032d00af402e9b571a6eb9ea29b489e8513044970b6920394b42ab7b7597`
- `/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261101T021307599847Z/manifest.json`
  SHA256 `cf8151846c964ace8a71bcb60fec38231daf2aee8d59e281038ba385abd63a26`
- `/Users/eversse/Documents/Codex/2026-10-18/task/monitoring/source-id-repair-20261101/child-plan.json`
  SHA256 `db3e1445c0aa3a16fe5a714fe7ec9242b1efa6071ee95ea5357cee9643256082`
- `/Users/eversse/Documents/Codex/2026-10-18/task/monitoring/source-id-output-limit-20261101/child-plan.json`
  SHA256 `5987e009d901827cd3b612e621b2513fc8fa4bb32147550c7509ee580f22e97f`
- `/Users/eversse/Documents/Codex/2026-10-18/task/monitoring/lme-expansion-20261031/scope-budget.json`
  SHA256 `76aa9e5ebbca0f656ad16ae85bb60a5870be6b1367a862d6bfdef2c261bd2d90`
- `/Users/eversse/Documents/Codex/2026-10-18/task/monitoring/ours-resume-20261031/FINAL-RESULTS.md`
  SHA256 `6706a836a0d65355cabb44d472a63ed8f474844a8adcc6c2f51174c6c1404dbc`
- `/Users/eversse/Documents/Codex/2026-10-18/task/monitoring/lme-expansion-20261031/FINAL-RESULTS.md`
  SHA256 `80f34374a7dc4cb73b9420e56cc2ec7cc2374470c9747221ec8a404897d36ef9`
- `/Users/eversse/Documents/Codex/2026-10-18/task/monitoring/source-id-repair-20261101/FINAL-RESULTS.md`
  SHA256 `2b6714a697ad5777b8d701a779401ac4307e31bcf17aed4c48682c56fc26d7c0`

本文件是workspace待发布记录，最终原项目research_records由父任务发布；没有改写原实验产物或当前运行。
