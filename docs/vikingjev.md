# VikingJev

本分支在 Jevsight 上增加目录化原文与固定主题多归属。当前采用可配置阈值切换：当前用户启用的 chunk/fact/observation 向量总数不超过 300,000 时直接全量检索，超过才调用 Jev 判断上层主题。历史实验报告对应当时源码，不能解释为当前版本成绩。

## 对照的 OpenViking 源码

参考版本：[`df32bf6e50a40843438f9491a26069ca4bd08f1f`](https://github.com/volcengine/OpenViking/tree/df32bf6e50a40843438f9491a26069ca4bd08f1f)。本次参考其设计并使用现有依赖实现，没有引入 OpenViking 运行时或复制其源码。

| 上游实现 | 本分支落地 |
| --- | --- |
| [`_SemanticMixin.write_context` / `abstract` / `overview`](https://github.com/volcengine/OpenViking/blob/df32bf6e50a40843438f9491a26069ca4bd08f1f/openviking/storage/viking_fs/_semantic.py)：目录内保存 `.abstract.md`、`.overview.md` 与完整内容 | `DirectoryMemory.build/publish` 生成每个写入批次的 L0/L1 文件，`messages.json` 保存完整原始消息。摘要复用已抽取事实，不额外请求生成模型。 |
| [`ContextLevel`、`Context.to_dict`](https://github.com/volcengine/OpenViking/blob/df32bf6e50a40843438f9491a26069ca4bd08f1f/openviking/core/context.py)：L0/L1/L2、URI、parent_uri、向量及来源元数据 | SQLite 注册目录的 `path/parent_path`、摘要、原文 SHA256；`message_sources` 绑定消息与目录；`directory_topics` 保存主题归属与判断记录；当前不生成目录向量。 |
| [`_AccessMixin._uri_to_path`](https://github.com/volcengine/OpenViking/blob/df32bf6e50a40843438f9491a26069ca4bd08f1f/openviking/storage/viking_fs/_access.py)：把逻辑命名空间映射到隔离存储路径 | 按用户／会话组织目录，外部 ID 经哈希后作为路径组件；原文读取验证所属用户、路径边界与内容哈希。没有新增公共 URI 接口。 |
| [`HierarchicalRetriever.retrieve`](https://github.com/volcengine/OpenViking/blob/df32bf6e50a40843438f9491a26069ca4bd08f1f/openviking/retrieve/hierarchical_retriever.py)：当前版本先做全局向量检索，再可选候选重排，支持目录与层级范围 | 超过规模阈值时，对固定主题摘要进行 Jev 判断，再限定 chunk、fact、observation 和 BM25 的检索范围。没有添加递归遍历或后台索引任务。 |

## 存储和调用路径

```text
data/vikingjev/
├── <user-hash>.sqlite3          # 向量、FTS、事实、归纳、目录注册与来源关系
└── user/<user-hash>/memories/sessions/<session-hash>/<content-version>/
    ├── .abstract.md            # L0：最多 256 token 的摘要
    ├── .overview.md            # L1：时间与事实概览，最多 1024 token 加截断提示
    └── messages.json           # L2：完整原始消息、角色、来源顺序、时间戳
```

目录索引面向不可变的写入批次。父目录用于用户和会话组织；本轮没有把所有父节点再生成一遍摘要，也没有自动给记忆分类。源版本由原文与摘要共同确定，系统写入时间不用于比较事实新旧。

SQLite 的 `messages.content` 和 `chunks.content` 在目录模式下留空。FTS 保留供检索使用的派生文本索引；最终证据与同会话抽取上下文从源文件读取，不拿事实摘要代替原文。

写入流程：既有事实抽取与归纳，加一次 Jev 多主题分类，输入本批原始消息与固定主题描述。主题为教育、工作、健康、旅行、财务、家庭关系、爱好、居住和其他。允许多个相关或不确定主题，分类结果与概率存入 directory_topics，与原文注册、事实、向量一起原子提交；分类失败或未归入任何主题直接报错。重复 Add 不重复分类。

配置 `retrieval.directory_routing_threshold: 300000`（非负整数）。统计仅当前用户且当前启用的向量路线，不统计主题或目录，也不因查询有选项而重复计数。小于或等于阈值：全量混合检索，不调用查询 Jev；超过阈值：只对有归属记忆的固定主题描述调用一次 Jev，再将相关/不确定概率 ≥0.5 的主题展开成去重目录集合，SQL 在读取向量前限定范围。广泛问题可选择全部主题，不固定 Top-4；全部不相关返回空证据，模型失败报错。两条路径都保留原有 RRF、20 个 seed、原文证据和 16,000 token 预算。阈值是部署初值，不是实验确认的通用最优点；修改阈值不改变索引指纹。

上层是固定主题描述，不是动态生成的用户事实摘要。跨主题分类与查询仍可能漏召回；没有加入自动重试或全量回退。来源目录继续按用户/会话保存，通过多对多关系实现主题层，不复制原文。

`/add`、`/search` 的请求响应结构保持不变，`/search` 返回原文证据。冒烟脚本把这些证据交给 `openai/gpt-4o-mini` 提取答案；服务没有新增 `/answer` 接口。归纳关系判断和目录筛选都用 `typesafe/jev-1.13`，属于两个独立阶段。

模型调用完成后才发布源目录并提交 SQLite 注册。提交失败会回滚表记录并移除本次发布的目录；进程在发布后、提交前崩溃时可能留下未注册目录，它不会参与检索。内容完全相同的重试可复用该目录，摘要变化则使用不同内容版本。源文件不可直接修改：缺失或哈希不符会报错，不回退到派生索引。新 schema 为 4，使用独立新索引，不迁移旧库；目录构建、选择器及主题定义源码纳入索引指纹。

## 运行

```bash
uv run --env-file .env aml-memory serve --config configs/service/vikingjev.yaml
uv run --env-file .env python -m scripts.vikingjev_smoke
```

配置复用环境中的 `OPENAI_API_KEY`；HTTP 鉴权使用 `AML_MEMORY_KEY`。冒烟脚本使用隔离的新索引、真实 OpenRouter 模型调用和进程内 HTTP 测试客户端。它保留逐次调用费用、耗时、目录判断和回答结果，不用 token 数推算费用；供应商未返回的费用明确标记缺失。已生成结果的输出目录不重复运行。

## 冻结索引检索对照

`aml-memory evaluate --config <evaluation.yaml> --reuse-index-from <prior-run>` 会复制已有运行的完整索引，校验数据、题目、模型及评分口径，并通过原有索引指纹和 Add 幂等校验验证载荷。请求缺失或不匹配会失败，不补写新记忆。报告标记 `index_mode=frozen-reuse`，原始索引不变，幂等跳过不计首次写入成本。该模式用于只修改检索的实验，具体五题对照和限制见 [结果报告](vikingjev-directory-first-results.md)。

## 阈值版验证

新配置使用 data/vikingjev-topics 独立索引路径。旧 schema 需从原始消息重建，不迁移旧库；既有冻结索引不可直接用于新版本评测。

[真实 API 小冒烟](../artifacts/vikingjev/threshold-smoke/results.json)验证一次写入分类、阈值下全量检索和阈值上主题检索都能回溯正确原文；不是五套 benchmark 重评。新增分类阶段计费标签 directory_classification_jev；查询目录判断仍为 directory_selection_jev。
