# VikingJev

本分支在 Jevsight 的时间事实、归纳版本和来源关系上，增加目录化原文、目录摘要向量召回，以及读取原文前的 Jev 相关性判断。实验目标是验证小样本链路可运行，不重跑已有 benchmark，也不声称准确率或费用已有提升。

## 对照的 OpenViking 源码

参考版本：[`df32bf6e50a40843438f9491a26069ca4bd08f1f`](https://github.com/volcengine/OpenViking/tree/df32bf6e50a40843438f9491a26069ca4bd08f1f)。本次参考其设计并使用现有依赖实现，没有引入 OpenViking 运行时或复制其源码。

| 上游实现 | 本分支落地 |
| --- | --- |
| [`_SemanticMixin.write_context` / `abstract` / `overview`](https://github.com/volcengine/OpenViking/blob/df32bf6e50a40843438f9491a26069ca4bd08f1f/openviking/storage/viking_fs/_semantic.py)：目录内保存 `.abstract.md`、`.overview.md` 与完整内容 | `DirectoryMemory.build/publish` 生成每个写入批次的 L0/L1 文件，`messages.json` 保存完整原始消息。摘要复用已抽取事实，不额外请求生成模型。 |
| [`ContextLevel`、`Context.to_dict`](https://github.com/volcengine/OpenViking/blob/df32bf6e50a40843438f9491a26069ca4bd08f1f/openviking/core/context.py)：L0/L1/L2、URI、parent_uri、向量及来源元数据 | SQLite 注册目录的 `path/parent_path`、摘要、原文 SHA256；`message_sources` 绑定消息与目录；`embeddings.owner_type=directory` 单独保存目录概览向量。 |
| [`_AccessMixin._uri_to_path`](https://github.com/volcengine/OpenViking/blob/df32bf6e50a40843438f9491a26069ca4bd08f1f/openviking/storage/viking_fs/_access.py)：把逻辑命名空间映射到隔离存储路径 | 按用户／会话组织目录，外部 ID 经哈希后作为路径组件；原文读取验证所属用户、路径边界与内容哈希。没有新增公共 URI 接口。 |
| [`HierarchicalRetriever.retrieve`](https://github.com/volcengine/OpenViking/blob/df32bf6e50a40843438f9491a26069ca4bd08f1f/openviking/retrieve/hierarchical_retriever.py)：当前版本先做全局向量检索，再可选候选重排，支持目录与层级范围 | 在现有 RRF 中加入目录向量路线，再由 Jev 批量判断候选目录，随后读取入选原文。没有添加递归遍历或后台索引任务。 |

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

检索流程：问题向量 → 原有 BM25/chunk/fact/observation 召回加目录概览召回（50 条、RRF 权重 1.0）→ 最多 10 个候选目录 → 一次 Jev 请求、每目录一个 `relevant/irrelevant/uncertain` 判断 → 保留相关和不确定目录 → 20 个 seed、已接受目录内的邻居恢复 → SHA256 验证并读取原文 → 原有 16,000 token 证据预算。被判定无关的目录不会通过邻居恢复重新进入结果。

`/add`、`/search` 的请求响应结构保持不变，`/search` 返回原文证据。冒烟脚本把这些证据交给 `openai/gpt-4o-mini` 提取答案；服务没有新增 `/answer` 接口。归纳关系判断和目录筛选都用 `typesafe/jev-1.13`，属于两个独立阶段。

模型调用完成后才发布源目录并提交 SQLite 注册。提交失败会回滚表记录并移除本次发布的目录；进程在发布后、提交前崩溃时可能留下未注册目录，它不会参与检索。内容完全相同的重试可复用该目录，摘要变化则使用不同内容版本。源文件不可直接修改：缺失或哈希不符会报错，不回退到派生索引。新 schema 为 3，使用独立新索引，不迁移旧库；目录构建和选择器源码也纳入索引指纹。

## 运行

```bash
uv run --env-file .env aml-memory serve --config configs/service/vikingjev.yaml
uv run --env-file .env python -m scripts.vikingjev_smoke
```

配置复用环境中的 `OPENAI_API_KEY`；HTTP 鉴权使用 `AML_MEMORY_KEY`。冒烟脚本使用隔离的新索引、真实 OpenRouter 模型调用和进程内 HTTP 测试客户端。它保留逐次调用费用、耗时、目录判断和回答结果，不用 token 数推算费用；供应商未返回的费用明确标记缺失。已生成结果的输出目录不重复运行。
