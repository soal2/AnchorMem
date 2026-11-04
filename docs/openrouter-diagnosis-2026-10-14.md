# OpenRouter 地区错误排查

2026-10-14 按用户要求核查 OpenRouter 官方文档、免费账户接口、模型端点和官方状态页。此次仅发送免费 GET 请求，没有模型推理请求，实验累计费用仍为 $0.00079174。

已确认错误类别为模型地域访问限制。三次固定模型调用返回 HTTP403，其中两次留有明确原始消息：`This model is not available in your region.`。沿实验网关相同网络路径读取 OpenRouter 的 Cloudflare 边缘诊断，得到来源国家 `loc=CN`。该结果与地域限制一致，但公开接口无法确认拦截发生在 OpenRouter 还是供应商层，也没有说明是否同时使用账户或账单地区。

| 检查 | 实测结果 |
|---|---|
| 密钥 | `/api/v1/key` 返回200；非免费账户 |
| 账户 credits | `/api/v1/credits` 返回200；累计credits 18美元、累计使用12.204030462美元，剩余 **5.795969538美元** |
| 密钥限额余量 | **87.795969538美元**，表示密钥100美元使用上限的剩余部分；这是额度限制，不是账户可支付余额 |
| 模型端点 | `gpt-4o-mini` 仍列出 OpenAI、Azure 和 Azure Sweden；三个端点最近5分钟及30分钟 uptime 均为100% |
| 官方平台状态 | Chat、Embeddings 为operational，活动事故为空；当前没有报告平台级事故 |
| 来源识别 | 同一网络路径的边缘诊断 `loc=CN`；`colo=LAX` 是处理请求的边缘节点位置，不能代替来源国家 |
| 环境代理 | 当前进程未设置 HTTP(S)/ALL_PROXY 环境变量；没有证据表明实验网关忽略既有环境代理导致本次错误 |

接口数据与核查时间保存在[诊断 JSON](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/openrouter-diagnosis.json)，官方状态原始响应保存在[状态快照](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/openrouter-status.json)。公开端点健康和平台状态不能保证特定地区或账户有模型访问权限。

[OpenRouter 条款](https://openrouter.ai/terms)第5.7节规定模型供应商的地区访问限制同样适用于通过 OpenRouter 调用；第6.8节说明会向模型传递请求来源国家，并承认来源国家可能被错误识别。[OpenAI API 支持国家名单](https://help.openai.com/en/articles/5347006-openai-api-supported-countries-and-territories)当前不包含中国大陆。

[官方错误文档](https://openrouter.ai/docs/api_reference/errors-and-debugging)区分了401认证错误、402 credits不足，以及权限、组织预算或内容规则导致的403。因此不能仅凭403判断原因；本次依据的是明确的region错误文本，加上来源识别和账户、模型状态的交叉核查。当前公开文档没有说明这条具体地域错误的内部判定分支。

在来源识别和访问权限保持现状时，重复请求不能解决这一限制。如果实际符合模型访问条件而被错误识别，可通过[OpenRouter 官方工单入口](https://openrouter.ai/support/new)要求核查。工单应提供模型ID `openai/gpt-4o-mini`、最近失败时间 `2026-10-14T12:56:31.163625Z`（北京时间20:56:31）、上述错误文本和边缘诊断 `loc=CN`，请其确认来源国家判定、账户地区影响和执行限制的层级。此次未创建或发送工单。

第一周20美元是已授权实验花费上限；目前账户可支付 credits 约5.80美元。地域问题解决后仍需按实际 credits 与实验费用门禁执行，账户额度与实验授权预算分别记录。

## VPN 接入后的复核

用户要求使用已有 VPN 后，发现 FlClash 已开启 macOS 系统代理 `127.0.0.1:7890`。此前只检查了环境变量代理；HTTPX 不会自动读取 macOS 的系统代理，而实验网关显式创建 HTTPTransport，因此此前走了直连。这是网关网络接入的遗漏。

以标准库 `urllib.request.getproxies()` 读取现有系统 HTTPS 代理并通过它查询同一个免费诊断接口，来源识别变为 `MY`，边缘节点为 `KUL`。随后已修正 Gateway，让外部 HTTPS 请求使用现有代理并尊重代理排除规则；本地 JSONL 子进程到预算网关的连接仍走本机回环。

固定模型 `openai/gpt-4o-mini` 已返回成功真实响应，实验已使用原题单、原作者源码和同一20美元费用账本续跑。RAG 与 Mem0 已分别完成首题官方 F1 评分，均为0，结果原样保留。代理通道证据见[连接记录](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/vpn-connection.json)；针对代理和预算、续跑的离线验证见[测试记录](/Users/eversse/Documents/科研/llm_mem/AnchorMem/artifacts/week1/20261014T085344Z/vpn-test-results.txt)。当前结果与费用以运行目录和共享 API 账本为准。
