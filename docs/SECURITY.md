# 🛡️ 安全规范
## 防线架构
| 层级 | 模块 | 目标 |
|------|------|------|
| 输入 | `InputSanitizer` | Prompt注入/越权/超长上下文 |
| 执行 | `ToolSandbox` | 超时熔断/异常隔离/危险API拦截 |
| 网络 | `RateLimiter` | IP限流/配额/慢攻击防护 |
| 记忆 | `MemoryGuard` | 奖励毒化/分布偏移/异常隔离 |

## 密钥管理
- 严禁硬编码，使用 `.env` 或 K8s Secrets
- 生产建议接入 HashiCorp Vault
- 漏洞报告: security@yourdomain.com (90天披露)
