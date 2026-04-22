# 外部 API 调用指南

## 概述
由于云端环境存储限制 (仅 504MB)，本项目设计为支持**完全外部 API 调用**模式，无需在本地安装大型模型库 (如 transformers, vllm, peft 等)。

## 核心架构

### 1. LLMPlanner - 已支持外部 API
文件：`src/llm_planner.py`

```python
from openai import OpenAI

class LLMPlanner:
    def __init__(self, config: dict):
        self.client = OpenAI(
            base_url=config.get("base_url", "http://localhost:8000/v1"),
            api_key=config.get("api_key", "mock"),
            timeout=config.get("timeout", 15.0),
        )
```

**特点**：
- ✅ 已使用 OpenAI 兼容接口
- ✅ 支持任意 OpenAI 兼容服务 (DeepSeek/Qwen/Ollama/vLLM)
- ✅ 内置启发式降级机制 (LLM 失败时自动切换)

### 2. LangGraph 集成 - 轻量级工具调用
文件：`src/yogacara_langgraph.py`

**工具函数**：
```python
@tool
def query_knowledge_base(query: str) -> str:
    """查询知识库"""
    
@tool  
def call_external_api(endpoint: str, payload: dict) -> dict:
    """调用外部 API"""
    
@tool
def calculate_metric(metric_name: str, values: list[float]) -> float:
    """计算指标"""
```

## 配置方案

### 方案 A：使用公有云 API (推荐快速开始)

#### 1. DeepSeek (性价比最高)
```bash
export DEEPSEEK_API_KEY="sk-xxxxx"
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"
```

**价格**：¥0.27/百万 tokens (输入) | ¥1.12/百万 tokens (输出)

#### 2. Qwen (通义千问)
```bash
export DASHSCOPE_API_KEY="sk-xxxxx"
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen-plus"
```

**价格**：¥0.4/百万 tokens (输入) | ¥1.2/百万 tokens (输出)

#### 3. OpenAI GPT-4o
```bash
export OPENAI_API_KEY="sk-xxxxx"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
```

**价格**：$0.15/百万 tokens (输入) | $0.60/百万 tokens (输出)

### 方案 B：自建私有 API 服务

#### 使用 Ollama (本地/远程服务器)
```bash
# 在远程服务器部署
ollama serve
ollama pull qwen2.5:7b

# 本地配置
export LLM_BASE_URL="http://YOUR_SERVER_IP:11434/v1"
export LLM_API_KEY="ollama"
export LLM_MODEL="qwen2.5:7b"
```

#### 使用 vLLM (高性能推理)
```bash
# 远程服务器部署
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000

# 本地配置
export LLM_BASE_URL="http://YOUR_SERVER_IP:8000/v1"
export LLM_API_KEY="vllm"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

## 使用示例

### 1. 基础运行 (使用外部 API)
```python
import os
from src.llm_planner import LLMPlanner

config = {
    "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    "api_key": os.getenv("DEEPSEEK_API_KEY", "your-key"),
    "model": os.getenv("LLM_MODEL", "deepseek-chat"),
    "temperature": 0.3,
    "use_fallback": True  # 启用降级保护
}

planner = LLMPlanner(config)

obs = {"pos": (5, 5), "grid_view": [0, 0, 0, 0, 0, 0, 0, 0, 0]}
seeds = []

action, uncertainty, causal, tools = planner.plan(obs, seeds)
print(f"动作：{action}, 不确定性：{uncertainty:.2f}")
```

### 2. LangGraph 完整流程
```python
import asyncio
from src.yogacara_langgraph import create_session, build_graph

async def run_with_external_api():
    # 创建隔离会话
    session = create_session()
    env = session["env"]
    alaya = session["alaya"]
    manas = session["manas"]
    
    # 构建图
    graph = build_graph()
    
    # 初始状态
    init_state = {
        "obs": env.reset(),
        "action": "",
        "reward": 0.0,
        "done": False,
        "step": 0,
        "seeds": [],
        "unc": 0.0,
        "manas_passed": True,
        "tool_calls": [],
        "recent_rewards": [],
        "pos_history": [],
        "metrics": {},
    }
    
    # 运行
    final_state = await graph.ainvoke(init_state)
    print(f"完成！步数:{final_state['step']}, 奖励:{sum(final_state['recent_rewards']):.2f}")

asyncio.run(run_with_external_api())
```

### 3. API Server 部署 (FastAPI)
```python
# src/api_server.py 已实现
from fastapi import FastAPI
from src.llm_planner import LLMPlanner
import os

app = FastAPI()

planner = LLMPlanner({
    "base_url": os.getenv("LLM_BASE_URL"),
    "api_key": os.getenv("LLM_API_KEY"),
    "model": os.getenv("LLM_MODEL"),
})

@app.post("/plan")
async def plan_endpoint(obs: dict, seeds: list):
    action, unc, causal, tools = planner.plan(obs, seeds)
    return {"action": action, "uncertainty": unc, "causal": causal, "tools": tools}
```

启动：
```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

## 环境变量配置 (.env 文件)

创建 `.env` 文件：
```bash
# LLM 配置
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-your-api-key
LLM_MODEL=deepseek-chat
LLM_TIMEOUT=15.0
LLM_TEMPERATURE=0.3

# 可选：Milvus 向量数据库 (也可用 FAISS 替代)
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 安全限流
RATE_LIMIT_PER_MINUTE=60
```

加载：
```python
from dotenv import load_dotenv
load_dotenv()
```

## 成本估算

以典型实验为例 (100 次实验 × 平均 30 步/实验)：

| 服务商 | 模型 | 单次调用 | 总成本 |
|--------|------|----------|--------|
| DeepSeek | deepseek-chat | ~¥0.003 | ¥9 |
| Qwen | qwen-plus | ~¥0.004 | ¥12 |
| GPT-4o-mini | gpt-4o-mini | ~$0.001 | $11 |
| Ollama | qwen2.5:7b | ¥0 (自建) | ¥0 (电费) |

**注**：每次调用约 100-200 tokens (输入+输出)

## 降级保护机制

当外部 API 不可用时，系统自动切换到启发式算法：

```python
# LLMPlanner 内置降级
try:
    response = self.client.chat.completions.create(...)
except Exception as e:
    logger.warning(f"LLM 规划失败：{e}，启用启发式降级")
    return self._heuristic_fallback(obs, seeds)  # 无依赖纯算法
```

**启发式降级特点**：
- ✅ 零外部依赖
- ✅ 基于局部视野和经验种子
- ✅ 保持基本探索能力
- ✅ 适合网络不稳定场景

## 测试验证

运行单元测试 (无需真实 API)：
```bash
# 测试会 mock API 调用
pytest tests/test_core.py -v
```

预期输出：
```
tests/test_core.py::test_llm_planner_fallback PASSED
tests/test_core.py::test_yogacara_graph PASSED
...
9 passed in 2.34s
```

## 常见问题

### Q1: 没有 API key 能测试吗？
A: 可以！设置 `use_fallback=True`，系统会自动使用启发式算法。

### Q2: 国内访问不稳定怎么办？
A: 推荐使用 DeepSeek 或 Qwen (阿里云)，国内访问速度快。或使用 Ollama 自建。

### Q3: 如何监控 API 调用成本？
A: 查看 `src/metrics.py`，已集成 Prometheus 指标收集。

### Q4: 能否批量调用降低成本？
A: 可以！修改 `llm_planner.py` 的 `plan()` 方法，支持 batch 请求。

## 下一步

1. **获取 API Key**: 注册 DeepSeek/Qwen/OpenAI
2. **配置环境变量**: 创建 `.env` 文件
3. **运行测试**: `pytest tests/ -v`
4. **开始实验**: `python src/exp_automator.py`

---

**优势总结**：
- ✅ 无需本地 GPU
- ✅ 无需安装大型库 (transformers, vllm 等)
- ✅ 504MB 存储限制下完美运行
- ✅ 支持多种 API 提供商
- ✅ 内置降级保护
- ✅ 成本可控 (¥10 内完成百次实验)
