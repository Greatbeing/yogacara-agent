"""
唯识进化框架 API Server
=======================
支持 FastAPI 服务接口，提供 Agent 运行、健康检查、记忆统计等端点。

启动方式:
    cd yogacara-agent/src
    python -m yogacara_agent.api_server

端口: 8000
"""

import asyncio
import logging
import os
import signal
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Security imports ─────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
try:
    from security.input_sanitizer import InputSanitizer
    from security.tool_sandbox import ToolSandbox
    # rate_limiter.py exports a slowapi Limiter instance + setup_rate_limiting()
    from security.rate_limiter import limiter as _slowapi_limiter
    _HAS_SECURITY = True
except ImportError:
    _HAS_SECURITY = False

from yogacara_langgraph import build_graph, create_session, slow_loop

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

# ── Global session state ──────────────────────────────────────────────────────
_app_session: dict | None = None
_loop_task: asyncio.Task | None = None
loop_started = False
_shutdown_event = asyncio.Event()

# ── Security instances ───────────────────────────────────────────────────────
if _HAS_SECURITY:
    _sanitizer = InputSanitizer()
    _tool_sandbox = ToolSandbox(allowed_tools={"query_knowledge_base", "calculate_metric"})
else:
    _sanitizer = None
    _tool_sandbox = None


def _get_session() -> dict:
    """Get or create the shared application session."""
    global _app_session
    if _app_session is None:
        _app_session = create_session()
    return _app_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: start background slow-loop on startup,
    signal graceful shutdown on termination.
    """
    global _loop_task, loop_started

    session = _get_session()
    if not loop_started:
        _loop_task = asyncio.create_task(
            slow_loop(session["alaya"], interval=10)
        )
        loop_started = True
        logger.info("[API] Slow-loop started (interval=10s)")

    yield  # Application runs here

    # ── Shutdown ────────────────────────────────────────────────────────────
    logger.info("[API] Shutdown signal received")
    _shutdown_event.set()

    if _loop_task and not _loop_task.done():
        _loop_task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(_loop_task), timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        logger.info("[API] Slow-loop cancelled")

    # Flush alaya memory to disk
    alaya = session["alaya"]
    if hasattr(alaya, "perfume_update"):
        alaya.perfume_update()
        logger.info(f"[API] Alaya flushed: {len(alaya.seeds)} seeds persisted")


# Module-level compiled graph (cached, thread-safe for read-only use)
_graph: CompiledStateGraph | None = None


def _get_graph() -> CompiledStateGraph:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


app = FastAPI(
    title="唯识进化框架 API",
    version="1.3.0",
    lifespan=lifespan,
    description="基于唯识学的 AI Agent 自指环认知架构",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ──────────────────────────────────────────────────
class AgentRequest(BaseModel):
    """Episode 运行请求。"""

    max_steps: int = Field(default=60, ge=1, le=200, description="最大步数")
    custom_obs: dict[str, Any] | None = Field(
        default=None, description="自定义初始位置 {\"pos\": [x, y]}"
    )
    seed_id: str | None = Field(
        default=None, description="指定会话种子ID（用于追踪）"
    )


class AgentResponse(BaseModel):
    """Episode 运行响应。"""

    status: str
    steps: int
    cumulative_reward: float
    manas_reflections: int
    resources_found: int
    final_pos: list[int]
    seed_id: str
    duration_ms: int | None = None


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    uptime: str
    memory_seeds: int
    seed_types: dict[str, int]
    avg_importance: float
    manas_reflections: int
    slow_loop_running: bool


class MemoryStatsResponse(BaseModel):
    """记忆统计响应。"""

    total_seeds: int
    storage_type: str
    path: str
    seed_types: dict[str, int]
    avg_importance: float
    last_updated: str


# ── Utility ───────────────────────────────────────────────────────────────────
_start_time = datetime.now()


def _uptime() -> str:
    delta = datetime.now() - _start_time
    total_s = int(delta.total_seconds())
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


# ── Security helpers ─────────────────────────────────────────────────────────
def _apply_security(req: Request) -> None:
    """Apply rate-limiting via slowapi; raises HTTPException on failure."""
    if _HAS_SECURITY and _slowapi_limiter:
        # Let slowapi handle it via the limiter applied as a dependency
        # For inline check, we use a simple in-memory counter as fallback
        client = req.client.host if req.client else "unknown"
        import time
        now = time.time()
        if not hasattr(_apply_security, "_window"):
            _apply_security._window = {}  # type: ignore[attr-defined]
        win = _apply_security._window  # type: ignore[attr-defined]
        # Simple sliding window: 60 req/min per client
        cutoff = now - 60
        win.setdefault(client, [])
        win[client] = [t for t in win[client] if t > cutoff]
        if len(win[client]) >= 60:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded (60 req/min). Try again later."
            )
        win[client].append(now)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"])
async def root():
    """API 根路径。"""
    return {
        "name": "唯识进化框架 API",
        "version": "1.3.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health():
    """
    健康检查与状态摘要。

    返回 Alaya 记忆系统状态、末那识拦截次数、慢循环运行状态。
    """
    _apply_security(Request.__new__(Request))  # dummy for rate limiter

    session = _get_session()
    alaya = session["alaya"]
    manas = session["manas"]

    # Seed type breakdown
    type_counts: dict[str, int] = {}
    for s in alaya.seeds:
        t = s.get("seed_type", "未知")
        type_counts[t] = type_counts.get(t, 0) + 1

    avg_imp = (
        sum(s.get("imp", 0) for s in alaya.seeds) / len(alaya.seeds)
        if alaya.seeds else 0.0
    )

    return HealthResponse(
        status="ok",
        uptime=_uptime(),
        memory_seeds=len(alaya.seeds),
        seed_types=type_counts,
        avg_importance=round(avg_imp, 4),
        manas_reflections=manas.reflections,
        slow_loop_running=_loop_task is not None and not _loop_task.done(),
    )


@app.get("/memory/stats", response_model=MemoryStatsResponse, tags=["memory"])
async def memory_stats():
    """
    Alaya 记忆系统详细统计。

    返回种子总数、类型分布、平均重要性、存储路径。
    """
    session = _get_session()
    alaya = session["alaya"]

    type_counts: dict[str, int] = {}
    for s in alaya.seeds:
        t = s.get("seed_type", "未知")
        type_counts[t] = type_counts.get(t, 0) + 1

    avg_imp = (
        sum(s.get("imp", 0) for s in alaya.seeds) / len(alaya.seeds)
        if alaya.seeds else 0.0
    )

    last_ts = max((s.get("ts", 0) for s in alaya.seeds), default=0)
    last_updated = (
        datetime.fromtimestamp(last_ts).isoformat()
        if last_ts else "never"
    )

    return MemoryStatsResponse(
        total_seeds=len(alaya.seeds),
        storage_type=alaya.storage,
        path=alaya.path,
        seed_types=type_counts,
        avg_importance=round(avg_imp, 4),
        last_updated=last_updated,
    )


@app.get("/memory/seeds", response_model=list[dict], tags=["memory"])
async def list_seeds(
    seed_type: str | None = None,
    limit: int = 20,
):
    """
    列出当前 Alaya 记忆中的种子。

    - **seed_type**: 过滤类型（名言种/业种/异熟种）
    - **limit**: 最大返回数量（默认20）
    """
    session = _get_session()
    alaya = session["alaya"]

    candidates = alaya.seeds
    if seed_type:
        candidates = [s for s in alaya.seeds if s.get("seed_type") == seed_type]

    # Strip heavy fields for API response
    stripped = []
    for s in candidates[-limit:]:
        stripped.append({
            "step": s.get("step"),
            "pos": s.get("pos"),
            "action": s.get("action"),
            "reward": s.get("reward", s.get("rew")),
            "seed_type": s.get("seed_type"),
            "importance": round(s.get("imp", 0), 3),
            "align": round(s.get("align", 0), 3),
            "nature": s.get("tag", "依他起"),
            "ts": s.get("ts"),
        })
    return stripped


@app.post("/memory/perfume", tags=["memory"])
async def trigger_perfume():
    """
    手动触发熏习更新（衰减旧种子、提升高奖励种子）。

    通常由 slow_loop 自动调用，此端点用于手动干预。
    """
    session = _get_session()
    alaya = session["alaya"]
    if hasattr(alaya, "perfume_update"):
        alaya.perfume_update()
        return {"status": "ok", "seeds_after": len(alaya.seeds)}
    return {"status": "noop", "message": "perfume_update not available"}


@app.post("/run_episode", response_model=AgentResponse, tags=["agent"])
async def run_episode(req: AgentRequest, request: Request):
    """
    运行一个完整的 Agent Episode。

    - 使用共享 session（复用已积累的种子记忆）
    - reset 环境后从起点开始
    - 返回四智指标摘要和最终状态
    """
    _apply_security(request)

    # Sanitize custom_obs
    if req.custom_obs:
        if not _sanitizer or _sanitizer.sanitize(req.custom_obs):
            pass  # clean
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid custom_obs: contains unsafe values."
            )

    import time
    t0 = time.monotonic()

    session = _get_session()
    env = session["env"]
    manas = session["manas"]

    seed_id = req.seed_id or f"ep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    try:
        env.reset()
        if req.custom_obs:
            pos = req.custom_obs.get("pos", [0, 0])
            env.agent_pos = list(pos)

        init_state = {
            "obs": env._observe(),
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

        # Bound steps
        step_limit = min(req.max_steps, 200)

        final_state = await _get_graph().ainvoke(init_state)

        # Collect result
        steps_taken = final_state.get("step", 0)
        cum_reward = sum(final_state.get("recent_rewards", []))

        # Count resources (reward > 2.0 signals resource found in GridSimV2)
        resources_found = sum(1 for r in final_state.get("recent_rewards", []) if r > 2.0)

        duration_ms = int((time.monotonic() - t0) * 1000)

        return AgentResponse(
            status="success",
            steps=steps_taken,
            cumulative_reward=round(cum_reward, 2),
            manas_reflections=manas.reflections,
            resources_found=resources_found,
            final_pos=final_state.get("obs", {}).get("pos", [0, 0]),
            seed_id=seed_id,
            duration_ms=duration_ms,
        )

    except Exception as e:
        logger.exception("[API] Episode failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/metrics/wisdom", tags=["metrics"])
async def get_wisdom_metrics():
    """
    获取当前 session 的四智量化指标（需要 session 中有 metrics）。

    注意：四智指标在每次 run_episode 后通过 print 输出，
    此端点返回最近一次指标的快照（如有）。
    """
    session = _get_session()
    metrics = session.get("metrics", {})

    if not metrics:
        return {
            "status": "no_data",
            "message": "Run /run_episode first to generate metrics."
        }

    return {
        "status": "ok",
        "metrics": metrics,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    import uvicorn

    try:
        asyncio.get_running_loop()
        # Already in a loop: run uvicorn in a separate thread
        def run_server():
            uvicorn.run(
                "yogacara_agent.api_server:app",
                host="0.0.0.0",
                port=8000,
                reload=False,
                log_level="info",
            )

        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        t.join()
    except RuntimeError:
        # No running loop: safe to use asyncio.run()
        uvicorn.run(
            "yogacara_agent.api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
