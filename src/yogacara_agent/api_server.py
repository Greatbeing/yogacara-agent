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
import sys
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Security imports ─────────────────────────────────────────────────────────
try:
    from yogacara_agent.security.input_sanitizer import InputSanitizer
    from yogacara_agent.security.tool_sandbox import ToolSandbox

    # rate_limiter.py exports a slowapi Limiter instance + setup_rate_limiting()
    from yogacara_agent.security.rate_limiter import limiter as _slowapi_limiter, setup_rate_limiting

    _HAS_SECURITY = True
except ImportError:
    _HAS_SECURITY = False

from yogacara_agent.yogacara_langgraph import GRID_SIZE, build_graph, create_session, slow_loop
from yogacara_agent import yogacara_langgraph as ylg
from yogacara_agent.constants import RESOURCE_THRESHOLD
from yogacara_agent.evolution_tracker import EvolutionTracker

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

# ── Global session state ──────────────────────────────────────────────────────
_app_session: dict | None = None
_loop_task: asyncio.Task | None = None
loop_started = False
_shutdown_event = asyncio.Event()
_evolution_tracker = EvolutionTracker()

# Module-level compiled graph (cached, thread-safe for read-only use)
_graph: "CompiledStateGraph | None" = None

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
        _loop_task = asyncio.create_task(slow_loop(session["alaya"], interval=10, tracker=_evolution_tracker))
        loop_started = True
        logger.info("[API] Slow-loop started (interval=10s)")

    yield  # Application runs here

    # ── Shutdown ────────────────────────────────────────────────────────────
    logger.info("[API] Shutdown signal received")
    _shutdown_event.set()

    if _loop_task and not _loop_task.done():
        _loop_task.cancel()
        with suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(_loop_task), timeout=5.0)
        logger.info("[API] Slow-loop cancelled")

    # Flush alaya memory to disk
    alaya = session["alaya"]
    if hasattr(alaya, "perfume_update"):
        alaya.perfume_update()
        logger.info(f"[API] Alaya flushed: {len(alaya.seeds)} seeds persisted")


def _get_graph() -> "CompiledStateGraph":
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
# 通配符 origin 与 credentials 组合违反 CORS 规范；如需凭证请通过环境变量
# YOGACARA_ALLOWED_ORIGINS 显式指定白名单（逗号分隔）。
_allowed_origins = [o.strip() for o in os.environ.get("YOGACARA_ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials="*" not in _allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _HAS_SECURITY:
    setup_rate_limiting(app)  # 注册 slowapi 限流器与 429 异常处理器

# 写端点串行化：env/alaya 是进程级共享单例，并发 episode 会互相踩踏
_episode_lock = asyncio.Lock()


# ── Request/Response models ──────────────────────────────────────────────────
class AgentRequest(BaseModel):
    """Episode 运行请求。"""

    max_steps: int = Field(default=60, ge=1, le=200, description="最大步数")
    custom_obs: dict[str, Any] | None = Field(default=None, description='自定义初始位置 {"pos": [x, y]}')
    seed_id: str | None = Field(default=None, description="指定会话种子ID（用于追踪）")


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
    planner_source: str | None = None  # "heuristic" | "llm"（混合规划器来源）
    turning_level: float | None = None  # 转依引擎综合等级
    vitality: float | None = None  # 终局寿元（数字生命内稳态）
    death_cause: str | None = None  # 死因（None=未命终）
    lifetime: int | None = None  # 当下世数


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
    """简易滑动窗口限流（60 req/min per client）；超出抛 429。

    注意：部署在反向代理后所有请求共享代理 IP，应配置可信的
    X-Forwarded-For 解析或改用共享存储（Redis）限流。
    """
    if _HAS_SECURITY and _slowapi_limiter:
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
            raise HTTPException(status_code=429, detail="Rate limit exceeded (60 req/min). Try again later.")
        win[client].append(now)
        # 清理过期客户端，避免按 IP 累积导致内存泄漏
        if len(win) > 10000:
            stale = [c for c, ts in win.items() if not ts or ts[-1] <= cutoff]
            for c in stale:
                del win[c]


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
    session = _get_session()
    alaya = session["alaya"]
    manas = session["manas"]

    # Seed type breakdown
    type_counts: dict[str, int] = {}
    for s in alaya.seeds:
        t = s.get("seed_type", "未知")
        type_counts[t] = type_counts.get(t, 0) + 1

    avg_imp = sum(s.get("imp", 0) for s in alaya.seeds) / len(alaya.seeds) if alaya.seeds else 0.0

    return HealthResponse(
        status="ok",
        uptime=_uptime(),
        memory_seeds=len(alaya.seeds),
        seed_types=type_counts,
        avg_importance=round(avg_imp, 4),
        manas_reflections=manas.reflections,
        slow_loop_running=_loop_task is not None and not _loop_task.done(),
    )


@app.get("/dashboard", tags=["ui"])
async def dashboard():
    dashboard_path = Path(__file__).resolve().parents[2] / "desktop" / "dashboard.html"
    return FileResponse(dashboard_path)


@app.get("/api/agent/status", tags=["ui"])
async def agent_status():
    session = _get_session()
    alaya = session["alaya"]
    env = session["env"]
    manas = session["manas"]
    metrics = session.get("metrics", {})
    last_run = session.get("last_run", {})
    intro_logger = session.get("introspection")
    recent_events: list[dict[str, Any]] = []
    if intro_logger is not None and hasattr(intro_logger, "logs"):
        for record in list(getattr(intro_logger, "logs", []))[-10:]:
            if getattr(record, "manas_intercepted", False):
                recent_events.append(
                    {
                        "step": getattr(record, "step", 0),
                        "nature": getattr(record, "nature", ""),
                        "reasoning": getattr(record, "reasoning", ""),
                        "decision_gap": getattr(record, "decision_gap", 0.0),
                    }
                )

    wisdom = metrics.get("wisdom") or {}
    return {
        "agent_pos": list(getattr(env, "agent_pos", [0, 0])),
        "step": int(getattr(env, "step_count", 0)),
        "cumulative_reward": float(last_run.get("cumulative_reward", 0.0)),
        "uncertainty": float(last_run.get("uncertainty", 0.0)),
        "manas_reflections": int(getattr(manas, "reflections", 0)),
        "resources_found": int(last_run.get("resources_found", 0)),
        "wisdom": wisdom,
        "seed_count": len(getattr(alaya, "seeds", [])),
        "updated_at": last_run.get("updated_at", datetime.now().timestamp()),
        "resources": [list(p) for p in getattr(env, "resources", [])],
        "traps": [list(p) for p in getattr(env, "traps", [])],
        "path": [list(p) for p in last_run.get("pos_history", [])],
        "events": recent_events,
    }


@app.get("/api/evolution/snapshots", tags=["ui"])
async def evolution_snapshots(limit: int = Query(default=100, ge=1, le=200)):
    return {"snapshots": _evolution_tracker.get_snapshots(limit=limit)}


@app.post("/api/evolution/snapshot", tags=["ui"])
async def trigger_evolution_snapshot():
    session = _get_session()
    return {"status": "ok", "snapshot": _evolution_tracker.snapshot(session["alaya"]) }


@app.get("/api/awakening/status", tags=["ui"])
async def awakening_status():
    """觉醒引擎状态：好奇驱动、行为新颖性、实验类型分布、洞察数。"""
    engine = ylg._get_awakening_engine()
    experiments: dict[str, int] = {}
    for h in engine.action_history[-100:]:
        # 仅统计（experiment 类型不落历史，此处给出动作分布替代）
        experiments[h.get("action", "?")] = experiments.get(h.get("action", "?"), 0) + 1
    return {
        "novelty_score": round(float(engine.state.novelty_score), 4),
        "curiosity_threshold": engine.curiosity_threshold,
        "action_history_len": len(engine.action_history),
        "recent_action_distribution": experiments,
        "insight_count": len(engine.insight_log),
        "dream_sessions": len(engine.dream_sessions),
        "llm_planner_enabled": ylg._get_llm_planner() is not None,
    }


@app.get("/api/samsara/history", tags=["ui"])
async def samsara_history(limit: int = Query(default=20, ge=1, le=50)):
    """轮回史：历代生命的总结（步数/业力/死因/转依/烦恼/梦种数）。"""
    history = ylg.get_life_history()
    return {
        "current_lifetime": ylg.current_lifetime(),
        "total_deaths": ylg._samsara["deaths"],
        "lives": history[-limit:],
    }


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

    avg_imp = sum(s.get("imp", 0) for s in alaya.seeds) / len(alaya.seeds) if alaya.seeds else 0.0

    last_ts = max((s.get("ts", 0) for s in alaya.seeds), default=0)
    last_updated = datetime.fromtimestamp(last_ts).isoformat() if last_ts else "never"

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
    limit: int = Query(default=20, ge=1, le=500),
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
        stripped.append(
            {
                "step": s.get("step"),
                "pos": s.get("pos"),
                "action": s.get("action"),
                "reward": s.get("reward", s.get("rew")),
                "seed_type": s.get("seed_type"),
                "importance": round(s.get("imp", 0), 3),
                "align": round(s.get("align", 0), 3),
                "nature": s.get("tag", "依他起"),
                "ts": s.get("ts"),
            }
        )
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

    # 校验 custom_obs：安全模块缺失时拒绝（fail-closed），而不是放行
    if req.custom_obs:
        if _sanitizer is None:
            raise HTTPException(status_code=503, detail="Input sanitizer unavailable; custom_obs rejected.")
        try:
            _sanitizer.validate_obs(
                {
                    "grid_view": req.custom_obs.get("grid_view", [0.0] * 9),
                    "pos": req.custom_obs.get("pos", [0, 0]),
                }
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    import time

    t0 = time.monotonic()

    session = _get_session()
    env = session["env"]
    manas = session["manas"]

    seed_id = req.seed_id or f"ep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    try:
        async with _episode_lock:  # env/alaya 为共享单例，episode 必须串行
            reflections_before = manas.reflections
            env.reset()
            if req.custom_obs:
                pos = req.custom_obs.get("pos", [0, 0])
                if (
                    not isinstance(pos, (list, tuple))
                    or len(pos) != 2
                    or not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < GRID_SIZE for v in pos)
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"pos must be two ints within [0, {GRID_SIZE - 1}].",
                    )
                env.agent_pos = list(pos)

            # Bound steps（node_execute 会在达到上限时置 done）
            step_limit = min(req.max_steps, 200)

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
                "introspection_record": None,
                "ego_alert": None,
                "plan_scores": None,
                "reasoning": "",
                "steps_since_resource": 0,
                "steps_at_same_pos": 0,
                "step_limit": step_limit,
                "turning_result": None,
                "planner_source": "heuristic",
                "awakening": None,
                "vitality": 100.0,
                "death_cause": "",
                "klesha": {"greed": 0.0, "aversion": 0.0, "delusion": 0.0},
            }

            final_state = await _get_graph().ainvoke(init_state)

        # Collect result
        steps_taken = final_state.get("step", 0)
        cum_reward = sum(final_state.get("recent_rewards", []))

        # Count resources (reward > 2.0 signals resource found in GridSimV2)
        resources_found = sum(1 for r in final_state.get("recent_rewards", []) if r > RESOURCE_THRESHOLD)

        duration_ms = int((time.monotonic() - t0) * 1000)
        intro_logger = session.get("introspection")
        ego_monitor = session.get("ego_monitor")
        recent_intro = intro_logger.recent_summary(n=20) if intro_logger and hasattr(intro_logger, "recent_summary") else {}
        nature_dist = recent_intro.get("nature_distribution", {}) if isinstance(recent_intro, dict) else {}
        round_real = nature_dist.get("圆成实", 0)
        total_natures = sum(nature_dist.values()) or 1
        mirror_ratio = round_real / total_natures
        wisdom_report = ego_monitor.four_wisdoms_report(intro_logger=intro_logger, mirror_ratio=mirror_ratio) if ego_monitor else {}

        session["metrics"] = {"wisdom": wisdom_report}
        session["last_run"] = {
            "cumulative_reward": round(cum_reward, 2),
            "resources_found": resources_found,
            "uncertainty": final_state.get("unc", 0.0),
            "updated_at": datetime.now().timestamp(),
            "step": steps_taken,
            "pos_history": final_state.get("pos_history", []),
        }

        return AgentResponse(
            status="success",
            steps=steps_taken,
            cumulative_reward=round(cum_reward, 2),
            manas_reflections=manas.reflections - reflections_before,
            resources_found=resources_found,
            final_pos=final_state.get("obs", {}).get("pos", [0, 0]),
            seed_id=seed_id,
            duration_ms=duration_ms,
            planner_source=final_state.get("planner_source", "heuristic"),
            turning_level=(
                (final_state.get("turning_result") or {}).get("turning_level")
            ),
            vitality=final_state.get("vitality"),
            death_cause=final_state.get("death_cause") or None,
            lifetime=ylg.current_lifetime(),
        )

    except HTTPException:
        raise
    except Exception:
        # 异常细节只进日志，不回传客户端
        logger.exception("[API] Episode failed")
        raise HTTPException(status_code=500, detail="internal error")


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
        return {"status": "no_data", "message": "Run /run_episode first to generate metrics."}

    return {
        "status": "ok",
        "metrics": metrics,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()  # .env 的 LLM_API_KEY / YOGACARA_LLM_PLAN 等对混合规划器生效

    # 直接以 app 对象启动，避免字符串 import 路径导致的二次导入
    uvicorn.run(
        app,
        host=os.environ.get("YOGACARA_HOST", "0.0.0.0"),
        port=int(os.environ.get("YOGACARA_PORT", "8000")),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
