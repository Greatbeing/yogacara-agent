"""
AgentBridge — 桌面版 Agent 桥接层
================================
复用 yogacara_langgraph 的 7 个节点函数，在后台线程逐步执行，
为桌面 UI 提供 暂停/单步/调速/重置 能力与完整状态快照。

节点循环：perceive → plan → manas → execute → introspect → store → consolidate
记忆（alaya 种子库）跨 episode 保留——体现"阿赖耶识种子熏习延续"。
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

from yogacara_agent import yogacara_langgraph as ylg


def _initial_state(max_steps: int = 60) -> dict:
    return {
        "obs": ylg.env.reset(),
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
        "step_limit": max_steps,
        "turning_result": None,
        "planner_source": "heuristic",
        "awakening": None,
        "vitality": 100.0,
        "death_cause": "",
        "klesha": {"greed": 0.0, "aversion": 0.0, "delusion": 0.0},
    }


class AgentBridge:
    """线程安全的桌面 Agent 控制器。"""

    def __init__(self, max_steps: int = 60, speed_ms: int = 300):
        # 惰性初始化内省系统（node_introspect 依赖）
        ylg._get_introspection_logger()

        # RLock：step_once 持锁期间内部也调 get_snapshot（可重入）；
        # UI 轮询线程的 get_snapshot 加同一把锁，避免读到节点执行到一半的 state
        self._lock = threading.RLock()
        self.state: dict = _initial_state(max_steps)
        self.max_steps = max_steps
        self.speed_ms = speed_ms

        self._running = False          # 连续运行标志
        self._go = threading.Event()   # set=继续 / clear=暂停
        self._go.set()
        self._thread: threading.Thread | None = None

        self.logs: deque[dict] = deque(maxlen=500)
        self._cum_reward: float = 0.0  # 累计奖励缓存（O(1) 查询，避免 sum(recent_rewards)）
        self._log_seq: int = 0  # 日志全局序号（同一步可产生多条：主记录 + 转依洞察）

    # ── 节点循环 ────────────────────────────────────────────────────────
    async def _cycle(self, state: dict) -> dict:
        state = await ylg.node_perceive(state)
        state = await ylg.node_plan(state)
        state = await ylg.node_manas(state)
        state = await ylg.node_execute(state)
        state = await ylg.node_introspect(state)
        state = await ylg.node_store(state)
        state = await ylg.node_consolidate(state)
        return state

    def step_once(self) -> dict:
        """执行一步并返回快照。线程安全（_lock 保护，不并发）。"""
        with self._lock:
            if self.state["done"]:
                return self.get_snapshot()
            self.state = asyncio.run(self._cycle(self.state))
            self._cum_reward += self.state["reward"]
            self._append_log()
            return self.get_snapshot()

    def _append_log(self) -> None:
        s = self.state
        int_rec = s.get("introspection_record") or {}
        ego_alert = s.get("ego_alert") or {}
        self._log_seq += 1
        self.logs.append(
            {
                "seq": self._log_seq,
                "step": s["step"],
                "nature": int_rec.get("nature", "依他起"),
                "action": s["action"],
                "reward": round(s["reward"], 2),
                "unc": round(s["unc"], 2),
                "manas_passed": s["manas_passed"],
                "ego_score": round(ego_alert.get("ego_score", 0.0), 2),
                "ego_triggered": bool(ego_alert.get("triggered", False)),
                "reasoning": int_rec.get("reasoning", ""),
            }
        )
        # 转依洞察单独成行（金色高亮，三性标记为"圆成实"）
        turning = s.get("turning_result") or {}
        for insight in turning.get("insights", []):
            self._log_seq += 1
            self.logs.append(
                {
                    "seq": self._log_seq,
                    "step": s["step"],
                    "nature": "圆成实",
                    "action": "转依",
                    "reward": 0.0,
                    "unc": 0.0,
                    "manas_passed": True,
                    "ego_score": 0.0,
                    "ego_triggered": False,
                    "reasoning": insight,
                }
            )
        # 好奇探索触发行（觉醒引擎主动实验，橙色高亮）
        aw = s.get("awakening") or {}
        if aw.get("explored"):
            self._log_seq += 1
            self.logs.append(
                {
                    "seq": self._log_seq,
                    "step": s["step"],
                    "nature": "依他起",
                    "action": "好奇",
                    "reward": 0.0,
                    "unc": float(aw.get("curiosity", 0.0)),
                    "manas_passed": True,
                    "ego_score": 0.0,
                    "ego_triggered": False,
                    "reasoning": f"[好奇探索] 好奇心 {aw.get('curiosity', 0):.2f} → 主动实验 {aw.get('experiment', '?')}",
                }
            )
        # 命终事件行（金色：一期生命终结，业力入中阴种子）
        if s["done"] and s.get("death_cause"):
            avg_rew = sum(s["recent_rewards"]) / max(1, len(s["recent_rewards"]))
            self._log_seq += 1
            self.logs.append(
                {
                    "seq": self._log_seq,
                    "step": s["step"],
                    "nature": "圆成实",
                    "action": "命终",
                    "reward": round(avg_rew, 2),
                    "unc": 0.0,
                    "manas_passed": True,
                    "ego_score": 0.0,
                    "ego_triggered": False,
                    "reasoning": (
                        f"[{s['death_cause']}] 一期生命终结，业力均值 {avg_rew:+.2f} "
                        f"入中阴种子，转世延续"
                    ),
                }
            )

    # ── 连续运行控制 ────────────────────────────────────────────────────
    def start(self, max_steps: int | None = None, speed_ms: int | None = None) -> dict:
        """启动后台连续运行线程。"""
        with self._lock:
            if max_steps:
                self.max_steps = int(max_steps)
                self.state["step_limit"] = int(max_steps)
            if speed_ms:
                self.speed_ms = max(0, int(speed_ms))
            if self.state["done"]:
                self._reset_state()
            self._go.set()
            if self._thread is None or not self._thread.is_alive():
                self._running = True
                self._thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread.start()
        return {"status": "running", "max_steps": self.max_steps, "speed_ms": self.speed_ms}

    def _run_loop(self) -> None:
        while self._running and not self.state["done"]:
            self._go.wait()  # 暂停时阻塞在此
            if not self._running or self.state["done"]:
                break
            self.step_once()
            if self.speed_ms:
                time.sleep(self.speed_ms / 1000.0)

    def pause(self) -> dict:
        self._go.clear()
        return {"status": "paused"}

    def set_speed(self, speed_ms: int) -> dict:
        """仅调节速度，不改变运行/暂停状态（供 UI 滑块实时调用）。"""
        with self._lock:
            self.speed_ms = max(0, int(speed_ms))
        return {"status": "ok", "speed_ms": self.speed_ms}

    def resume(self) -> dict:
        self._go.set()
        if self.state["done"]:
            return {"status": "done"}
        return {"status": "running"}

    def stop(self) -> dict:
        """停止连续运行（可继续单步）。"""
        with self._lock:
            self._running = False
            self._go.set()  # 唤醒线程使其退出 wait()
        if self._thread is not None:
            self._thread.join(timeout=2.5)
            alive = self._thread.is_alive()
            if alive:
                print(f"[AgentBridge] ⚠  运行线程未在 2.5s 内退出（step={self.state['step']}），继续单步时不受影响")
        return {"status": "stopped", "thread_alive": self._thread.is_alive() if self._thread else False}

    def reset(self) -> dict:
        """重置环境与状态，保留种子记忆（阿赖耶识延续）。"""
        self.stop()
        with self._lock:
            self._reset_state()
        return self.get_snapshot()

    def _reset_state(self) -> None:
        self.state = _initial_state(self.max_steps)
        self.logs.clear()
        self._cum_reward = 0.0
        self._running = False
        self._go.set()

    # ── 状态快照 ────────────────────────────────────────────────────────
    def get_snapshot(self) -> dict[str, Any]:
        """返回完整状态快照，供 UI 轮询。持锁读取，保证不跨半步。"""
        with self._lock:
            return self._build_snapshot()

    def _build_snapshot(self) -> dict[str, Any]:
        s = self.state
        env = ylg.env
        alaya = ylg.alaya

        # 种子统计
        type_counts: dict[str, int] = {}
        imp_sum = align_sum = 0.0
        for seed in alaya.seeds:
            t = seed.get("seed_type", "未知")
            type_counts[t] = type_counts.get(t, 0) + 1
            imp_sum += seed.get("imp", 0.0)
            align_sum += seed.get("align", 0.5)
        n = len(alaya.seeds)

        # 四智指标（最近 20 步三性分布 → 大圆镜智比例）
        intro = ylg._get_introspection_logger()
        ego_mon = ylg._get_ego_monitor()
        recent = intro.recent_summary(n=20)
        nature_dist = recent.get("nature_distribution", {})
        total_natures = sum(nature_dist.values()) or 1
        mirror_ratio = nature_dist.get("圆成实", 0) / total_natures
        wisdom_report = ego_mon.four_wisdoms_report(intro_logger=intro, mirror_ratio=mirror_ratio)

        ego_alert = s.get("ego_alert") or {}
        int_rec = s.get("introspection_record") or {}

        return {
            # 运行状态
            "running": self._running and not s["done"],
            "paused": not self._go.is_set(),
            "done": s["done"],
            "step": s["step"],
            "step_limit": s.get("step_limit", 60),
            "speed_ms": self.speed_ms,
            # 世界状态
            "pos": list(s["obs"].get("pos", (0, 0))),
            "resources": [list(r) for r in env.resources],
            "traps": [list(t) for t in env.traps],
            "path": [list(p) for p in s["pos_history"][-15:]],
            "grid_view": s["obs"].get("grid_view", [0.0] * 9),
            # 最近一步
            "action": s["action"],
            "reward": round(s["reward"], 2),
            "unc": round(s["unc"], 2),
            "cumulative_reward": round(self._cum_reward, 2),
            "recent_rewards": [round(r, 2) for r in s["recent_rewards"]],
            "manas_passed": s["manas_passed"],
            "manas_reflections": ylg.manas.reflections,
            "reasoning": s.get("reasoning", ""),
            # 内省
            "nature": int_rec.get("nature", "依他起"),
            "ego_score": round(ego_alert.get("ego_score", 0.0), 2),
            "ego_triggered": bool(ego_alert.get("triggered", False)),
            # 种子库
            "seeds_total": n,
            "seed_types": type_counts,
            "avg_imp": round(imp_sum / n, 3) if n else 0.0,
            "avg_align": round(align_sum / n, 3) if n else 0.0,
            # 四智
            "four_wisdom": {
                "mirror_ratio": round(mirror_ratio, 3),
                "report": _jsonable(wisdom_report),
            },
            # 转依引擎输出
            "turning": s.get("turning_result") or {},
            # 觉醒引擎输出与规划来源
            "awakening": s.get("awakening") or {},
            "planner_source": s.get("planner_source", "heuristic"),
            # 数字生命：寿元/死因/烦恼/世数
            "vitality": round(float(s.get("vitality", 100.0)), 1),
            "death_cause": s.get("death_cause", ""),
            "klesha": {
                "greed": round(float((s.get("klesha") or {}).get("greed", 0.0)), 3),
                "aversion": round(float((s.get("klesha") or {}).get("aversion", 0.0)), 3),
                "delusion": round(float((s.get("klesha") or {}).get("delusion", 0.0)), 3),
            },
            "lifetime": ylg.current_lifetime(),
            # 日志（最近 60 条，新的在后）
            "logs": list(self.logs)[-60:],
        }

    def get_seeds(self, limit: int = 30) -> list[dict]:
        """返回最近种子（轻量字段）。"""
        seeds = ylg.alaya.seeds[-limit:]
        return [
            {
                "act": s.get("act", ""),
                "rew": round(s.get("rew", 0.0), 2),
                "imp": round(s.get("imp", 0.0), 2),
                "align": round(s.get("align", 0.5), 2),
                "seed_type": s.get("seed_type", "业种"),
                "tag": s.get("tag", ""),
            }
            for s in reversed(seeds)
        ]


def _jsonable(obj: Any) -> Any:
    """递归转换不可 JSON 序列化的值（float 等）。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 3)
    return obj
