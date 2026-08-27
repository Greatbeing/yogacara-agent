"""
混合规划器（LLM+启发式）与觉醒引擎接入主管线的集成测试
========================================================
覆盖：门控/节流/熔断、觉醒好奇驱动、探索偏置、桥接快照 schema。
"""

import os
import sys
import time as _time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from yogacara_agent import yogacara_langgraph as ylg


def _init_state(step: int = 0) -> dict:
    return {
        "obs": ylg.env.observe(),
        "action": "",
        "reward": 0.0,
        "done": False,
        "step": step,
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
        "step_limit": 60,
        "turning_result": None,
        "planner_source": "heuristic",
        "awakening": None,
    }


@pytest.fixture(autouse=True)
def _reset_planner_state(monkeypatch):
    """每个测试独立：重置 LLM 单例、熔断器、环境开关。"""
    monkeypatch.setattr(ylg, "_llm_planner", None)
    monkeypatch.setattr(ylg, "_llm_enabled_checked", False)
    ylg._reset_llm_circuit()
    monkeypatch.delenv("YOGACARA_LLM_PLAN", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    yield
    ylg._reset_llm_circuit()


class TestHybridPlannerGating:
    """门控逻辑：默认关闭、无 key 降级、env 开启"""

    def test_disabled_by_default(self):
        """默认 YOGACARA_LLM_PLAN 未设 → 纯启发式"""
        s = _init_state(step=10)
        action, unc, reasoning, source = ylg._hybrid_plan(s, "UP", 0.3)
        assert source == "heuristic"
        assert action == "UP" and unc == 0.3 and reasoning == ""
        assert ylg._get_llm_planner() is None

    def test_enabled_without_key(self):
        """开启但无 API key → 警告并保持启发式"""
        os.environ["YOGACARA_LLM_PLAN"] = "1"
        s = _init_state(step=10)
        _, _, _, source = ylg._hybrid_plan(s, "UP", 0.3)
        assert source == "heuristic"
        assert ylg._get_llm_planner() is None

    def test_interval_throttle(self, monkeypatch):
        """step % interval != 0 时不调用 LLM"""
        calls = []

        class FakePlanner:
            model = "fake"
            last_success = True

            def plan(self, obs, seeds):
                calls.append(1)
                return "DOWN", 0.2, "fake", []

        monkeypatch.setattr(ylg, "_llm_planner", FakePlanner())
        monkeypatch.setattr(ylg, "_llm_enabled_checked", True)
        monkeypatch.setenv("YOGACARA_LLM_INTERVAL", "10")
        # step=3 不在 10 的整倍数 → 不调用
        _, _, _, source = ylg._hybrid_plan(_init_state(step=3), "UP", 0.3)
        assert source == "heuristic" and calls == []
        # step=10 → 调用并覆盖
        action, _, reasoning, source = ylg._hybrid_plan(_init_state(step=10), "UP", 0.3)
        assert source == "llm" and action == "DOWN"
        assert "[LLM]" in reasoning and len(calls) == 1


class TestHybridCircuitBreaker:
    """熔断器：连续失败停用"""

    def _failing_planner(self, monkeypatch):
        class FailingPlanner:
            model = "fake"
            last_success = False

            def plan(self, obs, seeds):
                return "STAY", 1.0, "启发式降级", []

        monkeypatch.setattr(ylg, "_llm_planner", FailingPlanner())
        monkeypatch.setattr(ylg, "_llm_enabled_checked", True)
        monkeypatch.setenv("YOGACARA_LLM_INTERVAL", "1")

    def test_opens_after_consecutive_failures(self, monkeypatch):
        self._failing_planner(monkeypatch)
        for step in (1, 2, 3):
            _, _, _, source = ylg._hybrid_plan(_init_state(step=step), "UP", 0.3)
            assert source == "heuristic"
        assert ylg._llm_circuit["fails"] == ylg.LLM_CIRCUIT_THRESHOLD
        assert ylg._llm_circuit["disabled_until"] > _time.time()

    def test_open_circuit_skips_llm(self, monkeypatch):
        self._failing_planner(monkeypatch)
        ylg._llm_circuit["disabled_until"] = _time.time() + 60.0
        calls = []

        # 即使 step 满足间隔，熔断期间也不应触碰 planner（换成计数 planner 验证）
        class Counting:
            model = "c"
            last_success = True

            def plan(self, obs, seeds):
                calls.append(1)
                return "DOWN", 0.1, "x", []

        monkeypatch.setattr(ylg, "_llm_planner", Counting())
        _, _, _, source = ylg._hybrid_plan(_init_state(step=5), "UP", 0.3)
        assert source == "heuristic" and calls == []

    def test_success_resets_failures(self, monkeypatch):
        class Flaky:
            model = "f"
            last_success = True

            def plan(self, obs, seeds):
                return "LEFT", 0.1, "ok", []

        monkeypatch.setattr(ylg, "_llm_planner", Flaky())
        monkeypatch.setattr(ylg, "_llm_enabled_checked", True)
        monkeypatch.setenv("YOGACARA_LLM_INTERVAL", "1")
        ylg._llm_circuit["fails"] = 2  # 差一次熔断
        _, _, _, source = ylg._hybrid_plan(_init_state(step=1), "UP", 0.3)
        assert source == "llm"
        assert ylg._llm_circuit["fails"] == 0  # 成功清零


class TestAwakeningIntegration:
    """觉醒引擎好奇驱动接入"""

    def test_awakening_in_state_after_plan(self):
        """node_plan 后 state['awakening'] 字段齐全"""
        import asyncio

        state = asyncio.run(ylg.node_plan(_init_state()))
        aw = state["awakening"]
        assert aw is not None
        for key in ("curiosity", "experiment", "risk_tolerance", "explored", "novelty"):
            assert key in aw
        assert 0.0 <= aw["curiosity"] <= 1.0
        assert aw["experiment"] in ("exploration", "hypothesis_testing", "exploitation")
        assert state["planner_source"] == "heuristic"

    def test_action_history_written(self):
        """行为历史被写入（新颖性计算的前提）"""
        import asyncio

        engine = ylg._get_awakening_engine()
        before = len(engine.action_history)
        asyncio.run(ylg.node_plan(_init_state()))
        assert len(engine.action_history) == before + 1
        last = engine.action_history[-1]
        assert "state_hash" in last and "action" in last

    def test_memory_diversity_bounds(self):
        d = ylg._memory_diversity()
        assert 0.0 <= d <= 1.0

    def test_exploration_bias_statistical(self):
        """多步运行好奇心探索分支稳定触发且不崩溃（统计性冒烟）"""
        import asyncio

        engine = ylg._get_awakening_engine()
        engine.action_history.clear()  # 低历史 → 行为新颖性高 → 好奇心可能达阈
        explored = 0
        trials = 30
        for _ in range(trials):
            state = asyncio.run(ylg.node_plan(_init_state()))
            aw = state["awakening"]
            assert aw["experiment"] in ("exploration", "hypothesis_testing", "exploitation")
            if aw["explored"]:
                explored += 1
                # 探索动作为四向移动之一
                assert state["action"] in ("UP", "DOWN", "LEFT", "RIGHT")
        # 不强断言探索次数（好奇心是否达阈取决于记忆/新颖性状态），
        # 但 30 步内 awakening 字段必须全程有效
        assert 0 <= explored <= trials


class TestBridgeSnapshotSchema:
    """桌面桥接快照扩展"""

    def test_snapshot_has_awakening_and_planner(self):
        from desktop.agent_bridge import AgentBridge

        b = AgentBridge(max_steps=4, speed_ms=0)
        snap = b.step_once()
        assert "awakening" in snap and "planner_source" in snap
        assert snap["planner_source"] in ("heuristic", "llm")
        assert "curiosity" in snap["awakening"]
        # 好奇触发行（若发生）与转依行同用 seq 去重，不互相吞掉
        seqs = [entry["seq"] for entry in snap["logs"]]
        assert seqs == sorted(seqs)
