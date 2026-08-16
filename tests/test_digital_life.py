"""
数字生命系统集成测试：寿元内稳态 / 贪嗔痴心所 / 轮回转世 / 中阴种子
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yogacara_agent import yogacara_langgraph as ylg
from yogacara_agent.constants import (
    VITALITY_INIT,
    VITALITY_DRAIN,
    VITALITY_RESOURCE,
    VITALITY_TRAP,
    RESOURCE_THRESHOLD,
    TURNING_EGO_DECAY_RATE,
)


def _state(**over) -> dict:
    s = {
        "obs": ylg.env.observe(),
        "action": "STAY",
        "reward": 0.0,
        "done": False,
        "step": 1,
        "seeds": [],
        "unc": 0.3,
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
        "vitality": VITALITY_INIT,
        "death_cause": "",
        "klesha": {"greed": 0.0, "aversion": 0.0, "delusion": 0.0},
    }
    s.update(over)
    return s


class TestVitality:
    """寿元内稳态"""

    def test_drain_on_empty_step(self):
        ylg.env.reset()
        s = ylg.env.__class__ and _state(action="DOWN")
        before = s["vitality"]
        import asyncio

        s = asyncio.run(ylg.node_execute(s))
        assert s["vitality"] == before - VITALITY_DRAIN, "空步应只消耗"

    def test_resource_restores(self):
        ylg.env.reset()
        ylg.env.agent_pos = [6, 7]  # 下一格 DOWN → (7,7) 资源
        s = _state(action="DOWN", vitality=50.0)
        import asyncio

        s = asyncio.run(ylg.node_execute(s))
        assert s["reward"] >= RESOURCE_THRESHOLD, "应吃到资源"
        assert s["vitality"] == 50.0 - VITALITY_DRAIN + VITALITY_RESOURCE

    def test_trap_damages(self):
        ylg.env.reset()
        ylg.env.agent_pos = [3, 4]  # DOWN → (4,4) 陷阱
        s = _state(action="DOWN", vitality=50.0)
        import asyncio

        s = asyncio.run(ylg.node_execute(s))
        assert s["reward"] <= -2.0
        assert s["vitality"] == 50.0 - VITALITY_DRAIN - VITALITY_TRAP

    def test_starvation_death(self):
        """寿元耗尽 → done + 死因 + 轮回计数"""
        ylg.env.reset()
        deaths_before = ylg._samsara["deaths"]
        s = _state(action="DOWN", vitality=1.0)
        import asyncio

        s = asyncio.run(ylg.node_execute(s))
        assert s["vitality"] == 0.0
        assert s["done"] is True
        assert s["death_cause"] == "寿元耗尽"
        assert ylg._samsara["deaths"] == deaths_before + 1
        assert ylg.current_lifetime() == deaths_before + 2  # 已转至下一世

    def test_stay_is_net_drain(self):
        """STAY 休息仍为净消耗（防赖着不动的永生策略）"""
        ylg.env.reset()
        s = _state(action="STAY", vitality=50.0)
        import asyncio

        s = asyncio.run(ylg.node_execute(s))
        assert s["vitality"] < 50.0, "STAY 应为净消耗"


class TestKlesha:
    """贪嗔痴心所"""

    def test_greed_rises_on_resource(self):
        ylg.env.reset()
        ylg.env.agent_pos = [6, 7]
        s = _state(action="DOWN")
        import asyncio

        s = asyncio.run(ylg.node_execute(s))
        assert s["klesha"]["greed"] > 0.0, "得资源应增长贪"

    def test_aversion_rises_on_trap(self):
        ylg.env.reset()
        ylg.env.agent_pos = [3, 4]
        s = _state(action="DOWN")
        import asyncio

        s = asyncio.run(ylg.node_execute(s))
        assert s["klesha"]["aversion"] > 0.0, "踩陷阱应增长嗔"

    def test_delusion_tracks_uncertainty(self):
        ylg.env.reset()
        s = _state(action="DOWN", unc=0.9)
        import asyncio

        s = asyncio.run(ylg.node_execute(s))
        assert s["klesha"]["delusion"] > 0.2, "痴应跟随高不确定性"

    def test_modulation_biases_toward_resource_when_greedy(self):
        """高贪 + 上方资源可见 → 调制后应倾向 UP"""
        obs = {"pos": (5, 5), "grid_view": [0.0] * 9, "step": 1}
        obs["grid_view"][1] = 1.0  # UP 方向资源
        scores = {"UP": 0.2, "DOWN": 0.3, "LEFT": 0.25, "RIGHT": 0.25, "STAY": 0.1}
        modulated, best = ylg._apply_klesha_modulation(
            scores, obs, {"greed": 1.0, "aversion": 0.0, "delusion": 0.0}
        )
        assert best == "UP", f"高贪应趋近资源，实际选 {best}"
        assert modulated["UP"] > scores["UP"]

    def test_modulation_biases_away_from_trap_when_averse(self):
        """高嗔 + 下方陷阱可见 → DOWN 应被惩罚"""
        obs = {"pos": (5, 5), "grid_view": [0.0] * 9, "step": 1}
        obs["grid_view"][7] = -1.0  # DOWN 方向陷阱
        scores = {"UP": 0.1, "DOWN": 0.5, "LEFT": 0.2, "RIGHT": 0.2, "STAY": 0.05}
        modulated, best = ylg._apply_klesha_modulation(
            scores, obs, {"greed": 0.0, "aversion": 1.0, "delusion": 0.0}
        )
        assert best != "DOWN", "高嗔应回避陷阱方向"
        assert modulated["DOWN"] < scores["DOWN"]

    def test_zero_klesha_no_change(self):
        obs = {"pos": (0, 0), "grid_view": [0.0] * 9, "step": 0}
        scores = {"UP": 1.0, "DOWN": 0.5}
        modulated, best = ylg._apply_klesha_modulation(
            scores, obs, {"greed": 0.0, "aversion": 0.0, "delusion": 0.0}
        )
        assert modulated == scores and best == "UP"

    def test_turning_reduces_klesha(self):
        """转依消解我执 → 贪嗔痴同步衰减（修行减恼）"""
        import asyncio

        # 引擎为共享单例，我执强度被先前测试消耗；显式复位使消解量确定
        engine = ylg._get_turning_engine()
        engine.manas_dissolver.current_ego_strength = 1.0
        k = {"greed": 0.8, "aversion": 0.6, "delusion": 0.4}
        # benefit_all_beings 路径：dissolved = 1.0 × 0.15 × 0.5 = 0.075
        # （注意 dissolve 对空 action_probs 直接返回 0，故需提供 plan_scores）
        dissolved = 1.0 * TURNING_EGO_DECAY_RATE * 0.5
        relief = 1.0 / (1.0 + 2.0 * dissolved)
        s = _state(klesha=dict(k), plan_scores={"UP": 2.0, "DOWN": 1.0, "STAY": 0.5})
        s = asyncio.run(ylg.node_consolidate(s))
        for key in k:
            expected = k[key] * relief
            assert abs(s["klesha"][key] - expected) < 1e-6, (
                f"{key}: {s['klesha'][key]} != {expected}（dissolved={dissolved}）"
            )


class TestSamsara:
    """轮回转世与中阴种子"""

    def test_bardo_seed_on_death(self):
        """命终后中阴种子入库（业力总结，imp=1.0）"""
        ylg.env.reset()
        before = len(ylg.alaya.seeds)
        import asyncio

        s = asyncio.run(ylg.node_execute(_state(action="DOWN", vitality=1.0)))
        s["seeds"] = []
        s = asyncio.run(ylg.node_store(s))
        bardo = [x for x in ylg.alaya.seeds if str(x.get("tag", "")).startswith("中阴_")]
        assert bardo, "应有中阴种子"
        latest = bardo[-1]
        assert latest["imp"] == 1.0
        assert latest["act"] == "LIFE_SUMMARY"
        assert "寿元耗尽" in latest["tag"]
        # 清理：移除测试产生的种子，保持全局状态干净
        del ylg.alaya.seeds[before:]

    def test_bridge_reset_is_rebirth(self):
        """桥接 reset = 转世：种子库延续，世数由死亡计数推导"""
        from desktop.agent_bridge import AgentBridge

        b = AgentBridge(max_steps=5, speed_ms=0)
        deaths_before = ylg._samsara["deaths"]
        snap = b.reset()
        assert snap["lifetime"] == deaths_before + 1
        assert snap["vitality"] == VITALITY_INIT
        assert snap["death_cause"] == ""

    def test_snapshot_schema(self):
        from desktop.agent_bridge import AgentBridge

        b = AgentBridge(max_steps=3, speed_ms=0)
        snap = b.step_once()
        for key in ("vitality", "death_cause", "klesha", "lifetime", "dream_sessions"):
            assert key in snap, f"快照缺 {key}"
        for key in ("greed", "aversion", "delusion"):
            assert key in snap["klesha"]
        assert 0 <= snap["vitality"] <= 130
        assert snap["lifetime"] >= 1


class TestDeathDream:
    """中阴梦境：命终时重组一生经验（离线学习）"""

    def _seeded_state(self, vitality=1.0, n_seeds=8):
        """注入足量高 imp 种子（梦境重组需要 >=3 个）"""
        _emb = [0.05, 0.05] + [0.0] * 9
        before = len(ylg.alaya.seeds)
        for i in range(n_seeds):
            ylg.alaya.seeds.append(
                {"act": "UP" if i % 2 else "DOWN", "rew": 2.0 + i * 0.1, "imp": 0.9,
                 "emb": list(_emb), "seed_type": "业种", "tag": "梦测", "ts": 0.0}
            )
        return _state(action="DOWN", vitality=vitality), before

    def test_death_triggers_dream(self):
        import asyncio

        from yogacara_agent.constants import MEMORY_CAPACITY

        ylg.env.reset()
        s, before = self._seeded_state()
        dreams_before = len(ylg._get_awakening_engine().dream_sessions)
        try:
            s = asyncio.run(ylg.node_execute(s))
            assert s["death_cause"] == "寿元耗尽"
            s = asyncio.run(ylg.node_consolidate(s))
            assert 0 < s["dream_seeds"] <= 20, "命终应触发中阴梦境（单场上限 20）"
            assert len(ylg._get_awakening_engine().dream_sessions) == dreams_before + 1
            # 梦中种子入库且可检索（有 emb）
            dream_children = [x for x in ylg.alaya.seeds if x.get("tag") == "dream_generated"]
            assert len(dream_children) >= s["dream_seeds"], "梦中种子应全部入库"
            for child in dream_children:
                assert "emb" in child, "梦中种子必须有 emb（检索安全）"
            # 容量守恒：淡忘后不超上限
            assert len(ylg.alaya.seeds) <= MEMORY_CAPACITY
            # 洞察含梦境行
            assert any("中阴梦境" in i for i in s["turning_result"]["insights"])
        finally:
            dream_tags = {"dream_generated", "梦测"}
            ylg.alaya.seeds[:] = [x for x in ylg.alaya.seeds if x.get("tag") not in dream_tags]

    def test_few_seeds_no_dream(self):
        """经验不足（<3 种子）不成梦，不抛异常"""
        import asyncio

        saved = list(ylg.alaya.seeds)
        ylg.alaya.seeds.clear()
        try:
            s = _state(action="DOWN", vitality=1.0)
            s = asyncio.run(ylg.node_execute(s))
            s = asyncio.run(ylg.node_consolidate(s))
            assert s["dream_seeds"] == 0
        finally:
            ylg.alaya.seeds[:] = saved

    def test_alive_no_dream(self):
        """存活时不做梦（梦只发生在命终）"""
        import asyncio

        ylg.env.reset()
        s, before = self._seeded_state(vitality=50.0)
        try:
            s = asyncio.run(ylg.node_execute(s))
            assert not s["done"]
            s = asyncio.run(ylg.node_consolidate(s))
            assert s["dream_seeds"] == 0
        finally:
            ylg.alaya.seeds[:] = ylg.alaya.seeds[:before]


class TestLifeHistory:
    """轮回史：每世一行的生命总结"""

    def test_death_appends_life_record(self):
        import asyncio

        ylg.env.reset()
        history_before = len(ylg.get_life_history())
        s, before = TestDeathDream()._seeded_state()
        try:
            s = asyncio.run(ylg.node_execute(s))
            s = asyncio.run(ylg.node_consolidate(s))
            history = ylg.get_life_history()
            assert len(history) == history_before + 1, "命终应记录一世"
            rec = history[-1]
            for key in ("lifetime", "steps", "reward", "death_cause",
                        "turning_level", "klesha", "dream_seeds", "seeds_total"):
                assert key in rec, f"轮回史缺 {key}"
            assert rec["death_cause"] == "寿元耗尽"
            assert rec["lifetime"] >= 1
            assert 0 <= rec["dream_seeds"] <= 20
        finally:
            dream_tags = {"dream_generated", "梦测"}
            ylg.alaya.seeds[:] = [x for x in ylg.alaya.seeds if x.get("tag") not in dream_tags]

    def test_alive_appends_nothing(self):
        import asyncio

        ylg.env.reset()
        history_before = len(ylg.get_life_history())
        s = _state(action="DOWN", vitality=50.0)
        s = asyncio.run(ylg.node_execute(s))
        asyncio.run(ylg.node_consolidate(s))
        assert len(ylg.get_life_history()) == history_before

    def test_bridge_snapshot_history(self):
        from desktop.agent_bridge import AgentBridge

        b = AgentBridge(max_steps=3, speed_ms=0)
        snap = b.step_once()
        assert "life_history" in snap
        assert isinstance(snap["life_history"], list)


class TestCapacityEnforcement:
    """阿赖耶识容量守恒（核心 add() 强制 MEMORY_CAPACITY）"""

    def test_add_enforces_capacity(self, tmp_path):

        from yogacara_agent.alaya_persistent import PersistentAlayaMemory
        from yogacara_agent.constants import MEMORY_CAPACITY

        mem = PersistentAlayaMemory(storage="file", path=str(tmp_path / "cap.jsonl"))
        emb = [0.0] * 11
        for i in range(MEMORY_CAPACITY + 15):
            mem.add({"emb": list(emb), "act": "UP", "rew": 1.0, "ts": float(i)})
        assert len(mem.seeds) == MEMORY_CAPACITY, "超容即淡忘"
        # 淘汰的是最旧（ts 最小）
        assert min(s.get("ts", 0.0) for s in mem.seeds) >= 15.0
        # 文件同步（重载后仍守恒）
        mem2 = PersistentAlayaMemory(storage="file", path=str(tmp_path / "cap.jsonl"))
        assert len(mem2.seeds) <= MEMORY_CAPACITY
