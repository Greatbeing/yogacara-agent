"""
转依引擎接入 LangGraph 主管线的集成测试
========================================
验证 node_consolidate 中的 TurningConsciousnessEngine 集成：
种子适配器、softmax 归一化、reward_context 接线、净化回写。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from yogacara_agent import yogacara_langgraph as ylg
from yogacara_agent.constants import (
    TURNING_PURITY_THRESHOLD,
    VIPAKA_FUNCTIONAL_CLARITY,
)


def _init_state(step_limit: int = 5) -> dict:
    return {
        "obs": ylg.env.observe(),
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
    }


async def _one_cycle(state: dict) -> dict:
    state = await ylg.node_perceive(state)
    state = await ylg.node_plan(state)
    state = await ylg.node_manas(state)
    state = await ylg.node_execute(state)
    state = await ylg.node_introspect(state)
    state = await ylg.node_store(state)
    state = await ylg.node_consolidate(state)
    return state


class TestAdapters:
    """适配器单元测试"""

    def test_softmax_normalization(self):
        """softmax 后概率和为 1，且全部为正"""
        scores = {"UP": 2.5, "DOWN": -1.0, "LEFT": 0.3, "RIGHT": 1.1, "STAY": -0.4}
        probs = ylg._softmax_scores(scores)
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert all(p > 0 for p in probs.values())
        # 分数最高者概率最大
        assert max(probs, key=probs.get) == "UP"

    def test_softmax_empty(self):
        assert ylg._softmax_scores({}) == {}
        assert ylg._softmax_scores(None) == {}

    def test_to_turning_seeds_mapping(self):
        """alaya dict → Seed 字段映射正确"""
        seeds = [
            {"act": "UP", "rew": 5.0, "align": 0.85, "seed_type": "业种", "tag": "业_正反馈_UP"},
            {"act": "DOWN", "rew": -3.0, "align": 0.45, "seed_type": "名言种", "tag": "名言_遍计"},
            {"act": "LEFT", "rew": -0.1, "align": 0.25, "seed_type": "异熟种", "tag": "异熟_连续失败"},
        ]
        t_seeds, indices = ylg._to_turning_seeds(seeds)
        assert indices == [0, 1, 2]
        assert t_seeds[0].is_defiled is False          # 普通 tag 无"遍计"
        assert t_seeds[0].clarity == 0.85              # 非 vipaka → clarity = align
        assert t_seeds[1].is_defiled is True           # tag 含"遍计" → 染污
        assert t_seeds[2].is_defiled is False          # 异熟种非染污
        assert t_seeds[2].clarity == VIPAKA_FUNCTIONAL_CLARITY  # 异熟种受保护

    def test_reward_context_wiring(self):
        """ego_alert.triggered → reward_context 含 'self'（触发我执消解）"""
        s = _init_state()
        assert "self" not in ylg._reward_context_of(s)
        s["ego_alert"] = {"triggered": True, "ego_score": 0.7}
        assert "self" in ylg._reward_context_of(s)


class TestTurningIntegration:
    """node_consolidate 集成测试"""

    def test_turning_result_in_state(self):
        """完整 cycle 后 state['turning_result'] 字段齐全且为 Python 原生类型"""
        import asyncio

        state = asyncio.run(_one_cycle(_init_state()))
        tr = state["turning_result"]
        assert tr is not None, "node_consolidate 应写入 turning_result"
        for key in ("mirror", "equality", "observation", "action", "turning_level",
                    "defiled_removed", "ego_dissolved", "insights"):
            assert key in tr, f"缺少字段 {key}"
        # JSON 可序列化（无 numpy 标量）
        import json
        json.dumps(tr)
        # 数值范围合理
        for k in ("mirror", "equality", "observation", "action", "turning_level"):
            assert 0.0 <= tr[k] <= 1.0, f"{k}={tr[k]} 超出 [0,1]"

    def test_defiled_seeds_pruned(self):
        """注入遍计染污种后运行被移除，异熟种受保护（去染存净）"""
        import asyncio

        before = len(ylg.alaya.seeds)
        _emb = [0.0] * 11  # retrieve() 需要 emb 字段（11 维）
        ylg.alaya.seeds.extend(
            [
                {"act": "LEFT", "rew": -0.1, "align": 0.50, "seed_type": "名言种", "tag": "名言_遍计", "emb": list(_emb)},
                {"act": "RIGHT", "rew": -0.1, "align": 0.45, "seed_type": "名言种", "tag": "名言_遍计", "emb": list(_emb)},
                {"act": "UP", "rew": -0.1, "align": 0.25, "seed_type": "异熟种", "tag": "异熟_连续失败", "emb": list(_emb)},
                {"act": "DOWN", "rew": 5.0, "align": 0.85, "seed_type": "业种", "tag": "业_正反馈_DOWN", "emb": list(_emb)},
            ]
        )
        try:
            state = asyncio.run(_one_cycle(_init_state()))
            tr = state["turning_result"]
            assert tr["defiled_removed"] >= 2, "应至少移除 2 个遍计染污种"
            tags_left = [s.get("tag", "") for s in ylg.alaya.seeds]
            assert all("遍计" not in t for t in tags_left), "遍计染污种应全部净化"
            types_left = [s.get("seed_type") for s in ylg.alaya.seeds]
            assert "异熟种" in types_left, "异熟种（模式追踪器）不应被误删"
        finally:
            del ylg.alaya.seeds[before:]  # 清理测试注入

    def test_turning_insights_chinese(self):
        """洞察文案为中文（供桌面日志流金色高亮显示）"""
        import asyncio

        state = asyncio.run(_one_cycle(_init_state()))
        for insight in state["turning_result"]["insights"]:
            assert any("\u4e00" <= ch <= "\u9fff" for ch in insight), "洞察应含中文字符"

    def test_engine_singleton(self):
        """转依引擎为懒初始化单例（跨调用保持我执状态）"""
        e1 = ylg._get_turning_engine()
        e2 = ylg._get_turning_engine()
        assert e1 is e2
        assert e1.alaya_purifier.purity_threshold == TURNING_PURITY_THRESHOLD


class TestDesktopBridgeTurning:
    """桌面桥接层快照扩展"""

    def test_snapshot_contains_turning(self):
        from desktop.agent_bridge import AgentBridge

        b = AgentBridge(max_steps=5, speed_ms=0)
        snap = b.step_once()
        assert "turning" in snap, "快照应含 turning 字段"
        t = snap["turning"]
        assert "turning_level" in t and "insights" in t
        # 日志含转依洞察行（nature=圆成实, action=转依）且 seq 递增
        insight_rows = [entry for entry in snap["logs"] if entry["action"] == "转依"]
        assert all(r["nature"] == "圆成实" for r in insight_rows)
        if insight_rows:
            seqs = [entry["seq"] for entry in snap["logs"]]
            assert seqs == sorted(seqs), "日志 seq 应单调递增"
