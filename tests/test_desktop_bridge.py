"""
桌面版 AgentBridge 测试（不打开窗口）
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestAgentBridgeStep:
    """测试单步执行与快照"""

    def _bridge(self):
        from desktop.agent_bridge import AgentBridge

        return AgentBridge(max_steps=10, speed_ms=0)

    def test_step_once_snapshot_schema(self):
        """单步后快照应包含所有 UI 需要的字段"""
        b = self._bridge()
        snap = b.step_once()
        required = {
            "running", "paused", "done", "step", "step_limit",
            "pos", "resources", "traps", "path",
            "action", "reward", "unc", "cumulative_reward",
            "manas_passed", "seeds_total", "seed_types",
            "four_wisdom", "logs",
        }
        missing = required - set(snap.keys())
        assert not missing, f"Snapshot missing keys: {missing}"
        assert snap["step"] == 1
        assert snap["action"] in ("UP", "DOWN", "LEFT", "RIGHT", "STAY")
        assert len(snap["pos"]) == 2
        assert snap["seeds_total"] >= 1, "应已存入种子"

    def test_step_until_done(self):
        """跑满 step_limit 后 done=True，再步进为 no-op"""
        b = self._bridge()
        for _ in range(10):
            if b.state["done"]:
                break
            b.step_once()
        assert b.state["done"], "10 步后应结束"
        before = b.state["step"]
        b.step_once()
        assert b.state["step"] == before, "done 后不应继续步进"

    def test_reset_keeps_memory(self):
        """reset 重置状态但保留种子记忆（阿赖耶识延续）"""
        b = self._bridge()
        for _ in range(3):
            b.step_once()
        seeds_before = b.get_snapshot()["seeds_total"]
        assert seeds_before >= 3
        b.reset()
        snap = b.get_snapshot()
        assert snap["step"] == 0
        assert snap["done"] is False
        assert snap["seeds_total"] == seeds_before, "种子记忆应跨 episode 保留"
        assert snap["logs"] == [], "日志应清空"


class TestAgentBridgeRunControl:
    """测试连续运行/暂停/恢复"""

    def _bridge(self):
        from desktop.agent_bridge import AgentBridge

        return AgentBridge(max_steps=50, speed_ms=10)

    def test_start_pause_resume(self):
        b = self._bridge()
        b.start(max_steps=50, speed_ms=10)
        time.sleep(0.5)
        snap = b.get_snapshot()
        assert snap["running"] is True
        assert snap["step"] > 0, "应已执行若干步"

        b.pause()
        time.sleep(0.1)
        step_paused = b.get_snapshot()["step"]
        assert b.get_snapshot()["paused"] is True
        time.sleep(0.15)
        assert b.get_snapshot()["step"] == step_paused, "暂停后步数不应增长"

        b.resume()
        time.sleep(0.3)
        assert b.get_snapshot()["step"] > step_paused, "恢复后应继续步进"
        b.stop()

    def test_get_seeds(self):
        b = self._bridge()
        for _ in range(5):
            b.step_once()
        seeds = b.get_seeds(limit=3)
        assert len(seeds) == 3
        for s in seeds:
            assert "act" in s and "rew" in s and "seed_type" in s
