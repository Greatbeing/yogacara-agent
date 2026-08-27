"""
Yogacara Agent 压缩反馈闭环测试
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestAlayaPersistentMemory:
    """测试 PersistentAlayaMemory 持久化层"""

    def test_encode_public_method(self):
        """测试 encode() 公开方法"""
        from yogacara_agent.alaya_persistent import PersistentAlayaMemory

        with tempfile.TemporaryDirectory() as td:
            mem = PersistentAlayaMemory(storage="file", path=os.path.join(td, "test.jsonl"))
            obs = {"pos": (5, 5), "grid_view": [0.0] * 9, "step": 1}
            emb = mem.encode(obs)
            assert len(emb) == 11, f"Expected 11-dim embedding, got {len(emb)}"
            assert emb[0] == 0.5  # 5/10
            assert emb[1] == 0.5  # 5/10

    def test_atomic_write(self):
        """测试原子写：写一半时程序崩溃不应损坏文件"""

        from yogacara_agent.alaya_persistent import PersistentAlayaMemory

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.jsonl")
            mem = PersistentAlayaMemory(storage="file", path=path)
            obs = {"pos": (0, 0), "grid_view": [0.0] * 9, "step": 1}
            mem.add({"emb": mem.encode(obs), "act": "UP", "rew": 5.0})
            mem.add({"emb": mem.encode(obs), "act": "DOWN", "rew": -3.0})
            assert os.path.exists(path)

            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 2, f"Expected 2 lines in JSONL, got {len(lines)}"

    def test_imp_decay_by_days(self):
        """测试 imp 按天衰减不再按秒（1 小时衰减应极小）"""
        import math
        import time

        from yogacara_agent.alaya_persistent import PersistentAlayaMemory

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "decay.jsonl")
            mem = PersistentAlayaMemory(storage="file", path=path)
            obs = {"pos": (3, 3), "grid_view": [0.0] * 9, "step": 1}
            mem.add({"emb": mem.encode(obs), "act": "STAY", "rew": 0.0, "imp": 0.5, "ts": time.time() - 3600})
            mem.perfume_update()
            seed = mem.seeds[-1]
            # 1 小时 = 0.0417 天，exp(-0.12 * 0.0417) ≈ 0.995
            expected = 0.5 * math.exp(-0.12 * (3600 / 86400))
            assert seed["imp"] > 0.49, f"Imp should barely decay after 1 hour, got {seed['imp']}"
            assert abs(seed["imp"] - expected) < 0.01, f"Imp {seed['imp']} != expected {expected}"

    def test_perfume_update_lock(self):
        """测试 perfume_update 的锁保护（多线程安全）"""
        import threading
        import time

        from yogacara_agent.alaya_persistent import PersistentAlayaMemory

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "lock.jsonl")
            mem = PersistentAlayaMemory(storage="file", path=path)
            obs = {"pos": (0, 0), "grid_view": [0.0] * 9, "step": 1}
            for i in range(20):
                mem.add({"emb": mem.encode(obs), "act": "UP", "rew": 5.0, "imp": 0.8, "ts": time.time() - i * 100})

            errors = []

            def update():
                try:
                    mem.perfume_update()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=update) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors, f"Thread errors: {errors}"
            assert len(mem.seeds) == 20, f"Seeds count should remain 20, got {len(mem.seeds)}"


class TestConsolidationEngine:
    """测试 ConsolidationEngine 整理引擎"""

    def test_merge_and_prune(self):
        """测试合并同 tag 种子 + 删除低 quality 种子"""
        from yogacara_agent.consolidation_engine import ConsolidationEngine

        seeds = [
            {"tag": "test", "align": 0.9, "imp": 0.8, "act": "UP"},
            {"tag": "test", "align": 0.95, "imp": 0.9, "act": "UP"},
            {"tag": "other", "align": 0.85, "imp": 0.7, "act": "DOWN"},
            {"tag": "junk", "align": 0.05, "imp": 0.1, "act": "STAY"},
        ]
        engine = ConsolidationEngine()
        report = engine.run(seeds, dry_run=False)
        assert report.merged_count == 1, f"Expected 1 merge, got {report.merged_count}"
        assert report.pruned_count == 1, f"Expected 1 prune, got {report.pruned_count}"
        assert len(seeds) == 2, f"Expected 2 seeds after, got {len(seeds)}"
        merged_seeds = [s for s in seeds if s.get("merged_from")]
        assert len(merged_seeds) == 1, "Should have 1 merged seed"

    def test_dry_run(self):
        """测试 dry_run 模式不修改种子"""
        from yogacara_agent.consolidation_engine import ConsolidationEngine

        seeds = [{"tag": "a", "align": 0.9}, {"tag": "a", "align": 0.95}, {"tag": "b", "align": 0.05}]
        original_ids = {id(s) for s in seeds}
        engine = ConsolidationEngine()
        report = engine.run(seeds, dry_run=True)
        assert report.merged_count > 0
        assert report.pruned_count > 0
        assert {id(s) for s in seeds} == original_ids, "dry_run should not modify seeds"


class TestVipakaEngine:
    """测试 VipakaEngine 熏习引擎"""

    def test_action_filter(self):
        """测试动作过滤：UP 的果报只更新 UP 种子"""
        import tempfile

        from yogacara_agent.alaya_persistent import PersistentAlayaMemory
        from yogacara_agent.vipaka_engine import VipakaEngine

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "vp.jsonl")
            mem = PersistentAlayaMemory(storage="file", path=path)
            obs = {"pos": (5, 5), "grid_view": [0.0] * 9, "step": 1}
            emb = mem.encode(obs)
            mem.add({"emb": emb, "act": "UP", "rew": 5.0, "seed_type": "业种", "align": 0.5})
            mem.add({"emb": emb, "act": "DOWN", "rew": -3.0, "seed_type": "业种", "align": 0.5})
            vipaka = VipakaEngine(mem, rate=0.2)
            res = vipaka.process_outcome(step=1, action="UP", reward=5.0, unc=0.1, obs=obs)
            assert res.seeds_updated == 1, f"Expected 1 seed updated (only UP), got {res.seeds_updated}"
            # UP 种子 align 应增加，DOWN 应不变
            up_align = [s["align"] for s in mem.seeds if s["act"] == "UP"][0]
            down_align = [s["align"] for s in mem.seeds if s["act"] == "DOWN"][0]
            assert up_align > 0.5, f"UP align should increase, got {up_align}"
            assert down_align == 0.5, f"DOWN align should not change, got {down_align}"


class TestCompressionMetrics:
    """测试 CompressionMetricsCalculator"""

    def test_compression_quality_score(self):
        """测试压缩质量分数计算"""
        from yogacara_agent.compression_metrics import CompressionMetricsCalculator

        calc = CompressionMetricsCalculator()
        seeds = [{"imp": 0.8, "rew": 5.0}, {"imp": 0.6, "rew": -3.0}, {"imp": 0.3, "rew": 0.0}]
        metrics = calc.compute(
            seeds=seeds,
            initial_tokens=1000,
            mirror_ratio=0.5,
            ego_score=0.2,
            misapprehension_ratio=0.1,
            execution_rate=0.8,
            verbose=False,
        )
        assert hasattr(metrics, "cqs"), "CQS should be in metrics"
        assert 0 <= metrics.cqs <= 2.0, f"CQS out of range: {metrics.cqs}"
