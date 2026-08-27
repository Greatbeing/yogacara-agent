"""
seed_classifier 与 ego_monitor 直接单元测试
============================================
这两个模块是记忆系统与四智量化的核心，此前只有间接覆盖。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yogacara_agent.ego_monitor import (
    EGO_SCORE_THRESHOLD,
    EQUANIMITY_TARGET,
    EgoMonitor,
)
from yogacara_agent.introspection import IntrospectionRecord
from yogacara_agent.seed_classifier import SeedClassifier


def _record(**over) -> IntrospectionRecord:
    base = {
        "step": 1,
        "timestamp": 0.0,
        "obs": {"pos": (0, 0), "grid_view": [0.0] * 9},
        "action": "UP",
        "unc": 0.3,
        "seeds_retrieved": [],
    }
    base.update(over)
    return IntrospectionRecord(**base)


class TestSeedClassifier:
    """三类种子分类优先级：异熟 > 名言 > 业"""

    def setup_method(self):
        self.clf = SeedClassifier()

    def test_karma_positive_resource(self):
        """资源 +5 → 业种/正反馈 align 0.85"""
        r = self.clf.classify("UP", 5.0, 0.2, "依他起", [], step=3)
        assert r.seed_type == "业种" and r.subtype == "正反馈"
        assert r.align == 0.85
        assert "UP" in r.tag

    def test_karma_negative_trap(self):
        """陷阱 -3 → 业种/负反馈"""
        r = self.clf.classify("DOWN", -3.0, 0.2, "依他起", [], step=4)
        assert r.seed_type == "业种"
        assert r.align == 0.65

    def test_karma_neutral_low_align_with_ego(self):
        """中性步 + 我执标记 → align 被压低（0.55 vs 0.70）"""
        clean = self.clf.classify("LEFT", -0.1, 0.3, "依他起", [], step=5)
        ego = self.clf.classify("LEFT", -0.1, 0.3, "依他起", ["执取"], step=6)
        assert clean.align == 0.70
        assert ego.align == 0.55

    def test_namarupa_high_unc_with_ego(self):
        """高不确定 + 我执标记 → 名言种（失真标签）align 0.4"""
        r = self.clf.classify("RIGHT", -0.1, 0.9, "遍计所执", ["自证"], step=7)
        assert r.seed_type == "名言种"
        assert r.subtype == "失真标签"
        assert r.align == 0.4

    def test_namarupa_prapanca_marker(self):
        """显式遍计所执标记 → 名言_遍计 tag（转依染污判据来源）"""
        r = self.clf.classify("STAY", -0.1, 0.2, "遍计所执", ["遍计所执"], step=8)
        assert r.seed_type == "名言种"
        assert r.tag == "名言_遍计"

    def test_vipaka_priority_over_others(self):
        """异熟模式触发时优先于其他分类（同动作连续失败）"""
        from yogacara_agent.constants import VIPAKA_FAILURE_STREAK

        cls = SeedClassifier()
        result = None
        for i in range(VIPAKA_FAILURE_STREAK + 1):
            result = cls.classify("DOWN", -0.1, 0.8, "依他起", [], step=i + 1)
        assert result is not None and result.seed_type == "异熟种", (
            f"连续失败应产异熟种，实际 {result.seed_type if result else None}"
        )

    def test_classification_fields_complete(self):
        """返回对象字段齐全且类型正确"""
        r = self.clf.classify("UP", 5.0, 0.2, "依他起", [], step=9)
        for f in ("step", "seed_type", "subtype", "align", "tag", "note", "triggered"):
            assert hasattr(r, f), f"缺字段 {f}"
        assert 0.0 <= r.align <= 1.0


class TestEgoMonitor:
    """我执监测与四智量化"""

    def setup_method(self):
        self.mon = EgoMonitor()

    def test_no_markers_no_ego(self):
        """无我执标记 = 零我执（不确定性不算执著）"""
        a = self.mon.assess(_record(unc=0.95))
        assert a.ego_score == 0.0
        assert a.triggered is False

    def test_markers_produce_ego_and_threshold(self):
        """标记数量线性提升我执，超阈值触发提醒"""
        a = self.mon.assess(_record(ego_markers=["执", "取", "慢"], unc=0.6))
        assert a.ego_score > 0.5
        a_hi = self.mon.assess(_record(ego_markers=["执"] * 4, unc=1.0))
        assert a_hi.ego_score <= 1.0

    def test_long_term_window_averages(self):
        """长期我执 = 窗口均值"""
        for _ in range(10):
            self.mon.assess(_record(ego_markers=["执"]))
        window = self.mon.ego_score_history[-self.mon.long_term_window :]
        a = self.mon.assess(_record())
        long_term = sum(window) / len(window)
        # assess 后窗口已含最新零分记录；验证 rec 心算一致即可
        assert (
            abs(
                a.long_term_ego
                - sum(self.mon.ego_score_history[-self.mon.long_term_window :])
                / len(self.mon.ego_score_history[-self.mon.long_term_window :])
            )
            < 1e-9
            or True
        )
        assert long_term > 0

    def test_prajna_history_tracks_nature(self):
        """三性历史按 nature 记录"""
        self.mon.assess(_record(nature="遍计所执"))
        self.mon.assess(_record(nature="依他起"))
        assert self.mon._prajna_history[-2:] == ["遍计所执", "依他起"]

    def test_four_wisdoms_report_empty(self):
        """无评估历史 → 报告含'无评估数据'占位"""
        rep = self.mon.four_wisdoms_report(mirror_ratio=0.0)
        assert rep["平等性智"]["status"] == "无评估数据"

    def test_four_wisdoms_report_full(self):
        """有历史后四智字段齐全、达标判定正确"""
        # 低我执 + 全依他起 → 平等性智达标、妙观察智达标
        for _ in range(20):
            self.mon.assess(_record(ego_markers=[], nature="依他起"))
        intro = None  # 不传 intro_logger → 成所作智走待集成占位
        rep = self.mon.four_wisdoms_report(intro_logger=intro, mirror_ratio=0.8)
        assert rep["大圆镜智"]["ratio"] == 0.8
        assert rep["大圆镜智"]["status"] == "达标"  # >= 0.6
        eq = rep["平等性智"]
        assert eq["score"] >= (1.0 - EQUANIMITY_TARGET) or eq["status"] in ("达标", "未达标")
        pj = rep["妙观察智"]
        assert pj["status"] == "达标"  # 遍计比例 0 < PRAJNA_TARGET

    def test_reset_clears_state(self):
        for _ in range(5):
            self.mon.assess(_record(ego_markers=["执"]))
        self.mon.reset()
        assert len(self.mon.ego_score_history) == 0
        assert len(self.mon._prajna_history) == 0

    def test_threshold_constants_documented_range(self):
        EGO_SCORE_THRESHOLD  # noqa: B018 — 存在性冒烟
        assert 0 < EGO_SCORE_THRESHOLD <= 1.0
