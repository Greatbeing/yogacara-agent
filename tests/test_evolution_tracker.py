"""种子演化快照测试。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DummyAlaya:
    seeds: list[dict]


def test_snapshot_counts_and_averages():
    from yogacara_agent.evolution_tracker import EvolutionTracker

    tracker = EvolutionTracker(max_snapshots=3)
    alaya = DummyAlaya(
        seeds=[
            {"seed_type": "业种", "imp": 0.5, "align": 0.2},
            {"seed_type": "名言种", "imp": 1.0, "align": 0.8},
        ]
    )

    snap = tracker.snapshot(alaya)
    assert snap["total_seeds"] == 2
    assert snap["seed_types"] == {"业种": 1, "名言种": 1}
    assert snap["avg_importance"] == 0.75
    assert snap["avg_alignment"] == 0.5


def test_snapshot_limit():
    from yogacara_agent.evolution_tracker import EvolutionTracker

    tracker = EvolutionTracker(max_snapshots=2)
    alaya = DummyAlaya(seeds=[])
    tracker.snapshot(alaya)
    tracker.snapshot(alaya)
    tracker.snapshot(alaya)
    assert len(tracker.get_snapshots()) == 2


def test_empty_snapshots():
    from yogacara_agent.evolution_tracker import EvolutionTracker

    tracker = EvolutionTracker()
    assert tracker.get_snapshots() == []
