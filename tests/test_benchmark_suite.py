"""学术基准测试套件测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd


def test_benchmark_runner_writes_report(tmp_path, monkeypatch):
    from yogacara_agent import exp_automator
    from yogacara_agent.benchmark_suite import BenchmarkConfig, BenchmarkRunner

    async def fake_run_all(self):
        self.last_stats = pd.DataFrame(
            [
                {"step": 1, "mean_reward": 1.0, "std_reward": 0.0, "intercept_rate": 0.0, "ci_lower": 1.0, "ci_upper": 1.0},
                {"step": 2, "mean_reward": 2.0, "std_reward": 0.0, "intercept_rate": 0.5, "ci_lower": 2.0, "ci_upper": 2.0},
            ]
        )
        self.last_results = [{"resources_found": 1}, {"resources_found": 0}]
        return self.last_stats

    monkeypatch.setattr(exp_automator.ExperimentAutomator, "run_all", fake_run_all)

    cfg = BenchmarkConfig(episodes=2, max_steps=5, seeds=[7], output_dir=str(tmp_path), no_figures=True)
    summary = asyncio.run(BenchmarkRunner(cfg).run())

    run_dir = Path(tmp_path) / summary["run_id"]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.txt").exists()
    assert summary["metrics"]["resource_found_rate"] == 0.5
    assert summary["metrics"]["manas_intercept_rate"] == 0.25


def test_history_compare(tmp_path):
    from yogacara_agent.benchmark_suite import HistoryComparator

    base = Path(tmp_path)
    previous = base / "20260101-000000"
    previous.mkdir(parents=True)
    (previous / "summary.json").write_text('{"run_id":"20260101-000000","metrics":{"cumulative_reward":{"mean":1.0}}}', encoding="utf-8")

    current = base / "20260102-000000"
    current.mkdir(parents=True)
    result = HistoryComparator().compare(base, current.name)
    assert result is not None
    assert result["previous_run_id"] == "20260101-000000"
