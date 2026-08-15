"""学术基准测试套件。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yogacara_agent.exp_automator import ExperimentAutomator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    episodes: int = 30
    max_steps: int = 60
    seeds: list[int] = field(default_factory=list)
    output_dir: str = "./experiments"
    no_figures: bool = False


def _seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _latest_summary_file(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("*/summary.json"))
    return candidates[-1] if candidates else None


class MetricAggregator:
    def aggregate(self, stats_by_seed: list[pd.DataFrame], episode_results: list[dict[str, Any]], config: BenchmarkConfig) -> dict[str, Any]:
        combined = pd.concat(stats_by_seed, ignore_index=True) if stats_by_seed else pd.DataFrame()
        if combined.empty:
            return {
                "cumulative_reward": {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
                "resource_found_rate": 0.0,
                "manas_intercept_rate": 0.0,
                "avg_steps": 0.0,
                "reward_per_step": 0.0,
            }

        mean_reward = float(combined["mean_reward"].mean())
        std_reward = float(combined["mean_reward"].std(ddof=0) or 0.0)
        episode_count = max(1, len(combined))
        ci_delta = 1.96 * std_reward / np.sqrt(episode_count)
        resource_found_rate = 0.0
        if episode_results:
            resource_found_rate = float(sum(1 for item in episode_results if item.get("resources_found", 0) > 0) / len(episode_results))
        return {
            "cumulative_reward": {
                "mean": round(mean_reward, 4),
                "std": round(std_reward, 4),
                "ci_lower": round(mean_reward - ci_delta, 4),
                "ci_upper": round(mean_reward + ci_delta, 4),
            },
            "resource_found_rate": round(resource_found_rate, 4),
            "manas_intercept_rate": round(float(combined.get("intercept_rate", pd.Series(dtype=float)).mean() or 0.0), 4),
            "avg_steps": round(float(combined["step"].mean()), 4),
            "reward_per_step": round(float(combined["mean_reward"].mean() / max(1.0, combined["step"].mean())), 4),
            "wisdom": {},
        }


class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def to_json(self, summary: dict[str, Any]) -> Path:
        path = self.output_dir / "summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def to_text(self, summary: dict[str, Any]) -> Path:
        lines = [
            f"Benchmark run: {summary['run_id']}",
            f"Episodes: {summary['config']['episodes']}",
            f"Max steps: {summary['config']['max_steps']}",
            f"Cumulative reward mean: {summary['metrics']['cumulative_reward']['mean']}",
            f"Resource found rate: {summary['metrics']['resource_found_rate']}",
            f"Manas intercept rate: {summary['metrics']['manas_intercept_rate']}",
        ]
        path = self.output_dir / "summary.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def generate_figures(self, stats_frames: list[pd.DataFrame], no_figures: bool) -> list[Path]:
        if no_figures:
            return []
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return []

        paths: list[Path] = []
        if stats_frames:
            merged = pd.concat(stats_frames, ignore_index=True)
            steps = merged["step"].values
            mean_r = merged["mean_reward"].values
            ci_l = merged["ci_lower"].values
            ci_u = merged["ci_upper"].values
            plt.figure(figsize=(8, 4.5))
            plt.plot(steps, mean_r, label="Mean Cumulative Reward", color="#2C7BB6", linewidth=2)
            plt.fill_between(steps, ci_l, ci_u, color="#2C7BB6", alpha=0.2, label="95% CI")
            plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            plt.xlabel("Step")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            pdf_path = self.output_dir / "fig1_reward_ci.pdf"
            png_path = self.output_dir / "fig1_reward_ci.png"
            plt.savefig(pdf_path)
            plt.savefig(png_path)
            plt.close()
            paths.extend([pdf_path, png_path])
        return paths


class HistoryComparator:
    def compare(self, output_dir: Path, current_run_id: str) -> dict[str, Any] | None:
        candidates = sorted(output_dir.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
        previous = None
        for candidate in candidates:
            if candidate.parent.name != current_run_id:
                previous = candidate
        if previous is None:
            return None
        try:
            previous_data = json.loads(previous.read_text(encoding="utf-8"))
        except Exception:
            return None
        return {
            "previous_run_id": previous_data.get("run_id"),
            "previous_reward": previous_data.get("metrics", {}).get("cumulative_reward", {}),
        }


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _run_seed_group(self, seed: int | None, run_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        if seed is not None:
            _seed_rng(seed)
        seed_dir = run_dir / (f"seed-{seed}" if seed is not None else "seed-default")
        automator = ExperimentAutomator(
            num_episodes=self.config.episodes,
            max_steps=self.config.max_steps,
            output_dir=str(seed_dir),
            seed=seed,
        )
        await automator.run_all()
        stats = automator.last_stats if automator.last_stats is not None else pd.DataFrame()
        results = automator.last_results if automator.last_results is not None else []
        return stats, results

    async def run(self) -> dict[str, Any]:
        seeds = self.config.seeds or [None]
        stats_by_seed: list[pd.DataFrame] = []
        episode_results: list[dict[str, Any]] = []
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if seeds and seeds[0] is not None:
            run_id = f"{run_id}-{seeds[0]}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            stats, results = await self._run_seed_group(seed, run_dir)
            if not stats.empty:
                stats_by_seed.append(stats)
            episode_results.extend(results)

        aggregator = MetricAggregator()
        metrics = aggregator.aggregate(stats_by_seed, episode_results, self.config)

        summary = {
            "config": asdict(self.config),
            "run_id": run_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "seed_used": seeds,
        }

        generator = ReportGenerator(run_dir)
        generator.to_json(summary)
        generator.to_text(summary)
        figures = generator.generate_figures(stats_by_seed, self.config.no_figures)
        summary["generated_files"] = [str(p.name) for p in figures]
        comparison = HistoryComparator().compare(self.output_dir, run_id)
        if comparison is not None:
            summary["history_compare"] = comparison
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


def _parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="唯识进化基准测试套件")
    parser.add_argument("-n", "--episodes", type=int, default=30)
    parser.add_argument("-s", "--max-steps", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument("-o", "--output", default="./experiments")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_steps < 1:
        raise SystemExit("episodes and max-steps must be positive")
    return BenchmarkConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seeds=args.seeds,
        output_dir=args.output,
        no_figures=args.no_figures,
    )


async def main(argv: list[str] | None = None):
    config = _parse_args(argv)
    runner = BenchmarkRunner(config)
    summary = await runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
