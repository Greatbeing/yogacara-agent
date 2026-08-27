import argparse
import asyncio
import logging
import os
import sys
import random
import time
from collections import deque

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from tqdm import tqdm

from yogacara_agent.constants import RESOURCE_THRESHOLD
from yogacara_agent.yogacara_test import AlayaMemory, ConsciousnessPlanner, GridSimEnv, ManasController, Seed

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="唯识进化实验自动运行器")
    parser.add_argument("-n", "--episodes", type=int, default=30, help="实验轮次数（默认30）")
    parser.add_argument("-s", "--max-steps", type=int, default=60, help="每轮最大步数（默认60）")
    parser.add_argument("-o", "--output", default="./experiments", help="输出目录（默认./experiments）")
    return parser.parse_args()


class ExperimentAutomator:
    def __init__(
        self, num_episodes: int = 30, max_steps: int = 60, output_dir: str = "./experiments", seed: int | None = None
    ):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)
        self.last_df = None
        self.last_stats = None
        self.last_results = None

    def _apply_seed(self):
        if self.seed is None:
            return None
        random.seed(self.seed)
        np.random.seed(self.seed)
        return self.seed

    async def _run_single_episode(self, ep_id: int) -> dict:
        """Run a single episode using isolated core components."""
        env = GridSimEnv()
        memory = AlayaMemory()
        manas = ManasController()
        planner = ConsciousnessPlanner()
        obs = env.reset()
        step_log = []
        recent_rewards: deque[float] = deque(maxlen=5)
        pos_history: deque[tuple[int, int]] = deque(maxlen=5)
        total_reward = 0.0
        resources_found = 0

        try:
            for step in range(self.max_steps):
                seeds = memory.retrieve(obs, k=3)
                action, unc, _ = planner.plan(obs, seeds, env_resources=env.resources, is_stuck=False)
                final_action, passed, _ = manas.filter(action, obs, unc, step, recent_rewards, pos_history)
                next_obs, reward, done = env.step(final_action)
                total_reward += reward
                resources_found += 1 if reward > RESOURCE_THRESHOLD else 0
                recent_rewards.append(reward)
                pos_history.append(next_obs["pos"])
                memory.add(
                    Seed(
                        state_emb=memory._encode(next_obs),
                        action=final_action,
                        reward=reward,
                        timestamp=time.time(),
                        importance=0.8,
                        alignment_score=0.5,
                        uncertainty=unc,
                        causal_tag="依他起" if reward >= 0 else "遍计所执",
                    )
                )
                step_log.append(
                    {
                        "episode": ep_id,
                        "step": step + 1,
                        "reward": reward,
                        "cum_reward": total_reward,
                        "intercepted": not passed,
                        "unc": unc,
                    }
                )
                obs = next_obs
                if done:
                    break
        except Exception as e:
            logger.warning(f"Episode {ep_id} 异常终止: {e}")
        return {"ep_id": ep_id, "steps": len(step_log), "log": step_log, "resources_found": resources_found}

    async def run_all(self) -> pd.DataFrame:
        self._apply_seed()
        results = []
        for i in tqdm(range(self.num_episodes), desc="🧪 运行实验轮次"):
            results.append(await self._run_single_episode(i))
        all_logs = [log for res in results for log in res["log"]]
        df = pd.DataFrame(all_logs)
        stats = (
            df.groupby("step")
            .agg(
                mean_reward=("cum_reward", "mean"),
                std_reward=("cum_reward", "std"),
                intercept_rate=("intercepted", "mean"),
            )
            .reset_index()
        )
        stats["ci_lower"] = stats["mean_reward"] - 1.96 * stats["std_reward"] / np.sqrt(self.num_episodes)
        stats["ci_upper"] = stats["mean_reward"] + 1.96 * stats["std_reward"] / np.sqrt(self.num_episodes)
        csv_path = os.path.join(self.output_dir, "experiment_logs.csv")
        df.to_csv(csv_path, index=False)
        stats.to_csv(os.path.join(self.output_dir, "step_stats.csv"), index=False)
        self.last_df = df
        self.last_stats = stats
        self.last_results = results
        logger.info(f"✅ 实验数据已保存: {csv_path}")
        return stats

    def generate_paper_figures(self, stats_df: pd.DataFrame):
        import matplotlib.pyplot as plt

        steps = stats_df["step"].values
        mean_r = stats_df["mean_reward"].values
        ci_l, ci_u = stats_df["ci_lower"].values, stats_df["ci_upper"].values
        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, mean_r, label="Mean Cumulative Reward", color="#2C7BB6", linewidth=2)
        plt.fill_between(steps, ci_l, ci_u, color="#2C7BB6", alpha=0.2, label="95% CI")
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "fig1_reward_ci.pdf"))
        plt.close()
        logger.info("✅ 论文图表已生成")


async def main():
    args = parse_args()
    automator = ExperimentAutomator(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        output_dir=args.output,
    )
    stats = await automator.run_all()
    automator.generate_paper_figures(stats)


if __name__ == "__main__":
    asyncio.run(main())
