import asyncio
import logging
import os

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from yogacara_langgraph import build_graph, env

logger = logging.getLogger(__name__)


class ExperimentAutomator:
    def __init__(self, num_episodes: int = 30, max_steps: int = 60, output_dir: str = "./experiments"):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.graph = build_graph()

    async def _run_single_episode(self, ep_id: int) -> dict:
        env.reset()
        state = {
            "obs": env._observe(),
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
        }
        step_log = []
        try:
            while not state["done"] and state["step"] < self.max_steps:
                state = await self.graph.ainvoke(state)
                step_log.append(
                    {
                        "episode": ep_id,
                        "step": state["step"],
                        "reward": state["reward"],
                        "cum_reward": sum(state["recent_rewards"]),
                        "intercepted": not state["manas_passed"],
                        "unc": state["unc"],
                    }
                )
        except Exception as e:
            logger.warning(f"Episode {ep_id} 异常终止: {e}")
        return {"ep_id": ep_id, "steps": state["step"], "log": step_log}

    async def run_all(self) -> pd.DataFrame:
        tasks = [self._run_single_episode(i) for i in range(self.num_episodes)]
        results = await tqdm_asyncio.gather(*tasks, desc="🧪 运行实验轮次")
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
    automator = ExperimentAutomator(num_episodes=30, max_steps=60)
    stats = await automator.run_all()
    automator.generate_paper_figures(stats)


if __name__ == "__main__":
    asyncio.run(main())
