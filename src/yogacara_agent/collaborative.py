"""进程内多智能体协作模式。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import time
from typing import Any

from yogacara_agent.constants import RESOURCE_THRESHOLD
from yogacara_agent.yogacara_test import AlayaMemory, ConsciousnessPlanner, GridSimEnv, ManasController, Seed


@dataclass
class AgentSession:
    agent_id: str
    env: GridSimEnv
    manas: ManasController
    planner: ConsciousnessPlanner
    recent_rewards: deque[float]
    pos_history: deque[tuple[int, int]]


class CollaborativeCoordinator:
    def __init__(self, agent_count: int, seed: int | None = None, share_alaya: AlayaMemory | None = None):
        if agent_count < 2 or agent_count > 16:
            raise ValueError("agent_count must be within [2, 16]")
        self.agent_count = agent_count
        self.seed = seed
        self.alaya = share_alaya or AlayaMemory()
        self._sessions: dict[str, AgentSession] = {}
        self._cross_agent_retrievals = 0
        self._closed = False

    def create_agent(self, agent_id: str) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("coordinator is closed")
        session = AgentSession(
            agent_id=agent_id,
            env=GridSimEnv(),
            manas=ManasController(),
            planner=ConsciousnessPlanner(),
            recent_rewards=deque(maxlen=5),
            pos_history=deque(maxlen=5),
        )
        self._sessions[agent_id] = session
        return {
            "agent_id": agent_id,
            "env": session.env,
            "manas": session.manas,
            "planner": session.planner,
            "alaya": self.alaya,
        }

    def _ensure_agent(self, agent_id: str) -> AgentSession:
        if agent_id not in self._sessions:
            self.create_agent(agent_id)
        return self._sessions[agent_id]

    def _store_seed(self, agent_id: str, obs: dict[str, Any], action: str, reward: float, uncertainty: float) -> None:
        seed = Seed(
            state_emb=self.alaya._encode(obs),
            action=action,
            reward=reward,
            timestamp=time(),
            importance=0.8,
            alignment_score=max(0.0, min(1.0, 0.5 + reward / 10.0)),
            uncertainty=uncertainty,
            causal_tag="依他起" if reward >= 0 else "遍计所执",
        )
        seed.source_agent = agent_id
        self.alaya.add(seed)

    def run_episode(self, agent_id: str, max_steps: int = 60) -> dict[str, Any]:
        session = self._ensure_agent(agent_id)
        obs = session.env.reset()
        total_reward = 0.0
        resources_found = 0
        cross_agent_hits = 0

        for step in range(max_steps):
            seeds = self.alaya.retrieve(obs, k=3)
            if any(getattr(seed, "source_agent", agent_id) != agent_id for seed in seeds):
                cross_agent_hits += 1
                self._cross_agent_retrievals += 1
            action, unc, _ = session.planner.plan(obs, seeds, env_resources=session.env.resources, is_stuck=False)
            final_action, _, _ = session.manas.filter(
                action, obs, unc, step, session.recent_rewards, session.pos_history
            )
            next_obs, reward, done = session.env.step(final_action)
            session.recent_rewards.append(reward)
            session.pos_history.append(next_obs["pos"])
            total_reward += reward
            # 资源判定与 exp_automator 一致：reward >= RESOURCE_THRESHOLD（STAY 奖励不算）
            resources_found += 1 if reward >= RESOURCE_THRESHOLD else 0
            self._store_seed(agent_id, next_obs, final_action, reward, unc)
            obs = next_obs
            if done:
                break

        return {
            "agent_id": agent_id,
            "steps": session.env.step_count,
            "cumulative_reward": round(total_reward, 2),
            "resources_found": resources_found,
            "cross_agent_seed_usage": cross_agent_hits,
            "seed_count": len(self.alaya.seeds),
        }

    def run_all(self, episodes_per_agent: int = 10, max_steps: int = 60) -> dict[str, Any]:
        per_agent: dict[str, dict[str, Any]] = {}
        # 逐 episode 累计，增益用各 agent 的跨轮均值（而非只看最后一轮）
        reward_sums: dict[str, float] = {}
        resource_sums: dict[str, int] = {}
        cross_sums: dict[str, int] = {}
        for idx in range(self.agent_count):
            agent_id = f"agent-{idx}"
            self.create_agent(agent_id)
            last: dict[str, Any] | None = None
            for _ in range(episodes_per_agent):
                last = self.run_episode(agent_id, max_steps=max_steps)
                reward_sums[agent_id] = reward_sums.get(agent_id, 0.0) + last["cumulative_reward"]
                resource_sums[agent_id] = resource_sums.get(agent_id, 0) + last["resources_found"]
                cross_sums[agent_id] = cross_sums.get(agent_id, 0) + last["cross_agent_seed_usage"]
            n = max(1, episodes_per_agent)
            per_agent[agent_id] = {
                "agent_id": agent_id,
                "mean_reward": round(reward_sums.get(agent_id, 0.0) / n, 2),
                "mean_resources": round(resource_sums.get(agent_id, 0) / n, 2),
                "cross_agent_seed_usage": cross_sums.get(agent_id, 0),
                "last_episode": last,
            }

        seed_contribution: dict[str, int] = {}
        for seed in self.alaya.seeds:
            source = getattr(seed, "source_agent", "unknown")
            seed_contribution[source] = seed_contribution.get(source, 0) + 1

        # 基线 = agent-0（先运行，未受他者种子影响）；均值用跨轮均值
        baseline_mean_reward = per_agent.get("agent-0", {}).get("mean_reward", 0.0)
        collaboration_mean_reward = sum(v["mean_reward"] for v in per_agent.values()) / max(1, len(per_agent))
        collaboration_gain = None
        if baseline_mean_reward != 0:
            collaboration_gain = (collaboration_mean_reward - baseline_mean_reward) / baseline_mean_reward

        return {
            "per_agent": per_agent,
            "cross_agent_retrievals": self._cross_agent_retrievals,
            "seed_contribution": seed_contribution,
            "collaboration_gain": collaboration_gain,
            "baseline_mean_reward": baseline_mean_reward,
            "collaboration_mean_reward": collaboration_mean_reward,
        }

    def collaboration_summary(self) -> dict[str, Any]:
        return {
            "agent_count": self.agent_count,
            "cross_agent_retrievals": self._cross_agent_retrievals,
            "seed_count": len(self.alaya.seeds),
            "agents": list(self._sessions),
        }

    def release(self) -> None:
        self._sessions.clear()
        self._closed = True


def create_collaborative_session(agent_count: int, seed: int | None = None) -> CollaborativeCoordinator:
    return CollaborativeCoordinator(agent_count=agent_count, seed=seed)
