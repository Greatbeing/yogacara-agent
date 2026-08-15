"""多智能体协作测试。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_collaborative_session_shares_memory():
    from yogacara_agent import create_collaborative_session
    from yogacara_agent.yogacara_test import Seed

    coordinator = create_collaborative_session(agent_count=2)
    agent0 = coordinator.create_agent("agent-0")
    agent1 = coordinator.create_agent("agent-1")
    assert agent0["env"] is not agent1["env"]
    assert agent0["alaya"] is agent1["alaya"]

    obs = agent0["env"].reset()
    seed = Seed(state_emb=agent0["alaya"]._encode(obs), action="RIGHT", reward=1.0, timestamp=0.0)
    setattr(seed, "source_agent", "agent-0")
    coordinator.alaya.add(seed)

    result = coordinator.run_episode("agent-1", max_steps=3)
    assert result["agent_id"] == "agent-1"
    assert result["seed_count"] >= 1
    assert result["cross_agent_seed_usage"] >= 1


def test_collaboration_summary_and_release():
    from yogacara_agent import create_collaborative_session

    coordinator = create_collaborative_session(agent_count=2)
    coordinator.create_agent("agent-0")
    summary = coordinator.collaboration_summary()
    assert summary["agent_count"] == 2
    coordinator.release()
