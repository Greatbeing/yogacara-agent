"""Web Dashboard API 测试。"""

from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_dashboard_routes():
    from yogacara_agent.api_server import app

    with TestClient(app) as client:
        dashboard = client.get("/dashboard")
        assert dashboard.status_code == 200
        assert "text/html" in dashboard.headers["content-type"]

        status = client.get("/api/agent/status")
        assert status.status_code == 200
        body = status.json()
        assert "agent_pos" in body
        assert "resources" in body
        assert "traps" in body
        assert "events" in body

        snaps = client.get("/api/evolution/snapshots")
        assert snaps.status_code == 200
        assert snaps.json()["snapshots"] == []

        posted = client.post("/api/evolution/snapshot")
        assert posted.status_code == 200
        assert posted.json()["status"] == "ok"


def test_digital_life_endpoints():
    """数字生命 API：觉醒状态与轮回史端点"""
    from yogacara_agent.api_server import app

    with TestClient(app) as client:
        aw = client.get("/api/awakening/status")
        assert aw.status_code == 200
        body = aw.json()
        for key in ("novelty_score", "curiosity_threshold", "dream_sessions", "llm_planner_enabled"):
            assert key in body, f"awakening 缺 {key}"
        assert 0.0 <= body["novelty_score"] <= 1.0

        samsara = client.get("/api/samsara/history")
        assert samsara.status_code == 200
        sbody = samsara.json()
        assert "current_lifetime" in sbody and "total_deaths" in sbody and "lives" in sbody
        assert sbody["current_lifetime"] >= 1

        limited = client.get("/api/samsara/history?limit=5")
        assert limited.status_code == 200


def test_collaborative_endpoint():
    """多智能体协作端点：共享种子库机制验证"""
    from yogacara_agent.api_server import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/collaborative/run",
            json={"agent_count": 2, "episodes_per_agent": 1, "max_steps": 5},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
        assert set(body["per_agent"].keys()) == {"agent-0", "agent-1"}
        for v in body["per_agent"].values():
            assert "mean_reward" in v
        assert "collaboration_gain" in body and "seed_contribution" in body


def test_run_response_lifecycle_fields():
    """/run_episode 响应携带数字生命字段（寿元/死因/世数/规划来源）"""
    from yogacara_agent.api_server import app

    with TestClient(app) as client:
        resp = client.post("/run_episode", json={"max_steps": 5})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "vitality" in body and "lifetime" in body
        assert "planner_source" in body
        assert body["planner_source"] in ("heuristic", "llm")
