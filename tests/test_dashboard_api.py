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
