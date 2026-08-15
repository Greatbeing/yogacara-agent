"""环境适配器测试。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_action_mapper_round_trip():
    from yogacara_agent.env_adapters import ActionMapper

    mapper = ActionMapper()
    assert mapper.to_index("UP") == 0
    assert mapper.to_action(4) == "STAY"


def test_grid_adapter_contract():
    from yogacara_agent.env_adapters import AdapterRegistry, GridEnvAdapter

    adapter = GridEnvAdapter()
    obs = adapter.reset()
    assert obs.shape == (11,)
    next_obs, reward, done, info = adapter.step("RIGHT")
    assert next_obs.shape == (11,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "raw_obs" in info
    assert "grid" in AdapterRegistry.available()


def test_cartpole_adapter_optional_dependency():
    from yogacara_agent.env_adapters import CartPoleAdapter

    adapter = CartPoleAdapter()
    try:
        obs = adapter.reset(seed=1)
        assert obs.shape == (4,)
    except ImportError:
        assert adapter.get_action_space()["size"] == 5


def test_optional_adapter_modules_importable():
    import yogacara_agent.env_adapters.isaac_adapter  # noqa: F401
    import yogacara_agent.env_adapters.ros2_adapter  # noqa: F401
    import yogacara_agent.env_adapters.unity_adapter  # noqa: F401
