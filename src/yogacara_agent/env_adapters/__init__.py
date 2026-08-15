"""环境适配器导出。"""

from .action_mapper import ActionMapper
from .base import AdapterRegistry, BaseSimEnv, EnvAdapter
from .cartpole_adapter import CartPoleAdapter
from .grid_env_adapter import GridEnvAdapter
from .observation_normalizer import ObservationNormalizer
from .isaac_adapter import IsaacEnv
from .ros2_adapter import ROS2Env
from .unity_adapter import UnityEnv

AdapterRegistry.register("grid", GridEnvAdapter)
AdapterRegistry.register("cartpole", CartPoleAdapter)

__all__ = [
    "EnvAdapter",
    "BaseSimEnv",
    "AdapterRegistry",
    "ActionMapper",
    "ObservationNormalizer",
    "GridEnvAdapter",
    "CartPoleAdapter",
    "ROS2Env",
    "UnityEnv",
    "IsaacEnv",
]
