"""Yogacara-Agent: 基于唯识理论的进化型AI框架。

公共API入口。核心模块位于 yogacara_test 子模块中。
"""

from yogacara_agent.yogacara_test import (
    # 核心类
    Seed,
    GridSimEnv,
    AlayaMemory,
    ManasController,
    ConsciousnessPlanner,
    YogacaraAgent,
    # 常量
    GRID_SIZE,
    MEMORY_CAPACITY,
    CONSOLIDATION_INTERVAL,
    DECAY_RATE,
    ACTIONS,
    ACTION_TO_IDX,
)

__all__ = [
    # 核心类
    "Seed",
    "GridSimEnv",
    "AlayaMemory",
    "ManasController",
    "ConsciousnessPlanner",
    "YogacaraAgent",
    # 常量
    "GRID_SIZE",
    "MEMORY_CAPACITY",
    "CONSOLIDATION_INTERVAL",
    "DECAY_RATE",
    "ACTIONS",
    "ACTION_TO_IDX",
]
