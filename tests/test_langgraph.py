"""
Yogacara Agent LangGraph 流水线测试
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestBuildGraph:
    """测试 LangGraph 图结构"""

    def test_graph_structure(self):
        """测试 build_graph 返回正确的节点结构"""
        from yogacara_agent.yogacara_langgraph import build_graph

        graph = build_graph()
        # 图应有 7 个节点：perceive, plan, introspect, manas, execute, store, consolidate
        node_names = {n for n, _ in graph.nodes.items()}
        expected = {"__start__", "perceive", "plan", "introspect", "manas", "execute", "store", "consolidate"}
        assert node_names == expected, f"Nodes: {node_names} != {expected}"

    def test_graph_compiles(self):
        """测试图编译成功"""
        from yogacara_agent.yogacara_langgraph import build_graph, YogacaraState

        graph = build_graph()
        # 验证图可以编译并接受有效输入
        state_schema = graph.config_schema
        assert state_schema is not None, "Graph should have a config schema"


class TestGraphNodes:
    """测试 LangGraph 各节点"""

    def test_node_perceive_retrieves_seeds(self):
        """测试 node_perceive 检索种子"""
        from yogacara_agent.yogacara_langgraph import (
            AlayaMemory,
            GridSimEnv,
            env,
            alaya,
            build_graph,
            node_perceive,
        )

        # 先添加一些种子到 alaya
        obs = {"pos": (0, 0), "grid_view": [0.0] * 9, "step": 0}
        emb = alaya.encode(obs)
        alaya.add({"emb": emb, "act": "RIGHT", "rew": 5.0, "seed_type": "业种"})

        state = {
            "obs": obs,
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
            "introspection_record": None,
            "ego_alert": None,
            "plan_scores": None,
            "reasoning": "",
            "steps_since_resource": 0,
            "steps_at_same_pos": 0,
            "step_limit": 60,
        }
        import asyncio
        result = asyncio.run(node_perceive(state))
        assert len(result["seeds"]) > 0, "Should retrieve seeds"
        assert result["seeds"][0]["act"] == "RIGHT"

    def test_node_execute_stuck_detection(self):
        """测试 node_execute 的 stuck 检测"""
        from yogacara_agent.yogacara_langgraph import node_execute, env

        env.reset()
        # 用 STAY 动作测试 stuck 递增
        state = {
            "obs": env.observe(),
            "action": "STAY",
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
            "introspection_record": None,
            "ego_alert": None,
            "plan_scores": None,
            "reasoning": "",
            "steps_since_resource": 0,
            "steps_at_same_pos": 0,
            "step_limit": 60,
        }
        import asyncio

        result = asyncio.run(node_execute(state))
        # STAY → 位置不变 → steps_at_same_pos 应递增
        assert result["steps_at_same_pos"] == 1, f"Expected 1, got {result['steps_at_same_pos']}"

        # 第二次 STAY → 应该再递增
        result2 = asyncio.run(node_execute(result))
        assert result2["steps_at_same_pos"] == 2, f"Expected 2, got {result2['steps_at_same_pos']}"

    def test_node_execute_step_limit(self):
        """测试 step_limit 终止"""
        from yogacara_agent.yogacara_langgraph import node_execute, env

        env.reset()
        state = {
            "obs": env.observe(),
            "action": "RIGHT",
            "reward": 0.0,
            "done": False,
            "step": 59,  # 接近上限
            "seeds": [],
            "unc": 0.0,
            "manas_passed": True,
            "tool_calls": [],
            "recent_rewards": [],
            "pos_history": [],
            "metrics": {},
            "introspection_record": None,
            "ego_alert": None,
            "plan_scores": None,
            "reasoning": "",
            "steps_since_resource": 0,
            "steps_at_same_pos": 0,
            "step_limit": 60,
        }
        import asyncio

        result = asyncio.run(node_execute(state))
        assert result["step"] == 60
        assert result["done"] is True, "Should be done when step >= step_limit"


class TestGraphExecution:
    """测试完整图执行"""

    def test_full_graph_short_episode(self):
        """测试完整图执行 5 步"""
        from yogacara_agent.yogacara_langgraph import build_graph, env, alaya

        graph = build_graph()
        init_state = {
            "obs": env.reset(),
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
            "introspection_record": None,
            "ego_alert": None,
            "plan_scores": None,
            "reasoning": "",
            "steps_since_resource": 0,
            "steps_at_same_pos": 0,
            "step_limit": 5,
        }
        import asyncio

        final = asyncio.run(graph.ainvoke(init_state))
        assert final["step"] == 5, f"Expected 5 steps, got {final['step']}"
        assert final["done"] is True, "Should be done after step_limit"
        assert len(final["recent_rewards"]) == 5, f"Expected 5 rewards, got {len(final['recent_rewards'])}"
        assert "action" in final, "Final state should have action"
        assert "plan_scores" in final, "Final state should have plan_scores"