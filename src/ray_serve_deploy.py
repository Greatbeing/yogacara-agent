import ray, asyncio, os
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

ray.init(address="auto", namespace="yogacara")
app = FastAPI()

class EpisodeRequest(BaseModel):
    max_steps: int = 60; custom_obs: Dict[str, Any] = None

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 2, "num_gpus": 0.5}, autoscaling_config={"min_replicas": 2, "max_replicas": 8, "target_num_ongoing_requests_per_replica": 10})
@serve.ingress(app)
class YogacaraServe:
    def __init__(self):
        from yogacara_langgraph import build_graph, env, alaya, manas
        self.graph = build_graph(); self.env = env; self.alaya = alaya; self.manas = manas; self.lock = asyncio.Lock()

    @app.post("/run_episode")
    async def run_episode(self, req: EpisodeRequest):
        async with self.lock:
            self.env.reset()
            if req.custom_obs: self.env.agent_pos = list(req.custom_obs.get("pos", [0,0]))
            init_state = {"obs": self.env._observe(), "action": "", "reward": 0.0, "done": False, "step": 0, "seeds": [], "unc": 0.0, "manas_passed": True, "tool_calls": [], "recent_rewards": [], "pos_history": [], "metrics": {}}
            final_state = await self.graph.ainvoke(init_state)
            return {"steps": final_state["step"], "cumulative_reward": sum(final_state["recent_rewards"]), "manas_reflections": self.manas.reflections, "final_pos": final_state["obs"]["pos"]}

# 启动: serve run src.ray_serve_deploy:YogacaraServe
