import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))
from yogacara_langgraph import build_graph, create_session, slow_loop

_app_session = None
loop_started = False


def _get_session():
    global _app_session
    if _app_session is None:
        _app_session = create_session()
    return _app_session


class AgentRequest(BaseModel):
    max_steps: int = 60
    custom_obs: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: start background tasks on startup."""
    global loop_started
    if not loop_started:
        session = _get_session()
        asyncio.create_task(slow_loop(session["alaya"], interval=10))
        loop_started = True
    yield


app = FastAPI(title="唯识进化框架 API", version="1.2.0", lifespan=lifespan)
graph = build_graph()


@app.post("/run_episode")
async def run_episode(req: AgentRequest, request: Request):
    session = _get_session()
    env = session["env"]
    manas = session["manas"]
    try:
        env.reset()
        if req.custom_obs:
            env.agent_pos = list(req.custom_obs.get("pos", [0, 0]))
        init_state = {
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
        final_state = await graph.ainvoke(init_state)
        return {
            "status": "success",
            "steps": final_state["step"],
            "cumulative_reward": sum(final_state["recent_rewards"]),
            "manas_reflections": manas.reflections,
            "resources_found": sum(1 for r in final_state["recent_rewards"] if r > 2.0),
            "final_pos": final_state["obs"]["pos"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    session = _get_session()
    alaya = session["alaya"]
    manas = session["manas"]
    return {"status": "ok", "memory_seeds": len(alaya.seeds), "manas_reflections": manas.reflections}


async def main():
    uvicorn.run("yogacara_agent.api_server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
