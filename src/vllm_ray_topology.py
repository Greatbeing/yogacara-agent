import logging
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Lazy Ray init — only connect when first needed, not at import time
_ray_initialized = False


def _ensure_ray():
    global _ray_initialized
    if not _ray_initialized:
        try:
            import ray

            ray.init(address="auto", namespace="yogacara", ignore_reinit_error=True)
            _ray_initialized = True
        except Exception as e:
            logger.warning(f"Ray init failed (will use fallback): {e}")
            _ray_initialized = True  # Don't retry


app = FastAPI()


class LLMRequest(BaseModel):
    prompts: list[str]
    temperature: float = 0.3
    max_tokens: int = 128


def _build_deployments():
    """Build Ray Serve deployments. Called lazily to avoid import-time Ray dependency."""
    from ray import serve
    from vllm import LLM, SamplingParams

    _ensure_ray()

    @serve.deployment(
        num_replicas=1,
        ray_actor_options={"num_gpus": 1},
        autoscaling_config={"min_replicas": 1, "max_replicas": 3},
    )
    class VLLMDeployment:
        def __init__(self, model_name: str, tensor_parallel_size: int = 1):
            self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.85)
            self.sampling_params = SamplingParams(temperature=0.3, max_tokens=128)

        async def __call__(self, req: LLMRequest) -> list[dict[str, Any]]:
            self.sampling_params.temperature = req.temperature
            self.sampling_params.max_tokens = req.max_tokens
            outputs = self.llm.generate(req.prompts, self.sampling_params, use_tqdm=False)
            return [{"text": o.outputs[0].text, "token_count": len(o.outputs[0].token_ids)} for o in outputs]

    @serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 2})
    @serve.ingress(app)
    class LLMRouter:
        def __init__(self, vllm_handle):
            self.vllm = vllm_handle
            self.cb_failures = 0
            self.cb_threshold = 5

        @app.post("/v1/chat/completions")
        async def chat(self, req: dict[str, Any]):
            if self.cb_failures >= self.cb_threshold:
                return {"error": "circuit_breaker_open", "fallback": "heuristic"}
            try:
                prompts = [m["content"] for m in req.get("messages", [])]
                llm_req = LLMRequest(prompts=prompts, temperature=req.get("temperature", 0.3))
                res = await self.vllm.remote(llm_req)
                self.cb_failures = 0
                return {"choices": [{"message": {"content": res[0]["text"]}}]}
            except Exception as e:
                self.cb_failures += 1
                logger.error(f"vLLM路由异常: {e}")
                return {"error": str(e), "fallback": "heuristic"}

    vllm_dep = VLLMDeployment.bind(model_name="Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=1)  # type: ignore[attr-defined]
    router_dep = LLMRouter.bind(vllm_dep)  # type: ignore[attr-defined]
    return router_dep


# Lazy access: router_dep is built on first attribute access
class _LazyRouter:
    """Lazy proxy that builds the deployment graph on first access."""

    def __init__(self):
        self._router = None

    def _build(self):
        if self._router is None:
            self._router = _build_deployments()
        return self._router

    def __getattr__(self, name):
        return getattr(self._build(), name)


router_dep = _LazyRouter()
# 启动: serve run src.vllm_ray_topology:router_dep
