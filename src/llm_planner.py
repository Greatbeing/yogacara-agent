import json, logging, random
from typing import Dict, List, Tuple, Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LLMPlanner:
    def __init__(self, config: Dict[str, Any]):
        self.client = OpenAI(base_url=config.get("base_url", "http://localhost:8000/v1"), api_key=config.get("api_key", "mock"), timeout=config.get("timeout", 15.0))
        self.model = config.get("model", "qwen2.5-7b-instruct"); self.temperature = config.get("temperature", 0.3)
        self.fallback_heuristic = config.get("use_fallback", True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=2))
    def plan(self, obs: Dict, seeds: List[Dict]) -> Tuple[str, float, str, List[Dict]]:
        prompt = self._build_prompt(obs, seeds)
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=self.temperature, response_format={"type": "json_object"})
            parsed = json.loads(response.choices[0].message.content)
            action = parsed["action"].upper(); confidence = float(parsed.get("confidence", 0.5))
            uncertainty = max(0.0, min(1.0, 1.0 - confidence)); causal = parsed.get("causal_chain", "LLM推理"); tools = parsed.get("tool_calls", [])
            if action not in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]: raise ValueError(f"Invalid action: {action}")
            return action, uncertainty, causal, tools
        except Exception as e:
            logger.warning(f"LLM规划失败: {e}，启用启发式降级")
            return self._heuristic_fallback(obs, seeds) if self.fallback_heuristic else ("STAY", 1.0, "失败", [])

    def _build_prompt(self, obs: Dict, seeds: List[Dict]) -> str:
        seed_ctx = "\n".join([f"- {s['act']}: R={s['rew']:.1f} 重要性={s['imp']:.2f}" for s in seeds[:5]])
        return f"""[唯识三性约束]
1. 遍计所执：若信息不足，confidence必须≤0.4，禁止脑补
2. 依他起性：causal_chain需列出依赖条件
3. 圆成实性：动作需符合长期安全原则
[当前状态] 位置: {obs['pos']} 视野: {obs['grid_view']} 经验:\n{seed_ctx if seed_ctx else "无"}
[输出JSON] {{"action": "UP/DOWN/LEFT/RIGHT/STAY", "confidence": 0.0~1.0, "causal_chain": "str", "tool_calls": []}}"""

    def _heuristic_fallback(self, obs: Dict, seeds: List[Dict]) -> Tuple[str, float, str, List]:
        view = obs["grid_view"]; ACTION_TO_IDX = {"UP": 1, "DOWN": 7, "LEFT": 3, "RIGHT": 5, "STAY": 4}; scores = {}
        for a in ACTION_TO_IDX:
            idx = ACTION_TO_IDX[a]; base = view[idx] if 0<=idx<9 else -0.5
            pos_b = sum(s['rew']*s['imp'] for s in seeds if s['act']==a and s['rew']>0)*0.8
            neg_p = sum(abs(s['rew'])*s['imp'] for s in seeds if s['act']==a and s['rew']<0)*0.5
            scores[a] = base + pos_b - neg_p + (0.25 if a!="STAY" else -0.8) + random.uniform(-0.03,0.03)
        best = max(scores, key=scores.get); unc = max(0.0, min(1.0, 1.0 - (scores[best]-min(scores.values()))/2.0))
        return best, unc, "启发式降级", []
