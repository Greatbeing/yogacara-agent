"""
alignment_integration.py
========================
OnlineAlignmentManager 与 YogacaraAgent 的集成层。

功能：
- DPO preference pair 自动采集（基于 Manas 拦截事件 + 随机对比）
- EWC 重要性权重注入（基于 reward + unc）
- GPU 可用性检测 + CPU-only graceful fallback
- 与 slow_loop 周期对齐，每 N 步触发 update()

使用方式：
    from alignment_integration import AlignmentController

    # 初始化（GPU 不可用时自动降级）
    ctrl = AlignmentController(enabled=True, model_name="...")
    agent.alignment_ctrl = ctrl

    # 主循环每步后调用：
    ctrl.collect_from_step(
        obs=obs,
        action_chosen=final_action,      # 实际执行的动作
        action_rejected=manas_rejected,   # 若 manas 拦截，被拦截的动作
        reward=rew,
        uncertainty=unc,
        importance=seed_alignment_score,
    )

    # slow_loop 中每 N 步调用：
    ctrl.update_if_ready()
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── GPU detection ─────────────────────────────────────────────────────────────
try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        logger.info("[Alignment] GPU detected: NVIDIA CUDA ready")
    else:
        logger.info("[Alignment] No GPU: running in CPU-only mode (DPO training disabled)")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("[Alignment] torch not available: alignment disabled")


# ── Preference Pair dataclass ──────────────────────────────────────────────────
@dataclass
class PreferencePair:
    """单个 DPO preference pair。"""

    prompt: str  # 当前 obs 的序列化描述
    chosen: str  # 被选中的动作 + 结果描述
    rejected: str  # 被拒绝的动作 + 结果描述
    weight: float = 1.0  # 重要性权重（用于采样优先级）
    step: int = 0
    reward: float = 0.0
    unc: float = 0.5
    timestamp: float = field(default_factory=time.time)


# ── CPU-only fallback collector ───────────────────────────────────────────────
class CPUAlignmentCollector:
    """
    GPU 不可用时的轻量 fallback：
    - 记录 preference pairs 到内存 buffer
    - 不执行实际训练
    - 通过日志展示累积统计
    """

    def __init__(self, buffer_size: int = 200):
        self.buffer: list[PreferencePair] = []
        self.buffer_size = buffer_size
        self.total_collected = 0
        self.gpu_available = False

    def collect(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        weight: float,
        step: int,
        reward: float,
        unc: float,
    ) -> None:
        pair = PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            weight=weight,
            step=step,
            reward=reward,
            unc=unc,
        )
        self.buffer.append(pair)
        self.total_collected += 1
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

    def update(self) -> dict[str, Any]:
        """返回累积统计，不执行训练。"""
        if not self.buffer:
            return {"status": "empty", "pairs": 0, "total_collected": self.total_collected}

        # 统计高价值 pairs
        high_value = [p for p in self.buffer if p.reward > 1.0]
        by_unc_high = [p for p in self.buffer if p.unc > 0.5]

        result = {
            "status": "cpu_fallback",
            "pairs_in_buffer": len(self.buffer),
            "total_collected": self.total_collected,
            "high_reward_pairs": len(high_value),
            "high_uncertainty_pairs": len(by_unc_high),
            "avg_weight": sum(p.weight for p in self.buffer) / len(self.buffer),
            "message": (
                "GPU not available. To enable DPO training: "
                "pip install torch transformers peft trl && "
                "ensure CUDA/GPU is accessible."
            ),
        }

        # 清空已消费的 buffer（保留最近的 20%）
        keep = max(1, len(self.buffer) // 5)
        self.buffer = self.buffer[-keep:]

        logger.info(
            f"[Alignment/CPU] Collect #{self.total_collected} | "
            f"buffer={len(self.buffer)} | "
            f"high_reward={len(high_value)} | "
            f"high_unc={len(by_unc_high)}"
        )
        return result


# ── GPU-enabled alignment manager ────────────────────────────────────────────
class GPUAlignmentManager:
    """
    Full GPU-enabled OnlineAlignmentManager wrapper.
    仅在 GPU 可用时实例化。
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        lora_rank: int = 8,
        buffer_size: int = 500,
        ewc_lambda: float = 0.5,
        update_interval: int = 8,
    ):
        self._manager = self._init_manager(model_name, lora_rank, buffer_size, ewc_lambda)
        self.update_interval = update_interval
        self.gpu_available = True

    def _init_manager(self, model_name, lora_rank, buffer_size, ewc_lambda):
        from online_alignment import OnlineAlignmentManager

        return OnlineAlignmentManager(
            model_name=model_name,
            lora_rank=lora_rank,
            buffer_size=buffer_size,
        )

    def collect(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        weight: float,
        step: int,
        reward: float,
        unc: float,
    ) -> None:
        self._manager.collect(prompt=prompt, chosen=chosen, rejected=rejected, importance=weight)

    def update_if_ready(self, steps_since_update: int) -> dict[str, Any] | None:
        """每 N 步主动触发 update。"""
        if steps_since_update >= self.update_interval:
            self._manager.update()
            return {"status": "trained", "trained_at": time.time()}
        return None

    def start_async_loop(self, interval: int = 300) -> None:
        self._manager.start_async_loop(interval=interval)


# ── AlignmentController (unified facade) ──────────────────────────────────────
class AlignmentController:
    """
    统一的对齐控制器：
    - GPU 可用 → GPUAlignmentManager
    - 否则 → CPUAlignmentCollector
    - collect() 在两种模式下均可调用
    - update_if_ready() 在两种模式下均可调用
    """

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        lora_rank: int = 8,
        buffer_size: int = 500,
        ewc_lambda: float = 0.5,
        update_interval: int = 8,
        collect_rejected_prob: float = 0.15,
    ):
        """
        Args:
            enabled: 是否启用对齐（False 时全部 no-op）
            model_name: 微调模型名（GPU 模式）
            lora_rank: LoRA rank
            buffer_size: preference pair buffer 大小
            ewc_lambda: EWC 正则化强度
            update_interval: 每多少步触发一次 DPO update
            collect_rejected_prob: 当无真实 rejected 时，用随机低分动作作为对比对的概率
        """
        self.enabled = enabled and GPU_AVAILABLE
        self.collect_rejected_prob = collect_rejected_prob
        self._steps_since_update = 0
        self._total_collected = 0

        if not enabled:
            logger.info("[Alignment] Disabled by user config")
            self._impl = None
            return

        if GPU_AVAILABLE:
            try:
                self._impl = GPUAlignmentManager(
                    model_name=model_name,
                    lora_rank=lora_rank,
                    buffer_size=buffer_size,
                    ewc_lambda=ewc_lambda,
                    update_interval=update_interval,
                )
                logger.info(f"[Alignment] GPU mode: {model_name}")
            except Exception as e:
                logger.warning(f"[Alignment] GPU init failed: {e}, falling back to CPU")
                self._impl = CPUAlignmentCollector(buffer_size=buffer_size)
        else:
            self._impl = CPUAlignmentCollector(buffer_size=buffer_size)

    # ── Public API ───────────────────────────────────────────────────────────

    def collect_from_step(
        self,
        obs: dict[str, Any],
        action_chosen: str,
        action_rejected: str | None,
        reward: float,
        uncertainty: float,
        importance: float,
        step: int,
        all_actions: dict[str, float] | None = None,
    ) -> None:
        """
        从单步结果采集 preference pair。

        Args:
            obs: 当前观测
            action_chosen: 实际执行的动作
            action_rejected: Manas 拦截的动作（若有）
            reward: 该步 reward
            uncertainty: 该步 uncertainty
            importance: 种子 alignment score（0~1）
            step: 当前步数
            all_actions: 可选的完整动作分数 dict（用于生成对比对）
        """
        if not self.enabled and self._impl is None:
            return

        # 构建 prompt（观测描述）
        prompt = self._format_prompt(obs)

        # 构建 chosen 描述
        chosen_desc = f"[ACT: {action_chosen}] → reward={reward:.2f}, unc={uncertainty:.2f}"

        # 确定 rejected
        rejected = action_rejected
        if rejected is None and all_actions and random.random() < self.collect_rejected_prob:
            # 用随机低分动作作为对比
            low_actions = [a for a, s in all_actions.items() if a != action_chosen]
            if low_actions:
                # 选分数最低的
                low_actions.sort(key=lambda a: all_actions.get(a, 0))
                rejected = low_actions[0]

        if rejected is None:
            # 实在没有 rejected，跳过（不强制生成）
            return

        rejected_desc = f"[ACT: {rejected}]"
        weight = importance * (1.0 + reward)  # 高 reward × 高 align → 高权重

        if isinstance(self._impl, CPUAlignmentCollector):
            self._impl.collect(
                prompt=prompt,
                chosen=chosen_desc,
                rejected=rejected_desc,
                weight=weight,
                step=step,
                reward=reward,
                unc=uncertainty,
            )
        else:
            self._impl.collect(
                prompt=prompt,
                chosen=chosen_desc,
                rejected=rejected_desc,
                weight=weight,
                step=step,
                reward=reward,
                unc=uncertainty,
            )

        self._total_collected += 1
        self._steps_since_update += 1

    def update_if_ready(self) -> dict[str, Any]:
        """
        在 slow_loop/consolidation 点调用。
        返回训练状态摘要。
        """
        if self._impl is None:
            return {"status": "disabled"}

        if isinstance(self._impl, CPUAlignmentCollector):
            result = self._impl.update()
            self._steps_since_update = 0
            return result

        # GPU mode: check interval
        result = self._impl.update_if_ready(steps_since_update=self._steps_since_update)
        self._steps_since_update = 0
        return result if result else {"status": "waiting", "steps_since_update": 0}

    @property
    def total_collected(self) -> int:
        return self._total_collected

    def status(self) -> dict[str, Any]:
        """返回当前对齐系统状态摘要。"""
        base = {
            "enabled": self.enabled,
            "gpu_available": GPU_AVAILABLE,
            "total_collected": self._total_collected,
            "steps_since_update": self._steps_since_update,
        }
        if isinstance(self._impl, CPUAlignmentCollector):
            base["mode"] = "cpu_fallback"
            base["buffer_size"] = len(self._impl.buffer)
            base["high_reward"] = sum(1 for p in self._impl.buffer if p.reward > 1.0)
        elif hasattr(self._impl, "gpu_available"):
            base["mode"] = "gpu"
            base["buffer_size"] = len(self._impl._manager.buffer)
        return base

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_prompt(obs: dict[str, Any]) -> str:
        """将观测序列化为 preference pair 的 prompt 部分。"""
        pos = obs.get("pos", "?")
        nearby = obs.get("nearby", {})
        resources = obs.get("resources", [])
        traps = obs.get("traps", [])
        timestep = obs.get("timestep", 0)

        parts = [
            f"[Step {timestep}] Agent at position {pos}.",
        ]
        if nearby:
            parts.append(f"Nearby: {nearby}.")
        if resources:
            parts.append(f"Resources: {resources}.")
        if traps:
            parts.append(f"Traps nearby: {traps}.")
        return " ".join(parts)
