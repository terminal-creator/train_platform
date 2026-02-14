"""
实时指标生成器 - 模拟训练过程中的指标变化

用于WebSocket推送的实时指标生成
"""
import math
import random
import time
from datetime import datetime
from typing import Dict, Optional, Generator
from dataclasses import dataclass, field


@dataclass
class TrainingState:
    """训练状态"""
    job_id: str
    algorithm: str
    current_step: int
    total_steps: int
    start_time: float = field(default_factory=time.time)

    # 指标状态
    reward_mean: float = 0.2
    policy_loss: float = 0.25
    value_loss: float = 0.15
    kl_divergence: float = 0.001
    train_loss: float = 2.5

    def progress(self) -> float:
        return self.current_step / self.total_steps if self.total_steps > 0 else 0


class MetricsGenerator:
    """
    指标生成器

    根据训练进度生成逼真的训练指标
    """

    def __init__(self, speed: float = 1.0):
        """
        Args:
            speed: 模拟速度倍率 (1.0=正常, 2.0=2倍速)
        """
        self.speed = speed
        self._states: Dict[str, TrainingState] = {}

    def init_training(
        self,
        job_id: str,
        algorithm: str,
        total_steps: int,
        current_step: int = 0,
    ) -> TrainingState:
        """初始化训练状态"""
        state = TrainingState(
            job_id=job_id,
            algorithm=algorithm,
            current_step=current_step,
            total_steps=total_steps,
        )

        # 根据算法设置初始值
        if algorithm == "sft":
            state.train_loss = 2.8
        elif algorithm in ["grpo", "ppo"]:
            state.reward_mean = 0.15
            state.policy_loss = 0.25
            state.value_loss = 0.15
            state.kl_divergence = 0.001
        elif algorithm == "dpo":
            state.train_loss = 0.69  # log(2)

        self._states[job_id] = state
        return state

    def get_state(self, job_id: str) -> Optional[TrainingState]:
        """获取训练状态"""
        return self._states.get(job_id)

    def step(self, job_id: str, num_steps: int = 1) -> Optional[Dict]:
        """
        推进训练步数并生成指标

        Returns:
            当前步的指标字典
        """
        state = self._states.get(job_id)
        if not state:
            return None

        state.current_step = min(state.current_step + num_steps, state.total_steps)
        progress = state.progress()

        metrics = self._generate_metrics(state, progress)
        return metrics

    def _generate_metrics(self, state: TrainingState, progress: float) -> Dict:
        """根据算法类型生成指标"""
        base_metrics = {
            "step": state.current_step,
            "epoch": int(progress * 3) + 1,  # 假设3个epoch
            "progress": round(progress * 100, 1),
            "timestamp": datetime.now().isoformat(),
            "throughput_tokens_per_sec": int(11000 + random.gauss(0, 500)),
            "gpu_memory_used_gb": round(72 + random.gauss(0, 1), 1),
        }

        if state.algorithm == "sft":
            return self._generate_sft_metrics(state, progress, base_metrics)
        elif state.algorithm in ["grpo", "ppo"]:
            return self._generate_grpo_metrics(state, progress, base_metrics)
        elif state.algorithm == "dpo":
            return self._generate_dpo_metrics(state, progress, base_metrics)

        return base_metrics

    def _generate_sft_metrics(
        self, state: TrainingState, progress: float, base: Dict
    ) -> Dict:
        """生成SFT指标"""
        # Loss曲线：指数衰减
        decay = math.exp(-3 * progress)
        target_loss = 0.42

        state.train_loss = target_loss + (2.8 - target_loss) * decay
        state.train_loss += random.gauss(0, 0.02)

        eval_loss = state.train_loss * 1.1 + random.gauss(0, 0.02)
        perplexity = math.exp(state.train_loss)

        # Learning rate with warmup and cosine decay
        warmup = 0.03
        if progress < warmup:
            lr = 1e-5 * (progress / warmup)
        else:
            lr = 1e-5 * 0.5 * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))

        base.update({
            "train_loss": round(max(state.train_loss, 0.35), 4),
            "eval_loss": round(max(eval_loss, 0.4), 4),
            "perplexity": round(max(perplexity, 1.4), 4),
            "learning_rate": round(lr, 9),
            "grad_norm": round(0.8 + 0.3 * decay + random.gauss(0, 0.05), 4),
        })

        return base

    def _generate_grpo_metrics(
        self, state: TrainingState, progress: float, base: Dict
    ) -> Dict:
        """生成GRPO/PPO指标"""
        # Reward: S型增长
        sigmoid = 1 / (1 + math.exp(-8 * (progress - 0.4)))
        target_reward = 0.92

        state.reward_mean = 0.15 + (target_reward - 0.15) * sigmoid
        state.reward_mean += random.gauss(0, 0.015)

        # Policy loss: 逐渐下降
        state.policy_loss = 0.25 * (1 - 0.65 * progress)
        state.policy_loss += random.gauss(0, 0.008)

        # Value loss: 逐渐下降
        state.value_loss = 0.15 * (1 - 0.55 * progress)
        state.value_loss += random.gauss(0, 0.005)

        # KL: 逐渐增加但保持可控
        state.kl_divergence = 0.025 * progress * (1 + 0.2 * math.sin(5 * math.pi * progress))
        state.kl_divergence = max(state.kl_divergence, 0.001)

        # 其他指标
        entropy = 2.1 - 0.25 * progress + random.gauss(0, 0.03)
        clip_fraction = 0.2 * (1 - 0.25 * progress) + random.gauss(0, 0.015)
        advantages = 0.05 + 0.12 * sigmoid + random.gauss(0, 0.015)

        base.update({
            "policy_loss": round(max(state.policy_loss, 0.05), 4),
            "value_loss": round(max(state.value_loss, 0.02), 4),
            "total_loss": round(max(state.policy_loss + 0.5 * state.value_loss, 0.06), 4),
            "reward_mean": round(min(state.reward_mean, 0.95), 4),
            "reward_std": round(0.25 * (1 - 0.4 * progress) + random.gauss(0, 0.01), 4),
            "kl_divergence": round(state.kl_divergence, 5),
            "entropy": round(max(entropy, 1.6), 4),
            "clip_fraction": round(min(max(clip_fraction, 0.05), 0.3), 4),
            "advantages_mean": round(advantages, 4),
            "response_length_mean": round(320 + 70 * progress + random.gauss(0, 15), 1),
            "learning_rate": round(5e-7 * (1 - 0.4 * progress), 9),
        })

        return base

    def _generate_dpo_metrics(
        self, state: TrainingState, progress: float, base: Dict
    ) -> Dict:
        """生成DPO指标"""
        # Loss: 从log(2)下降
        state.train_loss = 0.69 - 0.38 * progress + random.gauss(0, 0.015)

        # Rewards
        rewards_chosen = 1.5 + 1.8 * progress + random.gauss(0, 0.08)
        rewards_rejected = -0.5 - 1.2 * progress + random.gauss(0, 0.08)

        # Accuracy
        accuracy = 0.55 + 0.38 * progress + random.gauss(0, 0.015)

        base.update({
            "loss": round(max(state.train_loss, 0.2), 4),
            "rewards_chosen": round(rewards_chosen, 4),
            "rewards_rejected": round(rewards_rejected, 4),
            "reward_margin": round(rewards_chosen - rewards_rejected, 4),
            "accuracy": round(min(accuracy, 0.96), 4),
            "learning_rate": round(5e-7 * (1 - 0.25 * progress), 9),
        })

        return base

    def generate_stream(
        self,
        job_id: str,
        interval_seconds: float = 2.0,
    ) -> Generator[Dict, None, None]:
        """
        生成指标流（用于WebSocket推送）

        Yields:
            每隔interval_seconds生成一个指标字典
        """
        state = self._states.get(job_id)
        if not state:
            return

        actual_interval = interval_seconds / self.speed

        while state.current_step < state.total_steps:
            metrics = self.step(job_id, num_steps=25)  # 每次推进25步
            if metrics:
                yield metrics
            time.sleep(actual_interval)

        # 最后一次
        yield self.step(job_id, num_steps=0)


# 全局生成器实例
_generator: Optional[MetricsGenerator] = None


def get_metrics_generator(speed: float = 1.0) -> MetricsGenerator:
    """获取全局指标生成器"""
    global _generator
    if _generator is None or _generator.speed != speed:
        _generator = MetricsGenerator(speed=speed)
    return _generator
