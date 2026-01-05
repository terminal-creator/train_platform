"""
Checkpoint Selector for Model Surgery
Intelligently selects best checkpoints based on various criteria
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import os
import json
import re


class SelectionCriteria(Enum):
    HIGHEST_REWARD = "highest_reward"
    HIGHEST_BENCHMARK = "highest_benchmark"
    LOWEST_KL = "lowest_kl"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class CheckpointMetrics:
    """Metrics for a single checkpoint"""
    step: int
    path: str
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    kl_divergence: Optional[float] = None
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy: Optional[float] = None

    # Benchmark results
    gsm8k_accuracy: Optional[float] = None
    math_accuracy: Optional[float] = None
    humaneval_pass_rate: Optional[float] = None
    mmlu_accuracy: Optional[float] = None

    # Additional metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "step": self.step,
            "path": self.path,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "kl_divergence": self.kl_divergence,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "benchmarks": {
                "gsm8k": self.gsm8k_accuracy,
                "math": self.math_accuracy,
                "humaneval": self.humaneval_pass_rate,
                "mmlu": self.mmlu_accuracy,
            },
        }
        result.update(self.custom_metrics)
        return result

    def get_benchmark_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted benchmark score"""
        if weights is None:
            weights = {"gsm8k": 0.4, "math": 0.4, "humaneval": 0.1, "mmlu": 0.1}

        score = 0.0
        total_weight = 0.0

        benchmark_values = {
            "gsm8k": self.gsm8k_accuracy,
            "math": self.math_accuracy,
            "humaneval": self.humaneval_pass_rate,
            "mmlu": self.mmlu_accuracy,
        }

        for name, weight in weights.items():
            value = benchmark_values.get(name)
            if value is not None:
                score += value * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0


@dataclass
class CheckpointRecommendation:
    """Recommendation result for checkpoint selection"""
    recommended_checkpoint: CheckpointMetrics
    all_checkpoints: List[CheckpointMetrics]
    criteria_used: str
    score: float
    reasoning: str
    alternatives: List[CheckpointMetrics]


class CheckpointSelector:
    """
    Intelligently selects best checkpoints based on various criteria
    """

    def __init__(self):
        self.checkpoints: List[CheckpointMetrics] = []

    def add_checkpoint(self, checkpoint: CheckpointMetrics):
        """Add a checkpoint to the selector"""
        self.checkpoints.append(checkpoint)

    def load_from_experiment(
        self,
        experiment_path: str,
        metrics_file: str = "metrics.json",
    ) -> int:
        """
        Load checkpoints from an experiment directory
        Returns number of checkpoints loaded
        """
        experiment_path = Path(experiment_path)
        if not experiment_path.exists():
            return 0

        # Find checkpoint directories
        checkpoint_dirs = sorted(
            [d for d in experiment_path.iterdir() if d.is_dir() and "ckpt" in d.name.lower()],
            key=lambda x: self._extract_step(x.name)
        )

        # Load metrics file if exists
        metrics_path = experiment_path / metrics_file
        metrics_data = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics_data = json.load(f)

        loaded = 0
        for ckpt_dir in checkpoint_dirs:
            step = self._extract_step(ckpt_dir.name)
            if step is None:
                continue

            # Get metrics for this step
            step_metrics = metrics_data.get(str(step), {})

            checkpoint = CheckpointMetrics(
                step=step,
                path=str(ckpt_dir),
                reward_mean=step_metrics.get("reward_mean"),
                reward_std=step_metrics.get("reward_std"),
                kl_divergence=step_metrics.get("kl_divergence"),
                policy_loss=step_metrics.get("policy_loss"),
                value_loss=step_metrics.get("value_loss"),
                entropy=step_metrics.get("entropy"),
                gsm8k_accuracy=step_metrics.get("gsm8k_accuracy"),
                math_accuracy=step_metrics.get("math_accuracy"),
                humaneval_pass_rate=step_metrics.get("humaneval_pass_rate"),
                mmlu_accuracy=step_metrics.get("mmlu_accuracy"),
            )
            self.add_checkpoint(checkpoint)
            loaded += 1

        return loaded

    def select_best(
        self,
        criteria: SelectionCriteria = SelectionCriteria.BALANCED,
        custom_formula: Optional[str] = None,
        benchmark_weights: Optional[Dict[str, float]] = None,
    ) -> CheckpointRecommendation:
        """
        Select best checkpoint based on criteria
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints loaded")

        if criteria == SelectionCriteria.HIGHEST_REWARD:
            return self._select_by_reward()
        elif criteria == SelectionCriteria.HIGHEST_BENCHMARK:
            return self._select_by_benchmark(benchmark_weights)
        elif criteria == SelectionCriteria.LOWEST_KL:
            return self._select_by_kl()
        elif criteria == SelectionCriteria.BALANCED:
            return self._select_balanced(benchmark_weights)
        elif criteria == SelectionCriteria.CUSTOM:
            if custom_formula is None:
                raise ValueError("Custom formula required for CUSTOM criteria")
            return self._select_by_custom_formula(custom_formula)
        else:
            return self._select_balanced(benchmark_weights)

    def _select_by_reward(self) -> CheckpointRecommendation:
        """Select checkpoint with highest reward"""
        valid_checkpoints = [c for c in self.checkpoints if c.reward_mean is not None]
        if not valid_checkpoints:
            raise ValueError("No checkpoints with reward data")

        sorted_checkpoints = sorted(
            valid_checkpoints,
            key=lambda x: x.reward_mean,
            reverse=True
        )

        best = sorted_checkpoints[0]
        return CheckpointRecommendation(
            recommended_checkpoint=best,
            all_checkpoints=self.checkpoints,
            criteria_used="highest_reward",
            score=best.reward_mean,
            reasoning=f"Selected checkpoint at step {best.step} with highest reward ({best.reward_mean:.4f})",
            alternatives=sorted_checkpoints[1:4],
        )

    def _select_by_benchmark(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> CheckpointRecommendation:
        """Select checkpoint with highest benchmark score"""
        scores = [(c, c.get_benchmark_score(weights)) for c in self.checkpoints]
        scores = [(c, s) for c, s in scores if s > 0]

        if not scores:
            raise ValueError("No checkpoints with benchmark data")

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        best = sorted_scores[0][0]
        best_score = sorted_scores[0][1]

        return CheckpointRecommendation(
            recommended_checkpoint=best,
            all_checkpoints=self.checkpoints,
            criteria_used="highest_benchmark",
            score=best_score,
            reasoning=f"Selected checkpoint at step {best.step} with highest benchmark score ({best_score:.2f}%)",
            alternatives=[c for c, _ in sorted_scores[1:4]],
        )

    def _select_by_kl(self) -> CheckpointRecommendation:
        """Select checkpoint with lowest KL divergence"""
        valid_checkpoints = [c for c in self.checkpoints if c.kl_divergence is not None]
        if not valid_checkpoints:
            raise ValueError("No checkpoints with KL divergence data")

        sorted_checkpoints = sorted(
            valid_checkpoints,
            key=lambda x: x.kl_divergence
        )

        best = sorted_checkpoints[0]
        return CheckpointRecommendation(
            recommended_checkpoint=best,
            all_checkpoints=self.checkpoints,
            criteria_used="lowest_kl",
            score=best.kl_divergence,
            reasoning=f"Selected checkpoint at step {best.step} with lowest KL divergence ({best.kl_divergence:.4f})",
            alternatives=sorted_checkpoints[1:4],
        )

    def _select_balanced(
        self,
        benchmark_weights: Optional[Dict[str, float]] = None,
    ) -> CheckpointRecommendation:
        """
        Select checkpoint with balanced score considering:
        - Benchmark performance
        - Reward
        - KL divergence (penalty for too high)
        """
        scores = []

        for checkpoint in self.checkpoints:
            score = 0.0
            components = []

            # Benchmark score (weight: 0.5)
            bench_score = checkpoint.get_benchmark_score(benchmark_weights)
            if bench_score > 0:
                score += bench_score * 0.5
                components.append(f"bench={bench_score:.1f}")

            # Reward score (weight: 0.3)
            if checkpoint.reward_mean is not None:
                # Normalize reward to 0-100 scale (assuming reward in -1 to 1)
                norm_reward = (checkpoint.reward_mean + 1) * 50
                score += norm_reward * 0.3
                components.append(f"reward={checkpoint.reward_mean:.2f}")

            # KL penalty (weight: 0.2, inverted)
            if checkpoint.kl_divergence is not None:
                # Lower KL is better, penalize high KL
                kl_penalty = max(0, 20 - checkpoint.kl_divergence * 100)
                score += kl_penalty
                components.append(f"kl={checkpoint.kl_divergence:.3f}")

            scores.append((checkpoint, score, components))

        if not scores:
            raise ValueError("No checkpoints with sufficient metrics")

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        best, best_score, components = sorted_scores[0]

        return CheckpointRecommendation(
            recommended_checkpoint=best,
            all_checkpoints=self.checkpoints,
            criteria_used="balanced",
            score=best_score,
            reasoning=f"Selected checkpoint at step {best.step} with balanced score ({best_score:.2f}). Components: {', '.join(components)}",
            alternatives=[c for c, _, _ in sorted_scores[1:4]],
        )

    def _select_by_custom_formula(self, formula: str) -> CheckpointRecommendation:
        """
        Select checkpoint using custom formula
        Formula can use: reward, kl, gsm8k, math, humaneval, mmlu
        Example: "0.5*gsm8k + 0.3*math - 0.2*kl"
        """
        scores = []

        for checkpoint in self.checkpoints:
            try:
                # Build variable context
                context = {
                    "reward": checkpoint.reward_mean or 0,
                    "kl": checkpoint.kl_divergence or 0,
                    "gsm8k": checkpoint.gsm8k_accuracy or 0,
                    "math": checkpoint.math_accuracy or 0,
                    "humaneval": checkpoint.humaneval_pass_rate or 0,
                    "mmlu": checkpoint.mmlu_accuracy or 0,
                    "step": checkpoint.step,
                }

                # Evaluate formula safely
                score = eval(formula, {"__builtins__": {}}, context)
                scores.append((checkpoint, score))
            except Exception:
                continue

        if not scores:
            raise ValueError("No checkpoints could be evaluated with custom formula")

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        best, best_score = sorted_scores[0]

        return CheckpointRecommendation(
            recommended_checkpoint=best,
            all_checkpoints=self.checkpoints,
            criteria_used="custom",
            score=best_score,
            reasoning=f"Selected checkpoint at step {best.step} using custom formula: {formula}",
            alternatives=[c for c, _ in sorted_scores[1:4]],
        )

    def get_checkpoint_timeline(self) -> List[Dict[str, Any]]:
        """
        Get timeline of all checkpoints with their metrics
        """
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.step)
        return [c.to_dict() for c in sorted_checkpoints]

    def detect_overfitting(self, window_size: int = 3) -> List[Dict[str, Any]]:
        """
        Detect potential overfitting by looking for:
        - Increasing KL divergence
        - Decreasing benchmark scores after peak
        - Reward plateau or decrease
        """
        if len(self.checkpoints) < window_size + 1:
            return []

        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.step)
        warnings = []

        # Check for KL divergence increasing
        kl_values = [(c.step, c.kl_divergence) for c in sorted_checkpoints if c.kl_divergence is not None]
        if len(kl_values) >= window_size:
            for i in range(len(kl_values) - window_size):
                window = [kl for _, kl in kl_values[i:i + window_size]]
                if all(window[j] < window[j + 1] for j in range(len(window) - 1)):
                    if window[-1] > 0.15:  # KL threshold
                        warnings.append({
                            "step": kl_values[i + window_size - 1][0],
                            "type": "kl_divergence_high",
                            "message": f"KL divergence increasing and above threshold ({window[-1]:.3f})",
                            "severity": "warning",
                        })

        # Check for benchmark decrease
        bench_values = [
            (c.step, c.get_benchmark_score())
            for c in sorted_checkpoints
        ]
        bench_values = [(s, b) for s, b in bench_values if b > 0]

        if len(bench_values) >= 3:
            peak_idx = max(range(len(bench_values)), key=lambda i: bench_values[i][1])
            if peak_idx < len(bench_values) - 1:
                peak_score = bench_values[peak_idx][1]
                final_score = bench_values[-1][1]
                if final_score < peak_score * 0.95:  # 5% decrease
                    warnings.append({
                        "step": bench_values[peak_idx][0],
                        "type": "benchmark_decrease",
                        "message": f"Benchmark peaked at step {bench_values[peak_idx][0]} ({peak_score:.1f}%), now at {final_score:.1f}%",
                        "severity": "info",
                    })

        return warnings

    def _extract_step(self, name: str) -> Optional[int]:
        """Extract step number from checkpoint name"""
        patterns = [
            r"ckpt[-_]?(\d+)",
            r"step[-_]?(\d+)",
            r"checkpoint[-_]?(\d+)",
            r"global_step[-_]?(\d+)",
            r"-(\d+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None


def select_best_checkpoint(
    experiment_path: str,
    criteria: str = "balanced",
    custom_formula: Optional[str] = None,
    benchmark_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for checkpoint selection
    """
    selector = CheckpointSelector()
    loaded = selector.load_from_experiment(experiment_path)

    if loaded == 0:
        return {
            "success": False,
            "message": f"No checkpoints found in {experiment_path}",
        }

    try:
        recommendation = selector.select_best(
            criteria=SelectionCriteria(criteria),
            custom_formula=custom_formula,
            benchmark_weights=benchmark_weights,
        )

        overfitting_warnings = selector.detect_overfitting()

        return {
            "success": True,
            "recommended": recommendation.recommended_checkpoint.to_dict(),
            "score": recommendation.score,
            "criteria": recommendation.criteria_used,
            "reasoning": recommendation.reasoning,
            "alternatives": [a.to_dict() for a in recommendation.alternatives],
            "timeline": selector.get_checkpoint_timeline(),
            "overfitting_warnings": overfitting_warnings,
        }

    except ValueError as e:
        return {
            "success": False,
            "message": str(e),
        }
