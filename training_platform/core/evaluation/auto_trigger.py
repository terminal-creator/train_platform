"""
Auto-trigger Evaluation

Triggers evaluation automatically after training completion or at intervals.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvalTriggerConfig:
    """Configuration for auto-triggered evaluation."""
    benchmarks: List[str] = field(default_factory=lambda: ["gsm8k"])
    trigger_on_complete: bool = True
    trigger_every_n_steps: Optional[int] = None
    trigger_every_n_hours: Optional[float] = None
    max_eval_samples: Optional[int] = None
    async_eval: bool = True  # Don't block training


class EvalTrigger:
    """Manages auto-triggered evaluations."""

    def __init__(self, config: EvalTriggerConfig = None):
        self.config = config or EvalTriggerConfig()
        self._last_trigger_step = 0
        self._last_trigger_time = datetime.utcnow()
        self._eval_history: List[Dict[str, Any]] = []

    def should_trigger(self, current_step: int) -> bool:
        """Check if evaluation should be triggered at current step."""
        # Step-based trigger
        if self.config.trigger_every_n_steps:
            if current_step - self._last_trigger_step >= self.config.trigger_every_n_steps:
                return True

        # Time-based trigger
        if self.config.trigger_every_n_hours:
            elapsed = (datetime.utcnow() - self._last_trigger_time).total_seconds() / 3600
            if elapsed >= self.config.trigger_every_n_hours:
                return True

        return False

    def trigger_evaluation(
        self,
        job_id: str,
        checkpoint_path: str,
        step: int,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Trigger evaluation for a checkpoint.

        Args:
            job_id: Training job ID
            checkpoint_path: Path to model checkpoint
            step: Current training step
            callback: Optional callback with results

        Returns:
            Evaluation task info
        """
        self._last_trigger_step = step
        self._last_trigger_time = datetime.utcnow()

        eval_task = {
            "job_id": job_id,
            "checkpoint_path": checkpoint_path,
            "step": step,
            "benchmarks": self.config.benchmarks,
            "status": "pending",
            "triggered_at": datetime.utcnow().isoformat(),
            "max_samples": self.config.max_eval_samples,
        }

        self._eval_history.append(eval_task)

        logger.info(
            f"Triggered evaluation for job {job_id} at step {step} "
            f"on benchmarks: {self.config.benchmarks}"
        )

        return eval_task

    def on_training_complete(self, job_id: str, checkpoint_path: str, total_steps: int) -> Optional[Dict]:
        """Called when training completes. Triggers evaluation if configured."""
        if self.config.trigger_on_complete:
            return self.trigger_evaluation(job_id, checkpoint_path, total_steps)
        return None

    def on_checkpoint_saved(self, job_id: str, checkpoint_path: str, step: int) -> Optional[Dict]:
        """Called when a checkpoint is saved. Triggers evaluation if interval reached."""
        if self.should_trigger(step):
            return self.trigger_evaluation(job_id, checkpoint_path, step)
        return None

    def get_history(self) -> List[Dict[str, Any]]:
        """Get evaluation trigger history."""
        return self._eval_history

    def record_result(self, step: int, results: Dict[str, Any]):
        """Record evaluation results for a step."""
        for task in self._eval_history:
            if task["step"] == step:
                task["status"] = "completed"
                task["results"] = results
                task["completed_at"] = datetime.utcnow().isoformat()
                break
