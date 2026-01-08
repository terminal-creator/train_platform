"""
verl Framework Adapter

Provides integration with verl training framework for:
- Configuration generation
- Training job execution via Ray
- Metrics collection
- Checkpoint management
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import logging
import tempfile

logger = logging.getLogger(__name__)


class VerlAlgorithm(Enum):
    """Supported verl training algorithms"""
    SFT = "sft"
    PPO = "ppo"
    GRPO = "grpo"
    DPO = "dpo"
    GSPO = "gspo"
    DAPO = "dapo"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class VerlTrainingConfig:
    """Configuration for verl training job"""
    # Model
    model_path: str
    model_size: Optional[str] = None

    # Algorithm
    algorithm: VerlAlgorithm = VerlAlgorithm.GRPO

    # Data
    train_data_path: str = ""
    eval_data_path: Optional[str] = None

    # Training params
    num_epochs: int = 3
    max_steps: Optional[int] = None
    learning_rate: float = 1e-6
    batch_size: int = 256
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 512
    max_response_length: int = 1024

    # RL specific
    kl_coef: float = 0.001
    entropy_coef: float = 0.0
    clip_ratio: float = 0.2
    rollout_n: int = 5
    use_kl_loss: bool = True

    # GRPO Reward Function Configuration
    reward_fn_type: str = "math_verify"  # math_verify, format_check, custom
    reward_fn_extract_answer: str = "boxed"  # boxed, last_number, json
    reward_fn_compare_method: str = "exact"  # exact, numeric, fuzzy
    reward_fn_answer_key: str = "solution"  # Key in dataset for ground truth
    reward_fn_custom_path: Optional[str] = None  # Path to custom reward function script

    # PPO Reward Model Configuration
    reward_model_path: Optional[str] = None  # Path to reward model
    reward_model_enable_gc: bool = True  # Enable gradient checkpointing
    reward_model_offload: bool = False  # Offload params to CPU
    reward_model_micro_batch: int = 4

    # Unified Reward Script Configuration (for PPO/GRPO/GSPO)
    reward_script_path: Optional[str] = None  # Path to reward script
    reward_script_type: str = "rule"  # rule, api, model
    reward_script_metadata: Optional[Dict[str, Any]] = None  # Extra params for reward script

    # LoRA
    lora_enabled: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Resources
    num_gpus: int = 8
    num_nodes: int = 1
    gpu_type: str = "A100-80G"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.6

    # Checkpointing
    checkpoint_interval: int = 500
    eval_interval: int = 100
    output_dir: str = "./outputs"
    project_name: str = "training_platform"
    experiment_name: str = ""

    # Advanced
    zero_stage: int = 2
    param_offload: bool = False
    optimizer_offload: bool = False
    activation_checkpointing: bool = True
    sequence_length: int = 4096

    # Metrics callback
    metrics_endpoint: Optional[str] = None  # HTTP endpoint to push metrics

    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint directory
    resume_mode: str = "auto"  # auto, disable, or resume_path

    def _to_sft_args(self) -> List[str]:
        """Generate arguments for verl's fsdp_sft_trainer"""
        args = []

        # Data configuration
        args.append(f"data.train_files={self.train_data_path}")
        # Use eval_data_path if provided, otherwise use train file as validation
        val_files = self.eval_data_path if self.eval_data_path else self.train_data_path
        args.append(f"data.val_files={val_files}")
        args.append(f"data.train_batch_size={self.batch_size}")
        # micro_batch_size must be <= batch_size and batch_size % micro_batch_size == 0
        effective_micro_batch = min(self.micro_batch_size, self.batch_size)
        args.append(f"data.micro_batch_size_per_gpu={effective_micro_batch}")
        args.append(f"data.max_length={self.sequence_length}")
        args.append("data.prompt_key=prompt")
        args.append("data.response_key=response")

        # Model configuration (use partial_pretrain for model path)
        args.append(f"model.partial_pretrain={self.model_path}")
        args.append(f"model.enable_gradient_checkpointing={str(self.activation_checkpointing)}")
        args.append("model.trust_remote_code=True")

        # LoRA if enabled
        if self.lora_enabled:
            args.append(f"model.lora_rank={self.lora_rank}")
            args.append(f"model.lora_alpha={self.lora_alpha}")
        else:
            args.append("model.lora_rank=0")

        # Optimizer configuration
        args.append(f"optim.lr={self.learning_rate}")
        args.append("optim.weight_decay=0.01")
        args.append("optim.lr_warmup_steps_ratio=0.1")

        # Trainer configuration
        args.append(f"trainer.total_epochs={self.num_epochs}")
        if self.max_steps:
            args.append(f"trainer.total_training_steps={self.max_steps}")
        args.append(f"trainer.save_freq={self.checkpoint_interval}")
        args.append(f"trainer.project_name={self.project_name}")
        exp_name = self.experiment_name or f"sft_{Path(self.model_path).name}"
        args.append(f"trainer.experiment_name={exp_name}")
        args.append(f"trainer.default_local_dir={self.output_dir}")
        args.append("trainer.logger=[console]")
        args.append(f"trainer.n_gpus_per_node={self.num_gpus}")
        args.append(f"trainer.nnodes={self.num_nodes}")

        return args

    def to_verl_command_args(self) -> List[str]:
        """
        Generate verl Hydra-style command line arguments.

        This follows verl's actual configuration pattern.
        """
        # SFT has different parameter structure
        if self.algorithm == VerlAlgorithm.SFT:
            return self._to_sft_args()

        args = []

        # Algorithm
        if self.algorithm == VerlAlgorithm.GRPO:
            args.append("algorithm.adv_estimator=grpo")
        elif self.algorithm == VerlAlgorithm.PPO:
            args.append("algorithm.adv_estimator=gae")
        elif self.algorithm == VerlAlgorithm.DPO:
            args.append("algorithm=dpo")

        # Data configuration
        args.append(f"data.train_files={self.train_data_path}")
        if self.eval_data_path:
            args.append(f"data.val_files={self.eval_data_path}")
        args.append(f"data.train_batch_size={self.batch_size}")
        args.append(f"data.max_prompt_length={self.max_prompt_length}")
        args.append(f"data.max_response_length={self.max_response_length}")
        args.append("data.filter_overlong_prompts=True")
        args.append("data.truncation=error")

        # Model configuration
        args.append(f"actor_rollout_ref.model.path={self.model_path}")
        args.append(f"actor_rollout_ref.model.enable_gradient_checkpointing={str(self.activation_checkpointing)}")
        args.append("actor_rollout_ref.model.use_remove_padding=True")

        # Actor (training) configuration
        args.append(f"actor_rollout_ref.actor.optim.lr={self.learning_rate}")
        args.append(f"actor_rollout_ref.actor.ppo_mini_batch_size={self.batch_size // 4}")
        args.append(f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={self.micro_batch_size}")
        args.append(f"actor_rollout_ref.actor.use_kl_loss={str(self.use_kl_loss)}")
        args.append(f"actor_rollout_ref.actor.kl_loss_coef={self.kl_coef}")
        args.append("actor_rollout_ref.actor.kl_loss_type=low_var_kl")
        args.append(f"actor_rollout_ref.actor.entropy_coeff={self.entropy_coef}")

        # FSDP offload
        args.append(f"actor_rollout_ref.actor.fsdp_config.param_offload={str(self.param_offload)}")
        args.append(f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(self.optimizer_offload)}")

        # Rollout (inference) configuration
        args.append(f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.tensor_parallel_size}")
        args.append("actor_rollout_ref.rollout.name=vllm")
        args.append(f"actor_rollout_ref.rollout.gpu_memory_utilization={self.gpu_memory_utilization}")
        args.append(f"actor_rollout_ref.rollout.n={self.rollout_n}")
        args.append(f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={self.micro_batch_size * 2}")

        # Reference model
        args.append(f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={self.micro_batch_size * 2}")
        args.append("actor_rollout_ref.ref.fsdp_config.param_offload=True")

        # Algorithm specific
        args.append("algorithm.use_kl_in_reward=False")

        # Reward configuration for GRPO/PPO/GSPO
        if self.algorithm in [VerlAlgorithm.GRPO, VerlAlgorithm.GSPO]:
            # Use unified reward script if provided
            if self.reward_script_path:
                args.append(f"reward_model.reward_script_path={self.reward_script_path}")
                args.append(f"reward_model.reward_script_type={self.reward_script_type}")
                if self.reward_script_metadata:
                    # Serialize metadata to JSON for passing to verl
                    import json
                    metadata_json = json.dumps(self.reward_script_metadata)
                    args.append(f"reward_model.reward_script_metadata='{metadata_json}'")
            else:
                # Fallback to legacy reward function config
                args.append(f"reward_model.reward_fn_type={self.reward_fn_type}")
                if self.reward_fn_type == "math_verify":
                    args.append(f"reward_model.extract_answer={self.reward_fn_extract_answer}")
                    args.append(f"reward_model.compare_method={self.reward_fn_compare_method}")
                if self.reward_fn_custom_path:
                    args.append(f"reward_model.custom_path={self.reward_fn_custom_path}")

        # PPO specific: Reward model (critic) configuration
        if self.algorithm == VerlAlgorithm.PPO:
            # Use reward script if provided
            if self.reward_script_path:
                args.append(f"reward_model.reward_script_path={self.reward_script_path}")
                args.append(f"reward_model.reward_script_type={self.reward_script_type}")
                if self.reward_script_metadata:
                    import json
                    metadata_json = json.dumps(self.reward_script_metadata)
                    args.append(f"reward_model.reward_script_metadata='{metadata_json}'")

            # Add critic (value function) model if provided
            if self.reward_model_path:
                args.append(f"critic.model.path={self.reward_model_path}")
                args.append(f"critic.model.enable_gradient_checkpointing={str(self.reward_model_enable_gc)}")
                args.append(f"critic.model.fsdp_config.param_offload={str(self.reward_model_offload)}")
                args.append(f"critic.ppo_micro_batch_size_per_gpu={self.reward_model_micro_batch}")

        # Trainer configuration
        args.append("trainer.critic_warmup=0")
        args.append("trainer.logger=console")
        args.append(f"trainer.project_name={self.project_name}")
        exp_name = self.experiment_name or f"{self.algorithm.value}_{Path(self.model_path).name}"
        args.append(f"trainer.experiment_name={exp_name}")
        args.append(f"trainer.n_gpus_per_node={self.num_gpus}")
        args.append(f"trainer.nnodes={self.num_nodes}")
        args.append(f"trainer.save_freq={self.checkpoint_interval}")
        args.append(f"trainer.test_freq={self.eval_interval}")
        args.append(f"trainer.total_epochs={self.num_epochs}")
        args.append(f"trainer.default_local_dir={self.output_dir}")

        # LoRA if enabled
        if self.lora_enabled:
            args.append("actor_rollout_ref.actor.lora.enabled=True")
            args.append(f"actor_rollout_ref.actor.lora.rank={self.lora_rank}")
            args.append(f"actor_rollout_ref.actor.lora.alpha={self.lora_alpha}")

        # Resume from checkpoint
        args.append(f"trainer.resume_mode={self.resume_mode}")
        if self.resume_from_checkpoint:
            args.append(f"trainer.resume_from_path={self.resume_from_checkpoint}")

        return args

    def to_command(self) -> str:
        """Generate full verl training command as string (for display/logging)"""
        if self.algorithm == VerlAlgorithm.SFT:
            cmd = "python3 -m verl.trainer.fsdp_sft_trainer"
        else:
            cmd = "python3 -m verl.trainer.main_ppo"

        args = self.to_verl_command_args()
        return f"{cmd} \\\n    " + " \\\n    ".join(args)

    def to_command_list(self) -> List[str]:
        """
        Generate full verl training command as list (safe for subprocess without shell=True).

        Returns:
            List of command and arguments, e.g. ['python3', '-m', 'verl.trainer.main_ppo', 'arg1', 'arg2']
        """
        if self.algorithm == VerlAlgorithm.SFT:
            cmd_list = ["python3", "-m", "verl.trainer.fsdp_sft_trainer"]
        else:
            cmd_list = ["python3", "-m", "verl.trainer.main_ppo"]

        # Add all arguments
        args = self.to_verl_command_args()
        cmd_list.extend(args)

        return cmd_list

    def to_shell_script(self, script_path: str = None) -> str:
        """Generate a shell script for running the training"""
        script = f"""#!/bin/bash
set -x

# verl Training Script
# Generated by Training Platform

export PYTHONPATH="{os.environ.get('PYTHONPATH', '')}:${{PYTHONPATH:-}}"

# Ensure output directory exists
mkdir -p {self.output_dir}

# Run training
{self.to_command()}

echo "Training completed with exit code: $?"
"""
        if script_path:
            with open(script_path, 'w') as f:
                f.write(script)
            os.chmod(script_path, 0o755)

        return script


def create_training_script(config: VerlTrainingConfig, output_path: str) -> str:
    """
    Create a training script file that can be submitted to Ray.

    Args:
        config: Training configuration
        output_path: Where to save the script

    Returns:
        Path to the created script
    """
    script_content = config.to_shell_script()

    with open(output_path, 'w') as f:
        f.write(script_content)
    os.chmod(output_path, 0o755)

    logger.info(f"Created training script: {output_path}")
    return output_path


def create_ray_entrypoint(config: VerlTrainingConfig) -> str:
    """
    Create an entrypoint command for Ray job submission.

    This returns a single command that can be passed to Ray's submit_job.
    """
    if config.algorithm == VerlAlgorithm.SFT:
        # SFT uses torchrun for distributed training
        base_cmd = f"torchrun --standalone --nnodes=1 --nproc_per_node={config.num_gpus} -m verl.trainer.fsdp_sft_trainer"
    else:
        base_cmd = "python3 -m verl.trainer.main_ppo"

    args = config.to_verl_command_args()
    return f"{base_cmd} " + " ".join(args)


class VerlJobRunner:
    """
    Runs verl training jobs directly (for local development/testing)
    """

    def __init__(self, verl_path: Optional[str] = None):
        """
        Initialize verl job runner

        Args:
            verl_path: Path to verl installation. If None, assumes verl is in PYTHONPATH
        """
        self.verl_path = verl_path or self._find_verl_path()

    def _find_verl_path(self) -> Optional[str]:
        """Find verl installation path"""
        possible_paths = [
            Path(__file__).parent.parent.parent / "verl",
            Path.home() / "verl",
            Path("/opt/verl"),
        ]

        for path in possible_paths:
            if path.exists() and (path / "verl").exists():
                return str(path)

        return None

    def run_training(
        self,
        config: VerlTrainingConfig,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run verl training job locally.

        For production use, submit via Ray instead.
        """
        try:
            # Get command as list (safe - no shell injection)
            cmd_list = config.to_command_list()

            # Also log human-readable command
            cmd_str = config.to_command()
            logger.info(f"Starting verl training:\n{cmd_str}")

            # Run process WITHOUT shell=True for security
            process = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.verl_path,
                env={**os.environ, "PYTHONPATH": f"{self.verl_path}:{os.environ.get('PYTHONPATH', '')}"},
            )

            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                if callback:
                    callback({"type": "log", "data": line.strip()})

                # Parse metrics from output
                metrics = self._parse_metrics(line)
                if metrics and callback:
                    callback({"type": "metrics", "data": metrics})

            process.wait()

            return {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "output": "".join(output_lines),
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e)}

    def _parse_metrics(self, line: str) -> Optional[Dict[str, float]]:
        """Parse metrics from log line"""
        import re

        metrics = {}
        patterns = [
            (r"step[=:\s]+(\d+)", "step"),
            (r"reward[=:\s]+([\d.-]+)", "reward"),
            (r"kl[=:\s]+([\d.-]+)", "kl"),
            (r"policy_loss[=:\s]+([\d.-]+)", "policy_loss"),
            (r"value_loss[=:\s]+([\d.-]+)", "value_loss"),
            (r"entropy[=:\s]+([\d.-]+)", "entropy"),
            (r"lr[=:\s]+([\d.e-]+)", "learning_rate"),
        ]

        for pattern, name in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    metrics[name] = float(match.group(1))
                except ValueError:
                    pass

        return metrics if metrics else None


class VerlMetricsCollector:
    """
    Collects and processes metrics from verl training
    """

    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []

    def add_metrics(self, metrics: Dict[str, Any]):
        """Add metrics to history"""
        self.metrics_history.append(metrics)

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get latest metrics"""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_history(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics history with optional filtering"""
        history = self.metrics_history

        if start_step is not None:
            history = [m for m in history if m.get("step", 0) >= start_step]
        if end_step is not None:
            history = [m for m in history if m.get("step", 0) <= end_step]

        return history

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.metrics_history:
            return {}

        import statistics

        def safe_stats(values):
            if not values:
                return {"mean": 0, "min": 0, "max": 0, "std": 0}
            return {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
            }

        rewards = [m["reward"] for m in self.metrics_history if "reward" in m]
        kls = [m["kl"] for m in self.metrics_history if "kl" in m]
        losses = [m["policy_loss"] for m in self.metrics_history if "policy_loss" in m]

        return {
            "total_steps": len(self.metrics_history),
            "reward": safe_stats(rewards),
            "kl": safe_stats(kls),
            "policy_loss": safe_stats(losses),
        }


def create_verl_training_config(
    model_path: str,
    algorithm: str = "grpo",
    train_data_path: str = "",
    num_gpus: int = 8,
    lora_enabled: bool = False,
    **kwargs,
) -> VerlTrainingConfig:
    """
    Convenience function to create verl training configuration
    """
    return VerlTrainingConfig(
        model_path=model_path,
        algorithm=VerlAlgorithm(algorithm),
        train_data_path=train_data_path,
        num_gpus=num_gpus,
        lora_enabled=lora_enabled,
        **kwargs,
    )
