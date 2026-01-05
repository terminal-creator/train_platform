"""
Tests for Compute Calculator Module
"""

import pytest
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from training_platform.core.compute_calculator import (
    ComputeCalculator,
    VerlConfig,
    calculate_compute_config,
)
from training_platform.core.memory_estimator import ZeROStage


class TestVerlConfig:
    """Test VerlConfig class"""

    def test_to_dict(self):
        config = VerlConfig(
            actor_strategy="fsdp",
            actor_fsdp_sharding="SHARD_GRAD_OP",
            actor_micro_batch_size=32,
            actor_micro_batch_size_per_gpu=4,
            actor_lr=1e-6,
            rollout_tp_size=1,
            rollout_gpu_memory_utilization=0.85,
            rollout_n=8,
            gradient_accumulation_steps=4,
            global_batch_size=256,
            total_epochs=3,
            algorithm="grpo",
            kl_coef=0.02,
            entropy_coef=0.0,
            clip_ratio=0.2,
            critic_enabled=False,
            critic_lr=None,
            lora_enabled=False,
            lora_rank=8,
            lora_alpha=16,
            cpu_offload=False,
            activation_checkpointing=True,
            ref_offload=False,
        )

        config_dict = config.to_dict()

        assert config_dict["actor"]["strategy"] == "fsdp"
        assert config_dict["actor"]["micro_batch_size"] == 32
        assert config_dict["rollout"]["n"] == 8
        assert config_dict["algorithm"]["name"] == "grpo"

    def test_to_yaml_string(self):
        config = VerlConfig(
            actor_strategy="fsdp",
            actor_fsdp_sharding="SHARD_GRAD_OP",
            actor_micro_batch_size=32,
            actor_micro_batch_size_per_gpu=4,
            actor_lr=1e-6,
            rollout_tp_size=1,
            rollout_gpu_memory_utilization=0.85,
            rollout_n=8,
            gradient_accumulation_steps=4,
            global_batch_size=256,
            total_epochs=3,
            algorithm="grpo",
            kl_coef=0.02,
            entropy_coef=0.0,
            clip_ratio=0.2,
            critic_enabled=False,
            critic_lr=None,
            lora_enabled=False,
            lora_rank=8,
            lora_alpha=16,
            cpu_offload=False,
            activation_checkpointing=True,
            ref_offload=False,
        )

        yaml_str = config.to_yaml_string()
        assert "actor:" in yaml_str
        assert "rollout:" in yaml_str


class TestComputeCalculator:
    """Test ComputeCalculator class"""

    @pytest.fixture
    def calculator_7b(self):
        return ComputeCalculator(
            model_size="7B",
            gpu_type="A100-80G",
            num_gpus=8,
            context_length=4096,
            training_type="grpo",
        )

    def test_calculate_basic(self, calculator_7b):
        result = calculator_7b.calculate()

        assert "config" in result
        assert "yaml" in result
        assert "memory_estimate" in result
        assert "zero_stage" in result
        assert "summary" in result

    def test_calculate_grpo(self, calculator_7b):
        result = calculator_7b.calculate()

        # GRPO should use rollout_n=8
        assert result["config"]["rollout"]["n"] == 8
        # GRPO doesn't need critic
        assert result["config"]["critic"]["enabled"] is False

    def test_calculate_ppo(self):
        calculator = ComputeCalculator(
            model_size="7B",
            gpu_type="A100-80G",
            num_gpus=8,
            training_type="ppo",
        )
        result = calculator.calculate()

        # PPO should use rollout_n=1
        assert result["config"]["rollout"]["n"] == 1
        # PPO needs critic
        assert result["config"]["critic"]["enabled"] is True

    def test_calculate_sft(self):
        calculator = ComputeCalculator(
            model_size="7B",
            gpu_type="A100-80G",
            num_gpus=8,
            training_type="sft",
        )
        result = calculator.calculate()

        # SFT doesn't use RL params
        assert result["config"]["algorithm"]["kl_coef"] == 0.0
        assert result["config"]["critic"]["enabled"] is False

    def test_calculate_with_lora(self):
        calculator = ComputeCalculator(
            model_size="7B",
            gpu_type="A100-80G",
            num_gpus=8,
            training_type="grpo",
            lora_enabled=True,
            lora_rank=16,
        )
        result = calculator.calculate()

        assert result["config"]["actor"]["lora"]["enabled"] is True
        assert result["config"]["actor"]["lora"]["rank"] == 16
        # LoRA should have higher LR
        assert result["config"]["actor"]["optim"]["lr"] > 1e-6

    def test_calculate_large_model(self):
        calculator = ComputeCalculator(
            model_size="72B",
            gpu_type="A100-80G",
            num_gpus=8,
            training_type="grpo",
        )
        result = calculator.calculate()

        # Large model should use ZeRO-2 or ZeRO-3
        assert result["zero_stage"] >= 2
        # Should generate warnings
        assert len(result["warnings"]) > 0 or len(result["recommendations"]) > 0

    def test_determine_zero_stage(self, calculator_7b):
        zero_stage = calculator_7b._determine_zero_stage()
        assert isinstance(zero_stage, ZeROStage)

    def test_determine_tp_size(self, calculator_7b):
        tp_size = calculator_7b._determine_tp_size()
        assert tp_size >= 1

    def test_summary_fields(self, calculator_7b):
        result = calculator_7b.calculate()
        summary = result["summary"]

        assert "model" in summary
        assert "gpus" in summary
        assert "training_type" in summary
        assert "micro_batch_size" in summary
        assert "global_batch_size" in summary


class TestCalculateComputeConfigFunction:
    """Test the convenience function"""

    def test_calculate_compute_config_basic(self):
        result = calculate_compute_config(
            model_size="7B",
            gpu_type="A100-80G",
            num_gpus=8,
        )
        assert "config" in result
        assert "yaml" in result

    def test_calculate_compute_config_all_training_types(self):
        for training_type in ["sft", "ppo", "grpo", "dpo", "gspo"]:
            result = calculate_compute_config(
                model_size="7B",
                gpu_type="A100-80G",
                num_gpus=8,
                training_type=training_type,
            )
            assert result["config"]["algorithm"]["name"] == training_type

    def test_calculate_compute_config_all_model_sizes(self):
        for model_size in ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]:
            result = calculate_compute_config(
                model_size=model_size,
                gpu_type="A100-80G",
                num_gpus=8,
            )
            assert result["summary"]["model"] == model_size

    def test_calculate_compute_config_all_gpu_types(self):
        for gpu_type in ["A100-40G", "A100-80G", "H100-80G", "RTX4090", "L40S"]:
            result = calculate_compute_config(
                model_size="7B",
                gpu_type=gpu_type,
                num_gpus=8,
            )
            assert gpu_type in result["summary"]["gpus"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
