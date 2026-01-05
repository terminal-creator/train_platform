"""
Tests for Memory Estimator Module
"""

import pytest
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from training_platform.core.memory_estimator import (
    MemoryEstimator,
    ModelConfig,
    GPUConfig,
    LoRAConfig,
    Precision,
    ZeROStage,
    TrainingType,
    estimate_memory,
)


class TestModelConfig:
    """Test ModelConfig class"""

    def test_from_model_size_7b(self):
        config = ModelConfig.from_model_size("7B")
        assert config.params_billion == 7
        assert config.hidden_size == 3584
        assert config.num_layers == 28

    def test_from_model_size_72b(self):
        config = ModelConfig.from_model_size("72B")
        assert config.params_billion == 72
        assert config.hidden_size == 8192
        assert config.num_layers == 80

    def test_from_model_size_default(self):
        config = ModelConfig.from_model_size("unknown")
        assert config.params_billion == 7  # Falls back to 7B


class TestGPUConfig:
    """Test GPUConfig class"""

    def test_from_gpu_type_a100_80g(self):
        config = GPUConfig.from_gpu_type("A100-80G")
        assert config.memory_gb == 80
        assert config.fp16_tflops == 312

    def test_from_gpu_type_h100(self):
        config = GPUConfig.from_gpu_type("H100-80G")
        assert config.memory_gb == 80
        assert config.fp16_tflops == 989

    def test_from_gpu_type_rtx4090(self):
        config = GPUConfig.from_gpu_type("RTX4090")
        assert config.memory_gb == 24

    def test_from_gpu_type_apple_silicon_m3_max(self):
        config = GPUConfig.from_gpu_type("M3-Max-128G")
        assert config.memory_gb == 128
        assert config.is_apple_silicon == True
        assert config.fp16_tflops == 14.2

    def test_from_gpu_type_apple_silicon_m4_max(self):
        config = GPUConfig.from_gpu_type("M4-Max-36G")
        assert config.memory_gb == 36
        assert config.is_apple_silicon == True

    def test_apple_silicon_not_nvidia(self):
        nvidia_config = GPUConfig.from_gpu_type("A100-80G")
        apple_config = GPUConfig.from_gpu_type("M3-Max-128G")
        assert nvidia_config.is_apple_silicon == False
        assert apple_config.is_apple_silicon == True


class TestLoRAConfig:
    """Test LoRAConfig class"""

    def test_default_target_modules(self):
        config = LoRAConfig(enabled=True, rank=8)
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules


class TestMemoryEstimator:
    """Test MemoryEstimator class"""

    @pytest.fixture
    def estimator_7b(self):
        return MemoryEstimator(
            model_config=ModelConfig.from_model_size("7B"),
            gpu_config=GPUConfig.from_gpu_type("A100-80G"),
            num_gpus=8,
            precision=Precision.BF16,
            zero_stage=ZeROStage.ZERO_2,
            training_type=TrainingType.GRPO,
        )

    def test_estimate_model_weights(self, estimator_7b):
        weights = estimator_7b.estimate_model_weights()
        # 7B params * 2 bytes (BF16) = ~14GB
        assert 12 < weights < 16

    def test_estimate_model_weights_zero3(self):
        estimator = MemoryEstimator(
            model_config=ModelConfig.from_model_size("7B"),
            gpu_config=GPUConfig.from_gpu_type("A100-80G"),
            num_gpus=8,
            precision=Precision.BF16,
            zero_stage=ZeROStage.ZERO_3,
            training_type=TrainingType.GRPO,
        )
        weights = estimator.estimate_model_weights()
        # Zero-3 shards model weights across GPUs
        assert weights < 3  # 14GB / 8 = ~1.75GB

    def test_estimate_optimizer_states(self, estimator_7b):
        optimizer = estimator_7b.estimate_optimizer_states()
        # 7B * 12 bytes / 8 GPUs = ~10.5GB
        assert 8 < optimizer < 15

    def test_estimate_optimizer_states_with_lora(self):
        estimator = MemoryEstimator(
            model_config=ModelConfig.from_model_size("7B"),
            gpu_config=GPUConfig.from_gpu_type("A100-80G"),
            num_gpus=8,
            precision=Precision.BF16,
            zero_stage=ZeROStage.ZERO_2,
            training_type=TrainingType.GRPO,
            lora_config=LoRAConfig(enabled=True, rank=8),
        )
        optimizer = estimator.estimate_optimizer_states()
        # LoRA reduces optimizer states significantly
        assert optimizer < 0.5  # Much smaller than full fine-tuning

    def test_estimate_gradients(self, estimator_7b):
        gradients = estimator_7b.estimate_gradients()
        # 7B * 2 bytes / 8 GPUs (ZeRO-2) = ~1.75GB
        assert 1 < gradients < 3

    def test_estimate_activations(self, estimator_7b):
        activations = estimator_7b.estimate_activations(batch_size=4, seq_len=4096)
        assert activations > 0

    def test_estimate_kv_cache(self, estimator_7b):
        kv_cache = estimator_7b.estimate_kv_cache(batch_size=4, seq_len=4096)
        assert kv_cache > 0

    def test_estimate_total(self, estimator_7b):
        breakdown = estimator_7b.estimate_total(batch_size=4, seq_len=4096)
        assert breakdown.total_gb > 0
        assert breakdown.per_gpu_gb > 0
        assert 0 < breakdown.utilization_percent <= 100 or breakdown.utilization_percent > 100  # Can exceed if OOM

    def test_recommend_max_batch_size(self, estimator_7b):
        max_batch = estimator_7b.recommend_max_batch_size(seq_len=4096)
        assert max_batch >= 1

    def test_recommend_zero_stage_small_model(self):
        estimator = MemoryEstimator(
            model_config=ModelConfig.from_model_size("0.5B"),
            gpu_config=GPUConfig.from_gpu_type("A100-80G"),
            num_gpus=8,
            precision=Precision.BF16,
            zero_stage=ZeROStage.ZERO_2,
            training_type=TrainingType.SFT,
        )
        recommended = estimator.recommend_zero_stage()
        assert recommended == ZeROStage.ZERO_0  # Small model doesn't need ZeRO

    def test_recommend_zero_stage_large_model(self):
        estimator = MemoryEstimator(
            model_config=ModelConfig.from_model_size("72B"),
            gpu_config=GPUConfig.from_gpu_type("A100-80G"),
            num_gpus=8,
            precision=Precision.BF16,
            zero_stage=ZeROStage.ZERO_2,
            training_type=TrainingType.GRPO,
        )
        recommended = estimator.recommend_zero_stage()
        assert recommended in [ZeROStage.ZERO_2, ZeROStage.ZERO_3]


class TestEstimateMemoryFunction:
    """Test the convenience function"""

    def test_estimate_memory_basic(self):
        result = estimate_memory(
            model_size="7B",
            gpu_type="A100-80G",
            num_gpus=8,
            batch_size=4,
            seq_len=4096,
        )
        assert "breakdown" in result
        assert "recommended_zero_stage" in result
        assert "recommended_max_batch_size" in result
        assert "model_info" in result
        assert "gpu_info" in result

    def test_estimate_memory_with_lora(self):
        result = estimate_memory(
            model_size="7B",
            gpu_type="A100-80G",
            num_gpus=8,
            batch_size=4,
            seq_len=4096,
            lora_enabled=True,
            lora_rank=16,
        )
        assert result["breakdown"]["optimizer_states_gb"] < 1  # Much smaller with LoRA

    def test_estimate_memory_training_types(self):
        for training_type in ["sft", "ppo", "grpo", "dpo", "gspo"]:
            result = estimate_memory(
                model_size="7B",
                gpu_type="A100-80G",
                num_gpus=8,
                training_type=training_type,
            )
            assert result["breakdown"]["total_gb"] > 0

    def test_estimate_memory_generates_warnings(self):
        # Use large model with few GPUs to trigger warning
        result = estimate_memory(
            model_size="72B",
            gpu_type="A100-80G",
            num_gpus=2,
            batch_size=4,
            seq_len=4096,
        )
        assert "warnings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
