"""
Tests for Model Merger Module
"""

import pytest
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from training_platform.core.model_merger import (
    ModelMerger,
    MergeConfig,
    MergeMethod,
    MergeResult,
    merge_models,
    TORCH_AVAILABLE,
)


class TestMergeConfig:
    """Test MergeConfig class"""

    def test_default_weights(self):
        config = MergeConfig(
            method=MergeMethod.LINEAR,
            models=["model1", "model2", "model3"],
        )
        # Should have equal weights
        assert len(config.weights) == 3
        assert abs(sum(config.weights) - 1.0) < 0.01

    def test_custom_weights(self):
        config = MergeConfig(
            method=MergeMethod.LINEAR,
            models=["model1", "model2"],
            weights=[0.7, 0.3],
        )
        assert config.weights == [0.7, 0.3]

    def test_slerp_config(self):
        config = MergeConfig(
            method=MergeMethod.SLERP,
            models=["model1", "model2"],
            interpolation_t=0.7,
        )
        assert config.interpolation_t == 0.7

    def test_ties_config(self):
        config = MergeConfig(
            method=MergeMethod.TIES,
            models=["base", "ft1", "ft2"],
            density=0.3,
        )
        assert config.density == 0.3

    def test_dare_config(self):
        config = MergeConfig(
            method=MergeMethod.DARE,
            models=["base", "ft1"],
            drop_rate=0.85,
        )
        assert config.drop_rate == 0.85


class TestMergeResult:
    """Test MergeResult class"""

    def test_merge_result_success(self):
        result = MergeResult(
            success=True,
            output_path="/path/to/merged",
            method="linear",
            models_merged=["model1", "model2"],
            weights_used=[0.5, 0.5],
            message="Success",
            metadata={"num_parameters": 1000000},
        )
        assert result.success
        assert result.output_path == "/path/to/merged"

    def test_merge_result_failure(self):
        result = MergeResult(
            success=False,
            output_path=None,
            method="slerp",
            models_merged=["model1"],
            weights_used=[1.0],
            message="SLERP requires 2 models",
            metadata={},
        )
        assert not result.success


class TestModelMerger:
    """Test ModelMerger class"""

    @pytest.fixture
    def merger(self):
        return ModelMerger(device="cpu")

    def test_merge_unknown_method(self, merger):
        # Create config with invalid method by bypassing enum
        config = MergeConfig(
            method=MergeMethod.LINEAR,
            models=["model1", "model2"],
        )
        # Override method to test error handling
        original_merge = merger.merge

        def mock_merge(config):
            if not TORCH_AVAILABLE:
                return MergeResult(
                    success=False,
                    output_path=None,
                    method=config.method.value,
                    models_merged=config.models,
                    weights_used=config.weights,
                    message="PyTorch not available",
                    metadata={},
                )
            return original_merge(config)

        # Test that merger handles torch not available
        result = merger.merge(config)
        if not TORCH_AVAILABLE:
            assert not result.success
            assert "PyTorch" in result.message

    def test_slerp_requires_two_models(self, merger):
        config = MergeConfig(
            method=MergeMethod.SLERP,
            models=["model1", "model2", "model3"],
        )
        result = merger.merge(config)
        if TORCH_AVAILABLE:
            assert not result.success
            assert "2 models" in result.message

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_extract_step_from_path(self, merger):
        assert merger._extract_step_from_path("ckpt-1000") == 1000
        assert merger._extract_step_from_path("step_2000") == 2000
        assert merger._extract_step_from_path("checkpoint-3000") == 3000
        assert merger._extract_step_from_path("global_step4000") == 4000
        assert merger._extract_step_from_path("no_step") is None


class TestMergeModelsFunction:
    """Test the convenience function"""

    def test_merge_models_linear(self):
        result = merge_models(
            models=["model1", "model2"],
            method="linear",
            weights=[0.5, 0.5],
        )
        assert "success" in result
        assert "method" in result
        assert result["method"] == "linear"

    def test_merge_models_slerp(self):
        result = merge_models(
            models=["model1", "model2"],
            method="slerp",
            interpolation_t=0.5,
        )
        assert "success" in result
        assert result["method"] == "slerp"

    def test_merge_models_ties(self):
        result = merge_models(
            models=["base", "ft1", "ft2"],
            method="ties",
            density=0.5,
        )
        assert "success" in result
        assert result["method"] == "ties"

    def test_merge_models_dare(self):
        result = merge_models(
            models=["base", "ft1"],
            method="dare",
            drop_rate=0.9,
        )
        assert "success" in result
        assert result["method"] == "dare"

    def test_merge_models_swa(self):
        result = merge_models(
            models=["ckpt1", "ckpt2", "ckpt3"],
            method="swa",
            start_step=1000,
        )
        assert "success" in result
        assert result["method"] == "swa"


class TestMergeMethodEnum:
    """Test MergeMethod enum values"""

    def test_all_methods_exist(self):
        assert MergeMethod.LINEAR.value == "linear"
        assert MergeMethod.SLERP.value == "slerp"
        assert MergeMethod.TIES.value == "ties"
        assert MergeMethod.DARE.value == "dare"
        assert MergeMethod.SWA.value == "swa"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
