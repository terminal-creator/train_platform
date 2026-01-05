"""
Tests for Checkpoint Selector Module
"""

import pytest
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from training_platform.core.checkpoint_selector import (
    CheckpointSelector,
    CheckpointMetrics,
    CheckpointRecommendation,
    SelectionCriteria,
    select_best_checkpoint,
)


class TestCheckpointMetrics:
    """Test CheckpointMetrics class"""

    @pytest.fixture
    def sample_checkpoint(self):
        return CheckpointMetrics(
            step=1000,
            path="/path/to/ckpt-1000",
            reward_mean=0.75,
            reward_std=0.1,
            kl_divergence=0.05,
            policy_loss=0.5,
            gsm8k_accuracy=55.0,
            math_accuracy=20.0,
            humaneval_pass_rate=30.0,
            mmlu_accuracy=60.0,
        )

    def test_to_dict(self, sample_checkpoint):
        result = sample_checkpoint.to_dict()
        assert result["step"] == 1000
        assert result["reward_mean"] == 0.75
        assert result["benchmarks"]["gsm8k"] == 55.0

    def test_get_benchmark_score_default_weights(self, sample_checkpoint):
        score = sample_checkpoint.get_benchmark_score()
        # Default weights: gsm8k=0.4, math=0.4, humaneval=0.1, mmlu=0.1
        # 55*0.4 + 20*0.4 + 30*0.1 + 60*0.1 = 22 + 8 + 3 + 6 = 39
        assert abs(score - 39.0) < 0.1

    def test_get_benchmark_score_custom_weights(self, sample_checkpoint):
        weights = {"gsm8k": 1.0, "math": 0.0, "humaneval": 0.0, "mmlu": 0.0}
        score = sample_checkpoint.get_benchmark_score(weights)
        assert abs(score - 55.0) < 0.1

    def test_get_benchmark_score_no_data(self):
        checkpoint = CheckpointMetrics(step=1000, path="/path")
        score = checkpoint.get_benchmark_score()
        assert score == 0.0


class TestCheckpointSelector:
    """Test CheckpointSelector class"""

    @pytest.fixture
    def selector_with_checkpoints(self):
        selector = CheckpointSelector()
        # Add checkpoints with varying metrics
        selector.add_checkpoint(CheckpointMetrics(
            step=1000,
            path="/ckpt-1000",
            reward_mean=0.5,
            kl_divergence=0.02,
            gsm8k_accuracy=50.0,
            math_accuracy=15.0,
        ))
        selector.add_checkpoint(CheckpointMetrics(
            step=2000,
            path="/ckpt-2000",
            reward_mean=0.7,
            kl_divergence=0.05,
            gsm8k_accuracy=58.0,
            math_accuracy=22.0,
        ))
        selector.add_checkpoint(CheckpointMetrics(
            step=3000,
            path="/ckpt-3000",
            reward_mean=0.65,
            kl_divergence=0.12,
            gsm8k_accuracy=60.0,
            math_accuracy=25.0,
        ))
        return selector

    def test_add_checkpoint(self):
        selector = CheckpointSelector()
        checkpoint = CheckpointMetrics(step=1000, path="/ckpt-1000")
        selector.add_checkpoint(checkpoint)
        assert len(selector.checkpoints) == 1

    def test_select_by_reward(self, selector_with_checkpoints):
        result = selector_with_checkpoints.select_best(SelectionCriteria.HIGHEST_REWARD)
        assert result.recommended_checkpoint.step == 2000  # Highest reward
        assert result.criteria_used == "highest_reward"

    def test_select_by_benchmark(self, selector_with_checkpoints):
        result = selector_with_checkpoints.select_best(SelectionCriteria.HIGHEST_BENCHMARK)
        assert result.recommended_checkpoint.step == 3000  # Highest benchmark
        assert result.criteria_used == "highest_benchmark"

    def test_select_by_kl(self, selector_with_checkpoints):
        result = selector_with_checkpoints.select_best(SelectionCriteria.LOWEST_KL)
        assert result.recommended_checkpoint.step == 1000  # Lowest KL
        assert result.criteria_used == "lowest_kl"

    def test_select_balanced(self, selector_with_checkpoints):
        result = selector_with_checkpoints.select_best(SelectionCriteria.BALANCED)
        assert result.criteria_used == "balanced"
        assert len(result.reasoning) > 0

    def test_select_custom_formula(self, selector_with_checkpoints):
        result = selector_with_checkpoints.select_best(
            SelectionCriteria.CUSTOM,
            custom_formula="gsm8k + math - kl * 100"
        )
        assert result.criteria_used == "custom"

    def test_select_empty_raises_error(self):
        selector = CheckpointSelector()
        with pytest.raises(ValueError):
            selector.select_best(SelectionCriteria.BALANCED)

    def test_get_checkpoint_timeline(self, selector_with_checkpoints):
        timeline = selector_with_checkpoints.get_checkpoint_timeline()
        assert len(timeline) == 3
        assert timeline[0]["step"] == 1000  # Sorted by step
        assert timeline[2]["step"] == 3000

    def test_detect_overfitting(self, selector_with_checkpoints):
        warnings = selector_with_checkpoints.detect_overfitting(window_size=2)
        # Should not detect overfitting with current data
        assert isinstance(warnings, list)

    def test_detect_overfitting_kl_increase(self):
        selector = CheckpointSelector()
        # Add checkpoints with increasing KL
        for i, kl in enumerate([0.05, 0.08, 0.12, 0.18, 0.25]):
            selector.add_checkpoint(CheckpointMetrics(
                step=1000 * (i + 1),
                path=f"/ckpt-{1000 * (i + 1)}",
                kl_divergence=kl,
                gsm8k_accuracy=50.0,
            ))
        warnings = selector.detect_overfitting(window_size=3)
        # Should detect KL divergence warning
        kl_warnings = [w for w in warnings if w["type"] == "kl_divergence_high"]
        assert len(kl_warnings) > 0

    def test_extract_step(self, selector_with_checkpoints):
        assert selector_with_checkpoints._extract_step("ckpt-1000") == 1000
        assert selector_with_checkpoints._extract_step("step_2000") == 2000
        assert selector_with_checkpoints._extract_step("checkpoint-3000") == 3000
        assert selector_with_checkpoints._extract_step("model-4000") == 4000
        assert selector_with_checkpoints._extract_step("no_number") is None


class TestCheckpointRecommendation:
    """Test CheckpointRecommendation class"""

    def test_recommendation_attributes(self):
        checkpoint = CheckpointMetrics(step=1000, path="/ckpt-1000")
        recommendation = CheckpointRecommendation(
            recommended_checkpoint=checkpoint,
            all_checkpoints=[checkpoint],
            criteria_used="balanced",
            score=85.0,
            reasoning="Best balanced score",
            alternatives=[],
        )
        assert recommendation.recommended_checkpoint.step == 1000
        assert recommendation.score == 85.0
        assert recommendation.criteria_used == "balanced"


class TestSelectBestCheckpointFunction:
    """Test the convenience function"""

    def test_select_best_checkpoint_no_checkpoints(self):
        result = select_best_checkpoint("/nonexistent/path")
        assert not result["success"]
        assert "No checkpoints found" in result["message"]

    def test_select_best_checkpoint_all_criteria(self):
        for criteria in ["balanced", "highest_reward", "highest_benchmark", "lowest_kl"]:
            # These will fail due to no actual checkpoints, but tests the API
            result = select_best_checkpoint(
                "/nonexistent/path",
                criteria=criteria,
            )
            assert "success" in result


class TestSelectionCriteriaEnum:
    """Test SelectionCriteria enum values"""

    def test_all_criteria_exist(self):
        assert SelectionCriteria.HIGHEST_REWARD.value == "highest_reward"
        assert SelectionCriteria.HIGHEST_BENCHMARK.value == "highest_benchmark"
        assert SelectionCriteria.LOWEST_KL.value == "lowest_kl"
        assert SelectionCriteria.BALANCED.value == "balanced"
        assert SelectionCriteria.CUSTOM.value == "custom"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
