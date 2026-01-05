"""
Tests for verl Adapter Module
"""

import pytest
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from training_platform.core.verl_adapter import (
    VerlAlgorithm,
    VerlTrainingConfig,
    VerlJobRunner,
    VerlMetricsCollector,
    create_verl_training_config,
    get_verl_example_scripts,
)


class TestVerlAlgorithm:
    """Test VerlAlgorithm enum"""

    def test_all_algorithms_exist(self):
        assert VerlAlgorithm.SFT.value == "sft"
        assert VerlAlgorithm.PPO.value == "ppo"
        assert VerlAlgorithm.GRPO.value == "grpo"
        assert VerlAlgorithm.DPO.value == "dpo"
        assert VerlAlgorithm.GSPO.value == "gspo"
        assert VerlAlgorithm.DAPO.value == "dapo"
        assert VerlAlgorithm.REMAX.value == "remax"
        assert VerlAlgorithm.RLOO.value == "rloo"


class TestVerlTrainingConfig:
    """Test VerlTrainingConfig class"""

    @pytest.fixture
    def sample_config(self):
        return VerlTrainingConfig(
            model_path="/path/to/model",
            algorithm=VerlAlgorithm.GRPO,
            train_data_path="/path/to/data",
            num_epochs=3,
            learning_rate=1e-6,
            batch_size=256,
            num_gpus=8,
        )

    def test_default_values(self):
        config = VerlTrainingConfig(model_path="/model")
        assert config.algorithm == VerlAlgorithm.GRPO
        assert config.num_epochs == 3
        assert config.learning_rate == 1e-6
        assert config.lora_enabled is False

    def test_to_verl_config(self, sample_config):
        verl_config = sample_config.to_verl_config()
        assert "model" in verl_config
        assert verl_config["model"]["path"] == "/path/to/model"
        assert "actor" in verl_config
        assert "rollout" in verl_config
        assert "algorithm" in verl_config

    def test_to_verl_config_ppo(self):
        config = VerlTrainingConfig(
            model_path="/model",
            algorithm=VerlAlgorithm.PPO,
        )
        verl_config = config.to_verl_config()
        # PPO should include critic config
        assert "critic" in verl_config

    def test_to_verl_config_with_lora(self):
        config = VerlTrainingConfig(
            model_path="/model",
            lora_enabled=True,
            lora_rank=16,
            lora_alpha=32,
        )
        verl_config = config.to_verl_config()
        assert "lora" in verl_config["actor"]
        assert verl_config["actor"]["lora"]["rank"] == 16

    def test_to_yaml(self, sample_config):
        yaml_str = sample_config.to_yaml()
        assert "model:" in yaml_str
        assert "actor:" in yaml_str


class TestVerlJobRunner:
    """Test VerlJobRunner class"""

    @pytest.fixture
    def runner(self):
        return VerlJobRunner(verl_path=None)

    def test_init_without_verl(self, runner):
        # Should handle missing verl gracefully
        assert runner.verl_path is None

    def test_run_training_without_verl(self, runner):
        config = VerlTrainingConfig(
            model_path="/model",
            train_data_path="/data",
        )
        result = runner.run_training(config)
        assert not result["success"]
        # Accepts either message format
        assert "verl" in result["message"].lower()

    def test_get_trainer_path_without_verl(self, runner):
        path = runner.get_trainer_path(VerlAlgorithm.GRPO)
        assert path is None

    def test_parse_metrics(self, runner):
        line = "step=1000 reward=0.75 kl=0.05 policy_loss=0.5"
        metrics = runner._parse_metrics(line)
        assert metrics["step"] == 1000
        assert metrics["reward"] == 0.75
        assert metrics["kl"] == 0.05
        assert metrics["policy_loss"] == 0.5


class TestVerlMetricsCollector:
    """Test VerlMetricsCollector class"""

    @pytest.fixture
    def collector(self):
        return VerlMetricsCollector()

    def test_add_metrics(self, collector):
        collector.add_metrics({"step": 100, "reward": 0.5})
        assert len(collector.metrics_history) == 1

    def test_get_latest(self, collector):
        collector.add_metrics({"step": 100, "reward": 0.5})
        collector.add_metrics({"step": 200, "reward": 0.7})
        latest = collector.get_latest()
        assert latest["step"] == 200

    def test_get_latest_empty(self, collector):
        assert collector.get_latest() is None

    def test_get_history(self, collector):
        for i in range(5):
            collector.add_metrics({"step": i * 100, "reward": 0.1 * i})

        history = collector.get_history(start_step=200, end_step=400)
        assert len(history) == 3  # steps 200, 300, 400

    def test_get_summary(self, collector):
        collector.add_metrics({"step": 100, "reward": 0.5, "kl": 0.05, "policy_loss": 0.3})
        collector.add_metrics({"step": 200, "reward": 0.7, "kl": 0.08, "policy_loss": 0.2})

        summary = collector.get_summary()
        assert summary["total_steps"] == 2
        assert "reward" in summary
        assert "mean" in summary["reward"]

    def test_get_summary_empty(self, collector):
        summary = collector.get_summary()
        assert summary == {}


class TestCreateVerlTrainingConfig:
    """Test convenience function"""

    def test_create_basic_config(self):
        config = create_verl_training_config(
            model_path="/model",
            algorithm="grpo",
        )
        assert config.model_path == "/model"
        assert config.algorithm == VerlAlgorithm.GRPO

    def test_create_config_with_lora(self):
        config = create_verl_training_config(
            model_path="/model",
            algorithm="grpo",
            lora_enabled=True,
        )
        assert config.lora_enabled is True

    def test_create_config_all_algorithms(self):
        for algo in ["sft", "ppo", "grpo", "dpo", "gspo"]:
            config = create_verl_training_config(
                model_path="/model",
                algorithm=algo,
            )
            assert config.algorithm.value == algo


class TestGetVerlExampleScripts:
    """Test example scripts discovery"""

    def test_get_example_scripts(self):
        # This will return empty dict if verl is not installed
        scripts = get_verl_example_scripts()
        assert isinstance(scripts, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
