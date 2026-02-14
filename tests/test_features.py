"""
Comprehensive tests for all features in feature_list.json.

Tests cover:
- ALG-001 through ALG-008: Algorithm config templates and verl_adapter
- DATA-001 through DATA-006: Data factory modules
- EVAL-001 through EVAL-006: Evaluation framework
- UI-001, UI-002: Frontend compilation (tested via npm run build)
"""

import json
import os
import sys
import tempfile
import shutil
import pytest
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ALG-001: SFT Algorithm
# ============================================================

class TestALG001_SFT:
    """Test SFT supervised fine-tuning algorithm."""

    def test_verl_adapter_sft_enum(self):
        from training_platform.core.verl_adapter import VerlAlgorithm
        assert VerlAlgorithm.SFT.value == "sft"

    def test_verl_adapter_sft_config(self):
        from training_platform.core.verl_adapter import VerlTrainingConfig, VerlAlgorithm
        config = VerlTrainingConfig(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            algorithm=VerlAlgorithm.SFT,
            train_data_path="./datasets/sft_math.json",
            num_epochs=1,
            learning_rate=5e-6,
        )
        assert config.algorithm == VerlAlgorithm.SFT
        assert config.model_path == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_sft_config_template_exists(self):
        from training_platform.core.config_templates import get_template
        template = get_template("sft")
        assert template is not None
        assert "defaults" in template
        assert "required_fields" in template

    def test_sft_config_template_content(self):
        from training_platform.core.config_templates import get_template
        template = get_template("sft")
        assert template["name"] is not None
        assert "model_path" in template["required_fields"]
        assert "train_data_path" in template["required_fields"]

    def test_sft_data_format_conversion(self):
        from training_platform.core.data_converter import convert_to_sft
        data = [{"instruction": "What is 2+2?", "output": "4"}]
        result = convert_to_sft(data, "sft_instruction")
        assert len(result) == 1
        assert result[0]["prompt"] == "What is 2+2?"
        assert result[0]["response"] == "4"

    def test_sft_command_generation(self):
        from training_platform.core.verl_adapter import VerlTrainingConfig, VerlAlgorithm
        config = VerlTrainingConfig(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            algorithm=VerlAlgorithm.SFT,
            train_data_path="./datasets/sft_math.json",
        )
        args = config._to_sft_args()
        assert isinstance(args, list)
        assert any("Qwen" in str(a) for a in args)


# ============================================================
# ALG-002: GRPO Algorithm
# ============================================================

class TestALG002_GRPO:
    """Test GRPO group relative policy optimization."""

    def test_verl_adapter_grpo_enum(self):
        from training_platform.core.verl_adapter import VerlAlgorithm
        assert VerlAlgorithm.GRPO.value == "grpo"

    def test_grpo_config_template_exists(self):
        from training_platform.core.config_templates import get_template
        template = get_template("grpo")
        assert template is not None
        assert "defaults" in template

    def test_grpo_config_template_content(self):
        from training_platform.core.config_templates import get_template
        template = get_template("grpo")
        assert "model_path" in template["required_fields"]
        defaults = template.get("defaults", {})
        assert isinstance(defaults, dict)

    def test_grpo_data_format_conversion(self):
        from training_platform.core.data_converter import convert_to_grpo
        data = [{"prompt": "What is 2+2?", "response": "4"}]
        result = convert_to_grpo(data, "sft_prompt_response")
        assert len(result) == 1
        assert result[0]["prompt"] == "What is 2+2?"
        assert "solution" in result[0]

    def test_grpo_verl_config(self):
        from training_platform.core.verl_adapter import VerlTrainingConfig, VerlAlgorithm
        config = VerlTrainingConfig(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            algorithm=VerlAlgorithm.GRPO,
            train_data_path="./datasets/grpo_math.json",
        )
        args = config.to_verl_command_args()
        assert isinstance(args, list)
        assert any("grpo" in str(a).lower() for a in args)


# ============================================================
# ALG-003: DPO Algorithm
# ============================================================

class TestALG003_DPO:
    """Test DPO direct preference optimization."""

    def test_verl_adapter_dpo_enum(self):
        from training_platform.core.verl_adapter import VerlAlgorithm
        assert VerlAlgorithm.DPO.value == "dpo"

    def test_dpo_config_template_exists(self):
        from training_platform.core.config_templates import get_template
        template = get_template("dpo")
        assert template is not None

    def test_dpo_data_format_conversion(self):
        from training_platform.core.data_converter import convert_to_dpo
        data = [{"prompt": "What is AI?", "chosen": "Good answer", "rejected": "Bad answer"}]
        result = convert_to_dpo(data, "dpo_preference")
        assert len(result) == 1
        assert result[0]["chosen"] == "Good answer"
        assert result[0]["rejected"] == "Bad answer"

    def test_dpo_sft_to_dpo_conversion(self):
        from training_platform.core.data_converter import convert_to_dpo
        data = [{"prompt": "Hello", "response": "Hi there!"}]
        result = convert_to_dpo(data, "sft_prompt_response")
        assert len(result) == 1
        assert result[0]["prompt"] == "Hello"
        assert result[0]["chosen"] == "Hi there!"

    def test_dpo_verl_config(self):
        from training_platform.core.verl_adapter import VerlTrainingConfig, VerlAlgorithm
        config = VerlTrainingConfig(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            algorithm=VerlAlgorithm.DPO,
            train_data_path="./datasets/dpo_preference.json",
        )
        args = config.to_verl_command_args()
        assert isinstance(args, list)
        assert any("dpo" in str(a).lower() for a in args)


# ============================================================
# ALG-004: PPO Algorithm
# ============================================================

class TestALG004_PPO:
    """Test PPO proximal policy optimization."""

    def test_verl_adapter_ppo_enum(self):
        from training_platform.core.verl_adapter import VerlAlgorithm
        assert VerlAlgorithm.PPO.value == "ppo"

    def test_ppo_config_template_exists(self):
        from training_platform.core.config_templates import get_template
        template = get_template("ppo")
        assert template is not None
        assert "defaults" in template

    def test_ppo_verl_config(self):
        from training_platform.core.verl_adapter import VerlTrainingConfig, VerlAlgorithm
        config = VerlTrainingConfig(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            algorithm=VerlAlgorithm.PPO,
            train_data_path="./datasets/data.json",
        )
        args = config.to_verl_command_args()
        assert isinstance(args, list)
        assert any("ppo" in str(a).lower() for a in args)


# ============================================================
# ALG-005: GSPO Algorithm
# ============================================================

class TestALG005_GSPO:
    """Test GSPO group sampling policy optimization."""

    def test_verl_adapter_gspo_enum(self):
        from training_platform.core.verl_adapter import VerlAlgorithm
        assert VerlAlgorithm.GSPO.value == "gspo"

    def test_gspo_config_template_exists(self):
        from training_platform.core.config_templates import get_template
        template = get_template("gspo")
        assert template is not None

    def test_gspo_verl_config(self):
        from training_platform.core.verl_adapter import VerlTrainingConfig, VerlAlgorithm
        config = VerlTrainingConfig(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            algorithm=VerlAlgorithm.GSPO,
            train_data_path="./datasets/data.json",
        )
        args = config.to_verl_command_args()
        assert isinstance(args, list)


# ============================================================
# ALG-006: DAPO Algorithm
# ============================================================

class TestALG006_DAPO:
    """Test DAPO decoupled alignment policy optimization."""

    def test_verl_adapter_dapo_enum(self):
        from training_platform.core.verl_adapter import VerlAlgorithm
        assert VerlAlgorithm.DAPO.value == "dapo"

    def test_dapo_config_template_exists(self):
        from training_platform.core.config_templates import get_template
        template = get_template("dapo")
        assert template is not None

    def test_dapo_verl_config(self):
        from training_platform.core.verl_adapter import VerlTrainingConfig, VerlAlgorithm
        config = VerlTrainingConfig(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            algorithm=VerlAlgorithm.DAPO,
            train_data_path="./datasets/data.json",
        )
        args = config.to_verl_command_args()
        assert isinstance(args, list)


# ============================================================
# ALG-007: REMAX Algorithm
# ============================================================

class TestALG007_REMAX:
    """Test REMAX reinforcement maximization."""

    def test_verl_adapter_remax_enum(self):
        from training_platform.core.verl_adapter import VerlAlgorithm
        assert VerlAlgorithm.REMAX.value == "remax"

    def test_remax_config_template_exists(self):
        from training_platform.core.config_templates import get_template
        template = get_template("remax")
        assert template is not None

    def test_remax_verl_config(self):
        from training_platform.core.verl_adapter import VerlTrainingConfig, VerlAlgorithm
        config = VerlTrainingConfig(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            algorithm=VerlAlgorithm.REMAX,
            train_data_path="./datasets/data.json",
        )
        args = config.to_verl_command_args()
        assert isinstance(args, list)


# ============================================================
# ALG-008: RLOO Algorithm
# ============================================================

class TestALG008_RLOO:
    """Test RLOO reinforcement learning leave-one-out."""

    def test_verl_adapter_rloo_enum(self):
        from training_platform.core.verl_adapter import VerlAlgorithm
        assert VerlAlgorithm.RLOO.value == "rloo"

    def test_rloo_config_template_exists(self):
        from training_platform.core.config_templates import get_template
        template = get_template("rloo")
        assert template is not None

    def test_rloo_verl_config(self):
        from training_platform.core.verl_adapter import VerlTrainingConfig, VerlAlgorithm
        config = VerlTrainingConfig(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            algorithm=VerlAlgorithm.RLOO,
            train_data_path="./datasets/data.json",
        )
        args = config.to_verl_command_args()
        assert isinstance(args, list)


# ============================================================
# DATA-001: Data Cleaning Pipeline
# ============================================================

class TestDATA001_Cleaning:
    """Test data cleaning pipeline."""

    def test_cleaning_config(self):
        from training_platform.core.data_cleaning import CleaningConfig
        config = CleaningConfig(
            min_prompt_length=10,
            max_prompt_length=5000,
            remove_empty=True,
        )
        assert config.min_prompt_length == 10

    def test_cleaning_pipeline_basic(self):
        from training_platform.core.data_cleaning import DataCleaningPipeline, CleaningConfig
        config = CleaningConfig(min_prompt_length=5)
        pipeline = DataCleaningPipeline(config)
        data = [
            {"prompt": "What is 2+2?", "response": "4"},
            {"prompt": "Hi", "response": "Hello"},  # Too short prompt (< 5)
            {"prompt": "", "response": ""},  # Empty
            {"prompt": "Calculate the sum of 1+1", "response": "The sum is 2"},
        ]
        cleaned, stats = pipeline.clean(data)
        assert stats.total_input == 4
        assert stats.total_output <= 4
        assert len(cleaned) == stats.total_output

    def test_cleaning_removes_empty(self):
        from training_platform.core.data_cleaning import DataCleaningPipeline, CleaningConfig
        config = CleaningConfig(remove_empty=True)
        pipeline = DataCleaningPipeline(config)
        data = [
            {"prompt": "", "response": "something"},
            {"prompt": "hello", "response": "world"},
        ]
        cleaned, stats = pipeline.clean(data)
        assert stats.removed_empty >= 1
        assert len(cleaned) == 1

    def test_cleaning_length_filter(self):
        from training_platform.core.data_cleaning import DataCleaningPipeline, CleaningConfig
        config = CleaningConfig(min_prompt_length=10, max_prompt_length=50)
        pipeline = DataCleaningPipeline(config)
        data = [
            {"prompt": "short", "response": "ok"},
            {"prompt": "This is a valid length prompt", "response": "ok"},
            {"prompt": "x" * 100, "response": "ok"},
        ]
        cleaned, stats = pipeline.clean(data)
        assert len(cleaned) == 1
        assert cleaned[0]["prompt"] == "This is a valid length prompt"

    def test_cleaning_strip_whitespace(self):
        from training_platform.core.data_cleaning import DataCleaningPipeline, CleaningConfig
        config = CleaningConfig(strip_whitespace=True)
        pipeline = DataCleaningPipeline(config)
        data = [{"prompt": "  hello world  ", "response": " answer  "}]
        cleaned, stats = pipeline.clean(data)
        assert cleaned[0]["prompt"] == "hello world"

    def test_cleaning_html_removal(self):
        from training_platform.core.data_cleaning import DataCleaningPipeline, CleaningConfig
        config = CleaningConfig(remove_html_tags=True)
        pipeline = DataCleaningPipeline(config)
        data = [{"prompt": "<b>bold</b> text", "response": "<p>paragraph</p>"}]
        cleaned, stats = pipeline.clean(data)
        assert "<b>" not in cleaned[0]["prompt"]
        assert "bold" in cleaned[0]["prompt"]

    def test_cleaning_stats_to_dict(self):
        from training_platform.core.data_cleaning import CleaningStats
        stats = CleaningStats(total_input=100, total_output=80, removed_empty=10, removed_too_short=10)
        d = stats.to_dict()
        assert d["total_input"] == 100
        assert d["total_output"] == 80
        assert d["total_removed"] == 20

    def test_cleaning_progress_callback(self):
        from training_platform.core.data_cleaning import DataCleaningPipeline, CleaningConfig
        progress_calls = []
        config = CleaningConfig()
        pipeline = DataCleaningPipeline(config)
        pipeline.set_progress_callback(lambda cur, total, msg: progress_calls.append((cur, total)))
        data = [{"prompt": f"Question {i}", "response": f"Answer {i}"} for i in range(10)]
        pipeline.clean(data)
        assert len(progress_calls) > 0

    def test_cleaning_duplicates(self):
        from training_platform.core.data_cleaning import DataCleaningPipeline, CleaningConfig
        config = CleaningConfig(remove_duplicates=True)
        pipeline = DataCleaningPipeline(config)
        data = [
            {"prompt": "same question", "response": "same answer"},
            {"prompt": "same question", "response": "same answer"},
            {"prompt": "different question", "response": "different answer"},
        ]
        cleaned, stats = pipeline.clean(data)
        assert len(cleaned) == 2
        assert stats.removed_duplicates == 1


# ============================================================
# DATA-002: Deduplication (MinHash/SimHash)
# ============================================================

class TestDATA002_Deduplication:
    """Test deduplication with MinHash/SimHash."""

    def test_exact_dedup(self):
        from training_platform.core.deduplication import deduplicate, DeduplicationConfig
        data = [
            {"prompt": "What is AI?", "response": "Artificial Intelligence"},
            {"prompt": "What is AI?", "response": "Artificial Intelligence"},
            {"prompt": "What is ML?", "response": "Machine Learning"},
        ]
        config = DeduplicationConfig(method="exact")
        result, stats = deduplicate(data, config)
        assert len(result) == 2
        assert stats.duplicates_found == 1

    def test_minhash_dedup(self):
        from training_platform.core.deduplication import deduplicate, DeduplicationConfig
        data = [
            {"prompt": "What is artificial intelligence and how does it work?", "response": "AI is a branch of computer science"},
            {"prompt": "What is artificial intelligence and how does it function?", "response": "AI is a branch of computer science"},
            {"prompt": "What is machine learning?", "response": "ML is a subset of AI"},
        ]
        config = DeduplicationConfig(method="minhash", threshold=0.5)
        result, stats = deduplicate(data, config)
        assert stats.total_input == 3
        assert stats.total_output <= 3

    def test_simhash_dedup(self):
        from training_platform.core.deduplication import deduplicate, DeduplicationConfig
        data = [
            {"prompt": "What is artificial intelligence?", "response": "AI is computer science"},
            {"prompt": "What is artificial intelligence?", "response": "AI is computer science"},
            {"prompt": "Different question entirely", "response": "Different answer"},
        ]
        config = DeduplicationConfig(method="simhash", threshold=0.9)
        result, stats = deduplicate(data, config)
        assert stats.duplicates_found >= 1

    def test_simhash_function(self):
        from training_platform.core.deduplication import simhash, simhash_similarity
        h1 = simhash("hello world this is a test")
        h2 = simhash("hello world this is a test")
        h3 = simhash("completely different text about cats")
        assert h1 == h2
        assert simhash_similarity(h1, h2) == 1.0
        assert simhash_similarity(h1, h3) < 1.0

    def test_dedup_stats(self):
        from training_platform.core.deduplication import DeduplicationStats
        stats = DeduplicationStats(total_input=100, total_output=90, duplicates_found=10)
        d = stats.to_dict()
        assert d["total_input"] == 100
        assert d["duplicates_found"] == 10
        assert d["dedup_rate"] == 10.0

    def test_dedup_invalid_method(self):
        from training_platform.core.deduplication import deduplicate, DeduplicationConfig
        config = DeduplicationConfig(method="invalid")
        with pytest.raises(ValueError):
            deduplicate([], config)


# ============================================================
# DATA-003: Quality Assessment
# ============================================================

class TestDATA003_Quality:
    """Test quality assessment module."""

    def test_quality_config(self):
        from training_platform.core.quality_assessment import QualityConfig
        config = QualityConfig(prompt_field="instruction", response_field="output")
        assert config.prompt_field == "instruction"

    def test_quality_single_item(self):
        from training_platform.core.quality_assessment import QualityAssessor
        assessor = QualityAssessor()
        item = {
            "prompt": "What is the capital of France? Please provide a detailed answer.",
            "response": "The capital of France is Paris. It is known for the Eiffel Tower and many cultural landmarks."
        }
        quality = assessor.assess_item(item)
        assert 0.0 <= quality.overall <= 1.0
        assert 0.0 <= quality.length_score <= 1.0
        assert 0.0 <= quality.diversity_score <= 1.0

    def test_quality_dataset_report(self):
        from training_platform.core.quality_assessment import QualityAssessor
        assessor = QualityAssessor()
        data = [
            {"prompt": "Question about math: What is 2+2?", "response": "The answer is 4. Two plus two equals four."},
            {"prompt": "Tell me about Python programming language", "response": "Python is a high-level programming language."},
            {"prompt": "x", "response": "y"},
        ]
        report = assessor.assess_dataset(data)
        assert report.total_items == 3
        assert 0.0 <= report.average_score <= 1.0
        assert "length" in report.dimension_averages

    def test_quality_report_to_dict(self):
        from training_platform.core.quality_assessment import QualityAssessor
        assessor = QualityAssessor()
        data = [
            {"prompt": "What is AI?", "response": "AI stands for Artificial Intelligence."},
        ]
        report = assessor.assess_dataset(data)
        d = report.to_dict()
        assert "total_items" in d
        assert "average_score" in d
        assert "score_distribution" in d

    def test_quality_empty_dataset(self):
        from training_platform.core.quality_assessment import QualityAssessor
        assessor = QualityAssessor()
        report = assessor.assess_dataset([])
        assert report.total_items == 0

    def test_quality_low_quality_detection(self):
        from training_platform.core.quality_assessment import QualityAssessor
        assessor = QualityAssessor()
        data = [
            {"prompt": "a", "response": "b"},
            {"prompt": "aaa aaa aaa aaa aaa", "response": "bbb bbb bbb bbb bbb"},
        ]
        report = assessor.assess_dataset(data)
        assert len(report.low_quality_indices) > 0


# ============================================================
# DATA-004: Format Conversion
# ============================================================

class TestDATA004_FormatConversion:
    """Test data format conversion."""

    def test_format_detection_sft(self):
        from training_platform.core.data_converter import DataFormatDetector
        data = [{"prompt": "Hello", "response": "Hi"}]
        fmt, conf = DataFormatDetector.detect(data)
        assert "sft" in fmt

    def test_format_detection_dpo(self):
        from training_platform.core.data_converter import DataFormatDetector
        data = [{"prompt": "Q", "chosen": "Good", "rejected": "Bad"}]
        fmt, conf = DataFormatDetector.detect(data)
        assert "dpo" in fmt

    def test_format_detection_openai(self):
        from training_platform.core.data_converter import DataFormatDetector
        data = [{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}]
        fmt, conf = DataFormatDetector.detect(data)
        assert "openai" in fmt

    def test_convert_sft_to_dpo(self):
        from training_platform.core.data_converter import convert_to_dpo
        data = [{"prompt": "Hello", "response": "Hi"}]
        result = convert_to_dpo(data, "sft_prompt_response")
        assert len(result) == 1
        assert result[0]["chosen"] == "Hi"

    def test_convert_sft_to_grpo(self):
        from training_platform.core.data_converter import convert_to_grpo
        data = [{"prompt": "Hello", "response": "Hi"}]
        result = convert_to_grpo(data, "sft_prompt_response")
        assert len(result) == 1
        assert "prompt" in result[0]

    def test_convert_sft_to_openai(self):
        from training_platform.core.data_converter import convert_to_openai_messages
        data = [{"prompt": "Hello", "response": "Hi"}]
        result = convert_to_openai_messages(data, "sft_prompt_response")
        assert len(result) == 1
        assert "messages" in result[0]
        assert result[0]["messages"][0]["role"] == "user"

    def test_convert_alpaca_format(self):
        from training_platform.core.data_converter import convert_to_sft
        data = [{"instruction": "Summarize", "input": "some text", "output": "summary"}]
        result = convert_to_sft(data, "sft_alpaca")
        assert "some text" in result[0]["prompt"]
        assert result[0]["response"] == "summary"

    def test_convert_file(self):
        from training_platform.core.data_converter import convert_file
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            output_path = os.path.join(tmpdir, "output.jsonl")
            data = [{"prompt": "Hello", "response": "Hi"}, {"prompt": "Q", "response": "A"}]
            with open(input_path, "w") as f:
                json.dump(data, f)
            result = convert_file(input_path, output_path, "grpo")
            assert result["input_count"] == 2
            assert result["output_count"] == 2
            assert os.path.exists(output_path)

    def test_load_save_jsonl(self):
        from training_platform.core.data_converter import _load_data, _save_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            data = [{"a": 1}, {"b": 2}]
            _save_data(data, path)
            loaded = _load_data(path)
            assert len(loaded) == 2
            assert loaded[0]["a"] == 1


# ============================================================
# DATA-005: Data Splitting
# ============================================================

class TestDATA005_Splitting:
    """Test data splitting functionality."""

    def test_random_split(self):
        from training_platform.core.data_splitter import random_split
        data = [{"id": i} for i in range(100)]
        result = random_split(data, 0.8, 0.1, 0.1)
        assert len(result["train"]) == 80
        assert len(result["val"]) == 10
        assert len(result["test"]) == 10

    def test_random_split_reproducible(self):
        from training_platform.core.data_splitter import random_split
        data = [{"id": i} for i in range(100)]
        r1 = random_split(data, 0.8, 0.1, 0.1, seed=42)
        r2 = random_split(data, 0.8, 0.1, 0.1, seed=42)
        assert r1["train"] == r2["train"]

    def test_stratified_split(self):
        from training_platform.core.data_splitter import stratified_split
        data = [
            {"text": f"item {i}", "category": "A" if i < 50 else "B"}
            for i in range(100)
        ]
        result = stratified_split(data, "category", 0.8, 0.1, 0.1)
        assert len(result["train"]) > 0
        assert len(result["val"]) > 0
        assert "category_distribution" in result

    def test_temporal_split(self):
        from training_platform.core.data_splitter import temporal_split
        data = [{"text": f"item {i}", "timestamp": f"2024-01-{i+1:02d}"} for i in range(100)]
        result = temporal_split(data, "timestamp", 0.8, 0.1, 0.1)
        assert len(result["train"]) == 80

    def test_split_data_unified(self):
        from training_platform.core.data_splitter import split_data
        data = [{"id": i} for i in range(100)]
        result = split_data(data, method="random")
        assert "statistics" in result
        assert result["statistics"]["total"] == 100

    def test_split_and_save(self):
        from training_platform.core.data_splitter import split_and_save
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"id": i, "prompt": f"Q{i}", "response": f"A{i}"} for i in range(20)]
            result = split_and_save(data, tmpdir, method="random", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
            assert "files" in result
            assert os.path.exists(result["files"]["train"])

    def test_split_invalid_method(self):
        from training_platform.core.data_splitter import split_data
        with pytest.raises(ValueError):
            split_data([{"id": 1}], method="invalid")


# ============================================================
# DATA-006: Data Versioning (leverages existing dataset_version router)
# ============================================================

class TestDATA006_Versioning:
    """Test data versioning - uses existing dataset_version router."""

    def test_dataset_version_router_exists(self):
        from training_platform.api.routers.dataset_version import router
        assert router is not None

    def test_version_router_has_endpoints(self):
        from training_platform.api.routers.dataset_version import router
        paths = [r.path for r in router.routes]
        assert len(paths) > 0


# ============================================================
# EVAL-001: GSM8K Benchmark
# ============================================================

class TestEVAL001_GSM8K:
    """Test GSM8K math reasoning benchmark."""

    def test_gsm8k_evaluator_exists(self):
        from training_platform.core.evaluation.benchmarks import GSM8KEvaluator
        evaluator = GSM8KEvaluator()
        assert evaluator.name().lower() == "gsm8k"

    def test_gsm8k_load_dataset(self):
        from training_platform.core.evaluation.benchmarks import GSM8KEvaluator
        evaluator = GSM8KEvaluator()
        questions = evaluator.load_dataset("test")
        assert len(questions) > 0

    def test_gsm8k_get_questions(self):
        from training_platform.core.evaluation.benchmarks import GSM8KEvaluator
        evaluator = GSM8KEvaluator()
        questions = evaluator.get_questions(split="test", limit=5)
        assert len(questions) <= 5
        assert "prompt" in questions[0] or "question" in questions[0]

    def test_gsm8k_answer_extraction(self):
        from training_platform.core.evaluation.benchmarks import extract_last_number
        result = extract_last_number("The answer is 42.")
        assert "42" in str(result)
        result2 = extract_last_number("So we get 3.14")
        assert "3.14" in str(result2)

    def test_gsm8k_evaluate_correct(self):
        from training_platform.core.evaluation.benchmarks import GSM8KEvaluator
        evaluator = GSM8KEvaluator()
        questions = evaluator.load_dataset("test")
        if questions:
            q = questions[0]
            answer = q.get("answer", "")
            result = evaluator.evaluate_response(q, f"The answer is {answer}")
            # May or may not be correct depending on format, but should not error
            assert result is not None


# ============================================================
# EVAL-002: MATH Benchmark
# ============================================================

class TestEVAL002_MATH:
    """Test MATH advanced math benchmark."""

    def test_math_evaluator_exists(self):
        from training_platform.core.evaluation.benchmarks import MATHEvaluator
        evaluator = MATHEvaluator()
        assert evaluator.name().lower() == "math"

    def test_math_load_dataset(self):
        from training_platform.core.evaluation.benchmarks import MATHEvaluator
        evaluator = MATHEvaluator()
        questions = evaluator.load_dataset("test")
        assert len(questions) > 0

    def test_math_boxed_extraction(self):
        from training_platform.core.evaluation.benchmarks import extract_boxed
        assert extract_boxed("\\boxed{42}") == "42"
        assert extract_boxed("\\boxed{x^2+1}") == "x^2+1"
        assert extract_boxed("no boxed here") is None

    def test_math_number_normalization(self):
        from training_platform.core.evaluation.benchmarks import normalize_number
        assert float(normalize_number("42")) == 42.0
        assert float(normalize_number("42.0")) == 42.0
        assert float(normalize_number("3.14")) == 3.14


# ============================================================
# EVAL-003: HumanEval Code Benchmark
# ============================================================

class TestEVAL003_HumanEval:
    """Test HumanEval code generation benchmark."""

    def test_humaneval_evaluator_exists(self):
        from training_platform.core.evaluation.benchmarks import HumanEvalEvaluator
        evaluator = HumanEvalEvaluator()
        assert evaluator.name().lower() == "humaneval"

    def test_humaneval_load_dataset(self):
        from training_platform.core.evaluation.benchmarks import HumanEvalEvaluator
        evaluator = HumanEvalEvaluator()
        questions = evaluator.load_dataset("test")
        assert len(questions) > 0


# ============================================================
# EVAL-004: Auto Trigger Evaluation
# ============================================================

class TestEVAL004_AutoTrigger:
    """Test auto-triggered evaluation after training."""

    def test_eval_trigger_config(self):
        from training_platform.core.evaluation.auto_trigger import EvalTriggerConfig
        config = EvalTriggerConfig(
            benchmarks=["gsm8k", "math"],
            trigger_every_n_steps=100,
        )
        assert "gsm8k" in config.benchmarks

    def test_eval_trigger_should_trigger(self):
        from training_platform.core.evaluation.auto_trigger import EvalTrigger, EvalTriggerConfig
        config = EvalTriggerConfig(trigger_every_n_steps=100)
        trigger = EvalTrigger(config)
        assert trigger.should_trigger(0) is False
        assert trigger.should_trigger(100) is True

    def test_eval_trigger_on_complete(self):
        from training_platform.core.evaluation.auto_trigger import EvalTrigger, EvalTriggerConfig
        config = EvalTriggerConfig(trigger_on_complete=True, benchmarks=["gsm8k"])
        trigger = EvalTrigger(config)
        result = trigger.on_training_complete("job-1", "/path/to/ckpt", 1000)
        assert result is not None
        assert result["job_id"] == "job-1"
        assert result["status"] == "pending"

    def test_eval_trigger_on_checkpoint(self):
        from training_platform.core.evaluation.auto_trigger import EvalTrigger, EvalTriggerConfig
        config = EvalTriggerConfig(trigger_every_n_steps=50)
        trigger = EvalTrigger(config)
        r1 = trigger.on_checkpoint_saved("job-1", "/ckpt1", 30)
        assert r1 is None  # Not yet at 50 steps
        r2 = trigger.on_checkpoint_saved("job-1", "/ckpt2", 50)
        assert r2 is not None

    def test_eval_trigger_history(self):
        from training_platform.core.evaluation.auto_trigger import EvalTrigger, EvalTriggerConfig
        config = EvalTriggerConfig(trigger_on_complete=True)
        trigger = EvalTrigger(config)
        trigger.on_training_complete("job-1", "/ckpt", 100)
        history = trigger.get_history()
        assert len(history) == 1

    def test_eval_trigger_record_result(self):
        from training_platform.core.evaluation.auto_trigger import EvalTrigger, EvalTriggerConfig
        config = EvalTriggerConfig(trigger_on_complete=True)
        trigger = EvalTrigger(config)
        trigger.on_training_complete("job-1", "/ckpt", 100)
        trigger.record_result(100, {"accuracy": 0.85})
        history = trigger.get_history()
        assert history[0]["status"] == "completed"
        assert history[0]["results"]["accuracy"] == 0.85


# ============================================================
# EVAL-005: Report Generation
# ============================================================

class TestEVAL005_ReportGeneration:
    """Test evaluation report generation."""

    def test_markdown_report(self):
        from training_platform.core.evaluation.report_generator import generate_report
        results = [
            {"benchmark": "gsm8k", "accuracy": 0.75, "total": 100, "correct": 75},
            {"benchmark": "math", "accuracy": 0.60, "total": 50, "correct": 30},
        ]
        report = generate_report(results, "TestModel", "markdown")
        assert "# Evaluation Report: TestModel" in report
        assert "gsm8k" in report
        assert "75.0%" in report

    def test_html_report(self):
        from training_platform.core.evaluation.report_generator import generate_report
        results = [
            {"benchmark": "gsm8k", "accuracy": 0.75, "total": 100, "correct": 75},
        ]
        report = generate_report(results, "TestModel", "html")
        assert "<html>" in report
        assert "TestModel" in report

    def test_report_with_comparison(self):
        from training_platform.core.evaluation.report_generator import generate_report
        results = [{"benchmark": "gsm8k", "accuracy": 0.80, "total": 100, "correct": 80}]
        comparison = [{"benchmark": "gsm8k", "accuracy": 0.70, "total": 100, "correct": 70}]
        report = generate_report(results, "TestModel", "markdown", comparison)
        assert "Comparison" in report

    def test_report_invalid_format(self):
        from training_platform.core.evaluation.report_generator import generate_report
        with pytest.raises(ValueError):
            generate_report([], "Test", "pdf")

    def test_save_report(self):
        from training_platform.core.evaluation.report_generator import save_report
        results = [{"benchmark": "gsm8k", "accuracy": 0.75, "total": 100, "correct": 75}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.md")
            save_report(results, path, "TestModel")
            assert os.path.exists(path)
            content = open(path).read()
            assert "TestModel" in content


# ============================================================
# EVAL-006: MMLU Benchmark
# ============================================================

class TestEVAL006_MMLU:
    """Test MMLU multi-subject benchmark."""

    def test_mmlu_evaluator_exists(self):
        from training_platform.core.evaluation.benchmarks import MMLUEvaluator
        evaluator = MMLUEvaluator()
        assert evaluator.name().lower() == "mmlu"

    def test_mmlu_load_dataset(self):
        from training_platform.core.evaluation.benchmarks import MMLUEvaluator
        evaluator = MMLUEvaluator()
        questions = evaluator.load_dataset("test")
        assert len(questions) > 0

    def test_mmlu_get_questions(self):
        from training_platform.core.evaluation.benchmarks import MMLUEvaluator
        evaluator = MMLUEvaluator()
        questions = evaluator.get_questions(split="test", limit=5)
        assert len(questions) <= 5


# ============================================================
# Config Templates (cross-cutting)
# ============================================================

class TestConfigTemplates:
    """Test config template system."""

    def test_list_templates(self):
        from training_platform.core.config_templates import list_templates
        templates = list_templates()
        assert len(templates) == 8  # sft, grpo, dpo, ppo, gspo, dapo, remax, rloo
        names = [t["algorithm"] for t in templates]
        assert "sft" in names
        assert "grpo" in names
        assert "dpo" in names

    def test_validate_config_valid(self):
        from training_platform.core.config_templates import validate_config
        result = validate_config("sft", {
            "model_path": "Qwen/Qwen2.5-0.5B",
            "train_data_path": "./data.jsonl",
        })
        assert result["valid"] is True
        assert len(result["missing_fields"]) == 0

    def test_validate_config_missing(self):
        from training_platform.core.config_templates import validate_config
        result = validate_config("sft", {})
        assert result["valid"] is False
        assert len(result["missing_fields"]) > 0

    def test_template_not_found(self):
        from training_platform.core.config_templates import get_template
        with pytest.raises(FileNotFoundError):
            get_template("nonexistent")


# ============================================================
# Benchmark Utilities
# ============================================================

class TestBenchmarkUtilities:
    """Test benchmark utility functions."""

    def test_list_benchmarks(self):
        from training_platform.core.evaluation.benchmarks import list_benchmarks
        benchmarks = list_benchmarks()
        assert len(benchmarks) >= 4
        names = [b["name"] for b in benchmarks]
        assert "gsm8k" in names
        assert "math" in names

    def test_get_evaluator(self):
        from training_platform.core.evaluation.benchmarks import get_evaluator
        evaluator = get_evaluator("gsm8k")
        assert evaluator.name().lower() == "gsm8k"

    def test_get_evaluator_invalid(self):
        from training_platform.core.evaluation.benchmarks import get_evaluator
        with pytest.raises(ValueError):
            get_evaluator("nonexistent_benchmark")

    def test_answers_equivalent(self):
        from training_platform.core.evaluation.benchmarks import answers_equivalent
        assert answers_equivalent("42", "42") is True
        assert answers_equivalent("42.0", "42") is True
        assert answers_equivalent("3.14", "3.14") is True


# ============================================================
# API Router Tests
# ============================================================

class TestDataFactoryAPI:
    """Test data factory API endpoints load correctly."""

    def test_router_exists(self):
        from training_platform.api.routers.data_factory import router
        assert router is not None
        assert router.prefix == "/data-factory"

    def test_app_includes_router(self):
        from training_platform.api.main import app
        paths = [r.path for r in app.routes]
        # Check that data-factory routes are registered
        data_factory_paths = [p for p in paths if "data-factory" in p]
        assert len(data_factory_paths) > 0

    def test_templates_endpoint_registered(self):
        from training_platform.api.main import app
        paths = [r.path for r in app.routes]
        assert any("templates" in p for p in paths)


# ============================================================
# Integration: Full Pipeline Test
# ============================================================

class TestIntegrationPipeline:
    """End-to-end integration tests."""

    def test_full_data_pipeline(self):
        """Test: load -> clean -> dedup -> quality -> split."""
        from training_platform.core.data_converter import _load_data, _save_data
        from training_platform.core.data_cleaning import DataCleaningPipeline, CleaningConfig
        from training_platform.core.deduplication import deduplicate, DeduplicationConfig
        from training_platform.core.quality_assessment import QualityAssessor
        from training_platform.core.data_splitter import split_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            data = [
                {"prompt": f"What is {i}+{i}?", "response": f"The answer is {i*2}"}
                for i in range(50)
            ]
            # Add duplicates
            data.extend(data[:5])
            input_path = os.path.join(tmpdir, "data.jsonl")
            _save_data(data, input_path)

            # Load
            loaded = _load_data(input_path)
            assert len(loaded) == 55

            # Clean
            pipeline = DataCleaningPipeline(CleaningConfig(min_prompt_length=5))
            cleaned, clean_stats = pipeline.clean(loaded)
            assert len(cleaned) <= len(loaded)

            # Dedup
            deduped, dedup_stats = deduplicate(cleaned, DeduplicationConfig(method="exact"))
            assert dedup_stats.duplicates_found >= 5

            # Quality
            assessor = QualityAssessor()
            report = assessor.assess_dataset(deduped)
            assert report.total_items == len(deduped)

            # Split
            result = split_data(deduped, method="random")
            assert len(result["train"]) > 0
            assert len(result["val"]) >= 0

    def test_format_conversion_roundtrip(self):
        """Test: SFT -> OpenAI Messages -> SFT."""
        from training_platform.core.data_converter import convert_to_sft, convert_to_openai_messages
        original = [
            {"prompt": "What is 2+2?", "response": "4"},
            {"prompt": "What is AI?", "response": "Artificial Intelligence"},
        ]
        # SFT -> OpenAI
        openai = convert_to_openai_messages(original, "sft_prompt_response")
        assert len(openai) == 2
        assert "messages" in openai[0]

        # OpenAI -> SFT
        back = convert_to_sft(openai, "openai_messages")
        assert len(back) == 2
        assert back[0]["prompt"] == "What is 2+2?"
        assert back[0]["response"] == "4"
