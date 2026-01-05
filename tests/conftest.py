"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_path():
    """Return the project root path"""
    return project_root


@pytest.fixture
def sample_model_sizes():
    """Return list of supported model sizes"""
    return ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]


@pytest.fixture
def sample_gpu_types():
    """Return list of supported GPU types"""
    return ["A100-40G", "A100-80G", "H100-80G", "H100-SXM", "A800-80G", "H800-80G", "RTX4090", "L40S"]


@pytest.fixture
def sample_training_types():
    """Return list of supported training types"""
    return ["sft", "ppo", "grpo", "dpo", "gspo"]
