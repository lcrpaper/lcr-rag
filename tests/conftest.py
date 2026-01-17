"""
Pytest Configuration and Shared Fixtures

This module provides shared fixtures and configuration for the LCR test suite.
"""

import pytest
import torch
import json
import tempfile
from pathlib import Path
from typing import Dict, List


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests if no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def sample_query() -> str:
    """Sample query for testing."""
    return "What is the population of Tokyo?"


@pytest.fixture
def sample_documents() -> List[Dict]:
    """Sample documents with numerical conflict."""
    return [
        {
            "doc_id": "doc_001",
            "text": "Tokyo has a population of 13.96 million people as of 2020.",
            "source": "wikipedia",
        },
        {
            "doc_id": "doc_002",
            "text": "The Greater Tokyo Area has over 37 million residents.",
            "source": "un_stats",
        },
    ]


@pytest.fixture
def sample_conflict_example() -> Dict:
    """Complete sample example with conflict."""
    return {
        "id": "test_001",
        "query": "What is the population of Tokyo?",
        "documents": [
            {"text": "Tokyo has 13.96 million people.", "source": "wiki"},
            {"text": "Greater Tokyo has 37 million residents.", "source": "stats"},
        ],
        "gold_answer": "13.96 million (city) or 37 million (metro)",
        "conflict_type": "L2_numerical",
        "conflict_level": "L2",
    }


@pytest.fixture
def no_conflict_example() -> Dict:
    """Sample example without conflict."""
    return {
        "id": "test_002",
        "query": "When was SpaceX founded?",
        "documents": [
            {"text": "SpaceX was founded in 2002.", "source": "wiki"},
            {"text": "The company was established in May 2002.", "source": "news"},
        ],
        "gold_answer": "2002",
        "conflict_type": None,
    }


@pytest.fixture
def sample_hidden_states() -> torch.Tensor:
    """Sample hidden states tensor (batch=4, hidden=4096)."""
    torch.manual_seed(42)
    return torch.randn(4, 4096)


@pytest.fixture
def sample_hidden_states_gpu(sample_hidden_states) -> torch.Tensor:
    """Sample hidden states on GPU if available."""
    if torch.cuda.is_available():
        return sample_hidden_states.cuda()
    return sample_hidden_states


@pytest.fixture
def batch_hidden_states() -> torch.Tensor:
    """Larger batch of hidden states for batch testing."""
    torch.manual_seed(42)
    return torch.randn(32, 4096)


@pytest.fixture
def detector_config() -> Dict:
    """Configuration for ConflictDetector."""
    return {
        "hidden_dim": 4096,
        "intermediate_dim": 512,
        "dropout": 0.1,
    }


@pytest.fixture
def refinement_config() -> Dict:
    """Configuration for RefinementModule."""
    return {
        "hidden_dim": 4096,
        "bottleneck_dim": 732,
        "num_iterations": 3,
        "alpha": 0.3,
    }


@pytest.fixture
def small_detector_config() -> Dict:
    """Smaller config for fast testing."""
    return {
        "hidden_dim": 256,
        "intermediate_dim": 64,
        "dropout": 0.1,
    }


@pytest.fixture
def small_refinement_config() -> Dict:
    """Smaller config for fast testing."""
    return {
        "hidden_dim": 256,
        "bottleneck_dim": 64,
        "num_iterations": 2,
        "alpha": 0.3,
    }


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_jsonl_file(temp_dir, sample_conflict_example, no_conflict_example) -> Path:
    """Create a temporary JSONL file with test data."""
    filepath = temp_dir / "test_data.jsonl"
    with open(filepath, 'w') as f:
        f.write(json.dumps(sample_conflict_example) + '\n')
        f.write(json.dumps(no_conflict_example) + '\n')
    return filepath


@pytest.fixture
def sample_checkpoint_dir(temp_dir) -> Path:
    """Create a mock checkpoint directory structure."""
    checkpoint_dir = temp_dir / "checkpoints"
    (checkpoint_dir / "detector").mkdir(parents=True)
    (checkpoint_dir / "refinement").mkdir(parents=True)

    torch.save({"state_dict": {}}, checkpoint_dir / "detector" / "model.pt")
    torch.save({"state_dict": {}}, checkpoint_dir / "refinement" / "model.pt")

    return checkpoint_dir


@pytest.fixture
def paper_expected_results() -> Dict:
    """Expected results from the paper for regression testing."""
    return {
        "detector_f1": 0.871,
        "detector_f1_tolerance": 0.005,
        "l1_accuracy": 0.724,
        "l2_accuracy": 0.701,
        "l3_accuracy": 0.597,
        "l4_accuracy": 0.578,
        "accuracy_tolerance": 0.01,
        "classifier_macro_f1": 0.888,
        "classifier_f1_tolerance": 0.005,
    }


def assert_tensor_close(actual: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-4):
    """Assert two tensors are close within tolerance."""
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), \
        f"Tensor values differ beyond tolerance"


def assert_probability_valid(prob: torch.Tensor):
    """Assert tensor contains valid probabilities."""
    assert (prob >= 0).all(), "Probabilities must be >= 0"
    assert (prob <= 1).all(), "Probabilities must be <= 1"
