"""
Unit Tests for ConflictDetector

Tests the conflict detection module in isolation.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.conflict_detector import ConflictDetector


class TestConflictDetectorInit:
    """Tests for ConflictDetector initialization."""

    def test_default_initialization(self, detector_config):
        """Test detector initializes with default config."""
        detector = ConflictDetector(**detector_config)
        assert detector is not None

    def test_parameter_count(self, detector_config):
        """Test detector has expected parameter count."""
        detector = ConflictDetector(**detector_config)
        param_count = sum(p.numel() for p in detector.parameters())

        expected_params = 2_098_177
        assert param_count == expected_params, \
            f"Expected {expected_params} params, got {param_count}"

    def test_small_config(self, small_detector_config):
        """Test detector works with smaller config."""
        detector = ConflictDetector(**small_detector_config)
        assert detector is not None

    def test_invalid_hidden_dim(self):
        """Test detector rejects invalid hidden dimension."""
        with pytest.raises((ValueError, AssertionError)):
            ConflictDetector(hidden_dim=0, intermediate_dim=512)


class TestConflictDetectorForward:
    """Tests for ConflictDetector forward pass."""

    def test_forward_shape(self, small_detector_config):
        """Test output shape is correct."""
        detector = ConflictDetector(**small_detector_config)
        x = torch.randn(8, small_detector_config["hidden_dim"])

        output = detector(x)

        assert output.shape == (8, 1), f"Expected (8, 1), got {output.shape}"

    def test_forward_probability_range(self, small_detector_config):
        """Test output is valid probability."""
        detector = ConflictDetector(**small_detector_config)
        x = torch.randn(8, small_detector_config["hidden_dim"])

        output = detector(x)

        assert (output >= 0).all(), "Output should be >= 0"
        assert (output <= 1).all(), "Output should be <= 1"

    def test_forward_batch_sizes(self, small_detector_config):
        """Test forward works with various batch sizes."""
        detector = ConflictDetector(**small_detector_config)

        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, small_detector_config["hidden_dim"])
            output = detector(x)
            assert output.shape == (batch_size, 1)

    def test_forward_deterministic(self, small_detector_config):
        """Test forward pass is deterministic in eval mode."""
        detector = ConflictDetector(**small_detector_config)
        detector.eval()

        torch.manual_seed(42)
        x = torch.randn(4, small_detector_config["hidden_dim"])

        output1 = detector(x)
        output2 = detector(x)

        assert torch.allclose(output1, output2), "Eval mode should be deterministic"

    @pytest.mark.gpu
    def test_forward_gpu(self, small_detector_config):
        """Test forward pass on GPU."""
        detector = ConflictDetector(**small_detector_config).cuda()
        x = torch.randn(8, small_detector_config["hidden_dim"]).cuda()

        output = detector(x)

        assert output.device.type == "cuda"
        assert output.shape == (8, 1)


class TestConflictDetectorGradients:
    """Tests for gradient computation."""

    def test_backward_pass(self, small_detector_config):
        """Test backward pass computes gradients."""
        detector = ConflictDetector(**small_detector_config)
        x = torch.randn(4, small_detector_config["hidden_dim"], requires_grad=True)

        output = detector(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None, "Input should have gradients"
        for param in detector.parameters():
            assert param.grad is not None, "Parameters should have gradients"

    def test_gradient_flow(self, small_detector_config):
        """Test gradients flow through all layers."""
        detector = ConflictDetector(**small_detector_config)
        x = torch.randn(4, small_detector_config["hidden_dim"])

        output = detector(x)
        loss = output.sum()
        loss.backward()

        for name, param in detector.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestConflictDetectorSaveLoad:
    """Tests for saving and loading detector."""

    def test_save_load_state_dict(self, small_detector_config, temp_dir):
        """Test saving and loading state dict."""
        detector1 = ConflictDetector(**small_detector_config)

        save_path = temp_dir / "detector.pt"
        torch.save(detector1.state_dict(), save_path)

        detector2 = ConflictDetector(**small_detector_config)
        detector2.load_state_dict(torch.load(save_path))

        for (n1, p1), (n2, p2) in zip(
            detector1.named_parameters(), detector2.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Parameter {n1} differs after load"

    def test_checkpoint_format(self, small_detector_config, temp_dir):
        """Test checkpoint contains expected keys."""
        detector = ConflictDetector(**small_detector_config)

        state_dict = detector.state_dict()

        expected_keys = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        actual_keys = set(state_dict.keys())

        assert expected_keys.issubset(actual_keys), \
            f"Missing keys: {expected_keys - actual_keys}"
