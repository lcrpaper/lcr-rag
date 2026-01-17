"""
Unit Tests for RefinementModule

Tests the latent refinement module in isolation.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.refinement_module import RefinementModule


class TestRefinementModuleInit:
    """Tests for RefinementModule initialization."""

    def test_default_initialization(self, refinement_config):
        """Test module initializes with default config."""
        module = RefinementModule(**refinement_config)
        assert module is not None

    def test_parameter_count(self, refinement_config):
        """Test module has expected parameter count."""
        module = RefinementModule(**refinement_config)
        param_count = sum(p.numel() for p in module.parameters())

        expected_params = 5_996_544
        assert param_count == expected_params, \
            f"Expected {expected_params} params, got {param_count}"

    def test_alpha_range(self):
        """Test alpha must be in valid range."""
        module = RefinementModule(hidden_dim=256, bottleneck_dim=64, alpha=0.3)
        assert module.alpha == 0.3

        module = RefinementModule(hidden_dim=256, bottleneck_dim=64, alpha=0.0)
        assert module.alpha == 0.0

    def test_iterations_positive(self):
        """Test iterations must be positive."""
        module = RefinementModule(
            hidden_dim=256, bottleneck_dim=64, num_iterations=5
        )
        assert module.num_iterations == 5


class TestRefinementModuleForward:
    """Tests for RefinementModule forward pass."""

    def test_forward_shape_preserved(self, small_refinement_config):
        """Test output shape matches input shape."""
        module = RefinementModule(**small_refinement_config)
        x = torch.randn(8, small_refinement_config["hidden_dim"])

        output = module(x)

        assert output.shape == x.shape, \
            f"Expected {x.shape}, got {output.shape}"

    def test_forward_different_batch_sizes(self, small_refinement_config):
        """Test forward works with various batch sizes."""
        module = RefinementModule(**small_refinement_config)

        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, small_refinement_config["hidden_dim"])
            output = module(x)
            assert output.shape == x.shape

    def test_alpha_zero_returns_input(self, small_refinement_config):
        """Test alpha=0 returns unchanged input."""
        config = {**small_refinement_config, "alpha": 0.0}
        module = RefinementModule(**config)
        module.eval()

        x = torch.randn(4, config["hidden_dim"])
        output = module(x)

        assert torch.allclose(output, x, atol=1e-6), \
            "Alpha=0 should return input unchanged"

    def test_iterative_refinement(self, small_refinement_config):
        """Test multiple iterations produce different outputs."""
        module = RefinementModule(**small_refinement_config)
        module.eval()

        x = torch.randn(4, small_refinement_config["hidden_dim"])

        config_1iter = {**small_refinement_config, "num_iterations": 1}
        module_1 = RefinementModule(**config_1iter)
        module_1.load_state_dict(module.state_dict())
        out_1 = module_1(x)

        out_3 = module(x)

        assert not torch.allclose(out_1, out_3), \
            "Different iteration counts should produce different outputs"

    @pytest.mark.gpu
    def test_forward_gpu(self, small_refinement_config):
        """Test forward pass on GPU."""
        module = RefinementModule(**small_refinement_config).cuda()
        x = torch.randn(8, small_refinement_config["hidden_dim"]).cuda()

        output = module(x)

        assert output.device.type == "cuda"
        assert output.shape == x.shape


class TestRefinementModuleGradients:
    """Tests for gradient computation."""

    def test_backward_pass(self, small_refinement_config):
        """Test backward pass computes gradients."""
        module = RefinementModule(**small_refinement_config)
        x = torch.randn(4, small_refinement_config["hidden_dim"], requires_grad=True)

        output = module(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None, "Input should have gradients"
        for param in module.parameters():
            assert param.grad is not None, "Parameters should have gradients"

    def test_no_nan_gradients(self, small_refinement_config):
        """Test gradients don't contain NaN."""
        module = RefinementModule(**small_refinement_config)
        x = torch.randn(4, small_refinement_config["hidden_dim"])

        output = module(x)
        loss = output.sum()
        loss.backward()

        for name, param in module.named_parameters():
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestRefinementModuleDeterminism:
    """Tests for deterministic behavior."""

    def test_eval_deterministic(self, small_refinement_config):
        """Test eval mode is deterministic."""
        module = RefinementModule(**small_refinement_config)
        module.eval()

        x = torch.randn(4, small_refinement_config["hidden_dim"])

        out1 = module(x)
        out2 = module(x)

        assert torch.allclose(out1, out2), "Eval should be deterministic"

    def test_seed_reproducibility(self, small_refinement_config):
        """Test results are reproducible with same seed."""
        torch.manual_seed(42)
        module1 = RefinementModule(**small_refinement_config)

        torch.manual_seed(42)
        module2 = RefinementModule(**small_refinement_config)

        x = torch.randn(4, small_refinement_config["hidden_dim"])

        module1.eval()
        module2.eval()

        out1 = module1(x)
        out2 = module2(x)

        assert torch.allclose(out1, out2), "Same seed should give same results"


class TestRefinementModuleSaveLoad:
    """Tests for saving and loading."""

    def test_save_load_state_dict(self, small_refinement_config, temp_dir):
        """Test saving and loading state dict."""
        module1 = RefinementModule(**small_refinement_config)

        save_path = temp_dir / "refinement.pt"
        torch.save(module1.state_dict(), save_path)

        module2 = RefinementModule(**small_refinement_config)
        module2.load_state_dict(torch.load(save_path))

        for (n1, p1), (n2, p2) in zip(
            module1.named_parameters(), module2.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Parameter {n1} differs"

    def test_checkpoint_keys(self, small_refinement_config):
        """Test checkpoint contains expected keys."""
        module = RefinementModule(**small_refinement_config)
        state_dict = module.state_dict()

        expected_keys = {"W_up.weight", "W_down.weight"}
        actual_keys = set(state_dict.keys())

        assert expected_keys.issubset(actual_keys), \
            f"Missing keys: {expected_keys - actual_keys}"
