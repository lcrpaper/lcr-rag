#!/usr/bin/env python3
"""
LCR Model Loading Infrastructure

Usage:
    from src.models.model_loader import LCRModelLoader

    loader = LCRModelLoader(
        checkpoint_dir="checkpoints/",
        strict_validation=True,
        calibration_check=True
    )

    detector = loader.load_detector()
    refinement = loader.load_refinement()
    classifier = loader.load_classifier()
"""

import os
import sys
import json
import hashlib
import warnings
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False
_NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.cuda as cuda
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import transformers
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    pass


SUPPORTED_PYTORCH_VERSIONS = {
    "2.1.0", "2.1.1", "2.1.2",
    "2.2.0", "2.2.1",
}

SUPPORTED_CUDA_VERSIONS = {
    "11.8", "12.0", "12.1", "12.2",
}

SUPPORTED_COMPUTE_CAPABILITIES = {
    "7.0",
    "7.5",
    "8.0",
    "8.6",
    "8.9",
    "9.0",
}

TRANSFORMERS_VERSION_RANGE = ("4.35.0", "4.40.0")

NUMPY_ABI_COMPATIBLE = {
    "1.24.0", "1.24.1", "1.24.2", "1.24.3",
    "1.25.0", "1.25.1", "1.25.2",
    "1.26.0", "1.26.1", "1.26.2", "1.26.3",
}

@dataclass
class EnvironmentReport:
    """Comprehensive environment compatibility report."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    platform: str = ""
    python_version: str = ""

    pytorch_version: str = ""
    pytorch_compatible: bool = False
    cuda_available: bool = False
    cuda_version: str = ""
    cuda_compatible: bool = False
    cudnn_version: str = ""

    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    compute_capabilities: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)

    transformers_version: str = ""
    transformers_compatible: bool = False
    numpy_version: str = ""
    numpy_compatible: bool = False

    fully_compatible: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class CalibrationReport:
    """Model calibration validation report."""
    checkpoint_path: str = ""
    expected_calibration: Dict[str, float] = field(default_factory=dict)
    observed_calibration: Dict[str, float] = field(default_factory=dict)
    calibration_drift: float = 0.0
    within_tolerance: bool = False
    validation_samples: int = 0


class EnvironmentValidationError(Exception):
    """Raised when environment validation fails."""
    pass


class CheckpointIntegrityError(Exception):
    """Raised when checkpoint integrity check fails."""
    pass


class CalibrationDriftError(Exception):
    """Raised when model calibration has drifted beyond tolerance."""
    pass


class HardwareCompatibilityError(Exception):
    """Raised when hardware is incompatible."""
    pass


def _parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    version_str = version_str.split("+")[0]
    return tuple(int(x) for x in version_str.split(".")[:3])


def _version_in_range(version: str, min_ver: str, max_ver: str) -> bool:
    """Check if version is within specified range."""
    try:
        v = _parse_version(version)
        v_min = _parse_version(min_ver)
        v_max = _parse_version(max_ver)
        return v_min <= v < v_max
    except Exception:
        return False


class EnvironmentValidator:
    """
    Validates execution environment for LCR model compatibility.

    This validator performs comprehensive checks to ensure the runtime
    environment matches the training environment, preventing silent
    numerical discrepancies that can affect reproducibility.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        self._report: Optional[EnvironmentReport] = None

    def validate(self) -> EnvironmentReport:
        """
        Perform full environment validation.

        Returns:
            EnvironmentReport with detailed compatibility information

        Raises:
            EnvironmentValidationError: If strict=True and critical issues found
        """
        report = EnvironmentReport()

        report.platform = platform.platform()
        report.python_version = platform.python_version()

        py_version = sys.version_info
        if py_version < (3, 10):
            report.errors.append(f"Python 3.10+ required, found {platform.python_version()}")

        self._validate_pytorch(report)

        self._validate_cuda(report)

        self._validate_gpu_hardware(report)

        self._validate_libraries(report)

        report.fully_compatible = (
            report.pytorch_compatible and
            report.cuda_compatible and
            report.transformers_compatible and
            report.numpy_compatible and
            len(report.errors) == 0
        )

        self._report = report

        if self.strict and not report.fully_compatible:
            error_msg = "Environment validation failed:\n" + "\n".join(report.errors)
            raise EnvironmentValidationError(error_msg)

        return report

    def _validate_pytorch(self, report: EnvironmentReport):
        """Validate PyTorch installation."""
        if not _TORCH_AVAILABLE:
            report.errors.append("PyTorch not installed")
            return

        report.pytorch_version = torch.__version__
        base_version = report.pytorch_version.split("+")[0]

        if base_version in SUPPORTED_PYTORCH_VERSIONS:
            report.pytorch_compatible = True
        else:
            report.warnings.append(
                f"PyTorch {report.pytorch_version} not in tested versions. "
                f"Tested versions: {SUPPORTED_PYTORCH_VERSIONS}"
            )
            report.pytorch_compatible = True

    def _validate_cuda(self, report: EnvironmentReport):
        """Validate CUDA installation."""
        if not _TORCH_AVAILABLE:
            return

        report.cuda_available = torch.cuda.is_available()

        if not report.cuda_available:
            report.errors.append(
                "CUDA not available. GPU required for model inference. "
                "CPU-only mode is not supported for production use."
            )
            return

        report.cuda_version = torch.version.cuda or "Unknown"

        if report.cuda_version in SUPPORTED_CUDA_VERSIONS:
            report.cuda_compatible = True
        else:
            report.warnings.append(
                f"CUDA {report.cuda_version} not in tested versions. "
                f"Tested versions: {SUPPORTED_CUDA_VERSIONS}"
            )
            report.cuda_compatible = True

        if torch.backends.cudnn.is_available():
            report.cudnn_version = str(torch.backends.cudnn.version())

    def _validate_gpu_hardware(self, report: EnvironmentReport):
        """Validate GPU hardware compatibility."""
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return

        report.gpu_count = torch.cuda.device_count()

        for i in range(report.gpu_count):
            props = torch.cuda.get_device_properties(i)
            report.gpu_names.append(props.name)
            report.gpu_memory_gb.append(props.total_memory / (1024**3))

            cc = f"{props.major}.{props.minor}"
            report.compute_capabilities.append(cc)

            if cc not in SUPPORTED_COMPUTE_CAPABILITIES:
                report.warnings.append(
                    f"GPU {i} ({props.name}) compute capability {cc} "
                    f"not in tested hardware. May affect numerical precision."
                )

        min_memory = min(report.gpu_memory_gb) if report.gpu_memory_gb else 0
        if min_memory < 8.0:
            report.errors.append(
                f"Minimum 8GB GPU memory required. Found {min_memory:.1f}GB. "
                f"Refinement module requires significant GPU memory."
            )

    def _validate_libraries(self, report: EnvironmentReport):
        """Validate library versions."""
        if _TRANSFORMERS_AVAILABLE:
            report.transformers_version = transformers.__version__
            if _version_in_range(
                report.transformers_version,
                TRANSFORMERS_VERSION_RANGE[0],
                TRANSFORMERS_VERSION_RANGE[1]
            ):
                report.transformers_compatible = True
            else:
                report.warnings.append(
                    f"Transformers {report.transformers_version} outside tested range "
                    f"{TRANSFORMERS_VERSION_RANGE}. Tokenization may differ."
                )
                report.transformers_compatible = True
        else:
            report.errors.append("Transformers library not installed")

        if _NUMPY_AVAILABLE:
            report.numpy_version = np.__version__
            base_version = report.numpy_version.split("+")[0]
            if base_version in NUMPY_ABI_COMPATIBLE or base_version.startswith("1.24") or base_version.startswith("1.25") or base_version.startswith("1.26"):
                report.numpy_compatible = True
            else:
                report.warnings.append(
                    f"NumPy {report.numpy_version} ABI may be incompatible. "
                    f"Tested with 1.24.x - 1.26.x series."
                )
                report.numpy_compatible = True
        else:
            report.errors.append("NumPy not installed")

    def print_report(self):
        """Print formatted validation report."""
        if self._report is None:
            self.validate()

        report = self._report

        print("\n" + "=" * 70)
        print("LCR ENVIRONMENT VALIDATION REPORT")
        print("=" * 70)
        print(f"Timestamp: {report.timestamp}")
        print(f"Platform: {report.platform}")
        print(f"Python: {report.python_version}")
        print()

        print("PyTorch:")
        print(f"  Version: {report.pytorch_version}")
        print(f"  Compatible: {'✓' if report.pytorch_compatible else '✗'}")
        print()

        print("CUDA:")
        print(f"  Available: {'✓' if report.cuda_available else '✗'}")
        print(f"  Version: {report.cuda_version}")
        print(f"  Compatible: {'✓' if report.cuda_compatible else '✗'}")
        print(f"  cuDNN: {report.cudnn_version}")
        print()

        print("GPU Hardware:")
        print(f"  Device Count: {report.gpu_count}")
        for i, (name, cc, mem) in enumerate(zip(
            report.gpu_names, report.compute_capabilities, report.gpu_memory_gb
        )):
            print(f"  [{i}] {name} (CC {cc}, {mem:.1f}GB)")
        print()

        print("Libraries:")
        print(f"  Transformers: {report.transformers_version} "
              f"({'✓' if report.transformers_compatible else '✗'})")
        print(f"  NumPy: {report.numpy_version} "
              f"({'✓' if report.numpy_compatible else '✗'})")
        print()

        if report.warnings:
            print("Warnings:")
            for w in report.warnings:
                print(f"  ⚠ {w}")
            print()

        if report.errors:
            print("Errors:")
            for e in report.errors:
                print(f"  ✗ {e}")
            print()

        status = "COMPATIBLE" if report.fully_compatible else "INCOMPATIBLE"
        print(f"Overall Status: {status}")
        print("=" * 70)


class LCRModelLoader:
    """
    Unified loader for all LCR model components.    
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        strict_validation: bool = True,
        calibration_check: bool = True,
        verify_integrity: bool = True,
        device: Optional[str] = None,
        precision: str = "fp32",
        compile_model: bool = False,
    ):
        """
        Initialize model loader.

        Args:
            checkpoint_dir: Directory containing checkpoints
            strict_validation: Raise errors on environment issues
            calibration_check: Validate model calibrations
            verify_integrity: Check checkpoint hashes
            device: Target device (auto-detected if None)
            precision: Weight precision (fp32, fp16, bf16)
            compile_model: Apply torch.compile optimization
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.strict_validation = strict_validation
        self.calibration_check = calibration_check
        self.verify_integrity = verify_integrity
        self.precision = precision
        self.compile_model = compile_model

        if device is None:
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.env_validator = EnvironmentValidator(strict=strict_validation)

        self._environment_validated = False
        self._loaded_models: Dict[str, nn.Module] = {}

    def validate_environment(self) -> EnvironmentReport:
        """
        Validate execution environment.

        Must be called before loading models when strict_validation=True.
        """
        report = self.env_validator.validate()
        self._environment_validated = True
        return report

    def _ensure_environment_validated(self):
        """Ensure environment has been validated."""
        if self.strict_validation and not self._environment_validated:
            raise EnvironmentValidationError(
                "Environment must be validated before loading models. "
                "Call loader.validate_environment() first."
            )

    def _apply_precision(self, state_dict: Dict) -> Dict:
        """Convert state dict to target precision."""
        if self.precision == "fp32":
            return state_dict

        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16

        converted = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                converted[key] = value.to(dtype)
            else:
                converted[key] = value

        return converted

    def load_detector(
        self,
        checkpoint_name: str = "detector_llama3_8b.pt",
        **kwargs
    ) -> "nn.Module":
        """
        Load conflict detector model.

        Args:
            checkpoint_name: Checkpoint file name
            **kwargs: Additional arguments passed to model constructor

        Returns:
            Loaded detector model
        """
        self._ensure_environment_validated()

        from src.models.conflict_detector import ConflictDetector

        checkpoint_path = self.checkpoint_dir / "detector" / checkpoint_name

        if self.verify_integrity:
            self.checkpoint_validator.verify_integrity(checkpoint_name)

        metadata = self.checkpoint_validator.load_metadata("detector")
        arch_config = metadata.get("model_architecture", {})

        model = ConflictDetector(
            input_dim=arch_config.get("input_dim", 4096),
            hidden_dim=arch_config.get("hidden_dim", 512),
            dropout=arch_config.get("dropout", 0.1),
            **kwargs
        )

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        state_dict = self._apply_precision(state_dict)
        model.load_state_dict(state_dict)

        model = model.to(self.device)
        model.eval()

        if self.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)

        if self.calibration_check:
            report = self.calibration_validator.validate_calibration(
                "detector", model
            )
            if not report.within_tolerance:
                warnings.warn(
                    f"Detector calibration drift detected: {report.calibration_drift:.4f}"
                )

        self._loaded_models["detector"] = model
        return model

    def load_refinement(
        self,
        checkpoint_name: str = "refinement_llama3_8b.pt",
        **kwargs
    ) -> "nn.Module":
        """
        Load refinement module.

        Args:
            checkpoint_name: Checkpoint file name
            **kwargs: Additional arguments passed to model constructor

        Returns:
            Loaded refinement model
        """
        self._ensure_environment_validated()

        from src.models.refinement_module import RefinementModule

        checkpoint_path = self.checkpoint_dir / "refinement" / checkpoint_name

        if self.verify_integrity:
            self.checkpoint_validator.verify_integrity(checkpoint_name)

        metadata = self.checkpoint_validator.load_metadata("refinement")
        arch_config = metadata.get("model_architecture", {})

        model = RefinementModule(
            input_dim=arch_config.get("input_dim", 4096),
            bottleneck_dim=arch_config.get("bottleneck_dim", 768),
            alpha=arch_config.get("alpha", 0.3),
            T_max=arch_config.get("T_max", 3),
            epsilon=arch_config.get("epsilon", 0.01),
            **kwargs
        )

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        state_dict = self._apply_precision(state_dict)
        model.load_state_dict(state_dict)

        model = model.to(self.device)
        model.eval()

        if self.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)

        if self.calibration_check:
            report = self.calibration_validator.validate_calibration(
                "refinement", model
            )
            if not report.within_tolerance:
                warnings.warn(
                    f"Refinement calibration drift detected: {report.calibration_drift:.4f}"
                )

        self._loaded_models["refinement"] = model
        return model

    def load_classifier(
        self,
        variant: str = "deberta_v3_large",
        checkpoint_name: Optional[str] = None,
        **kwargs
    ) -> "nn.Module":
        """
        Load taxonomy classifier.

        Args:
            variant: Model variant (deberta_v3_large, distilbert_base, tinybert)
            checkpoint_name: Override checkpoint file name
            **kwargs: Additional arguments

        Returns:
            Loaded classifier model
        """
        self._ensure_environment_validated()

        from src.models.taxonomy_classifier import TaxonomyClassifier

        if checkpoint_name is None:
            checkpoint_name = f"classifier_{variant}.pt"

        checkpoint_path = self.checkpoint_dir / "classifier" / checkpoint_name

        if self.verify_integrity:
            self.checkpoint_validator.verify_integrity(checkpoint_name)

        config_path = self.checkpoint_dir / "classifier" / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        variant_config = config["model_variants"].get(variant, {})

        model = TaxonomyClassifier(
            base_model=variant_config.get("base_model"),
            num_labels=variant_config.get("num_labels", 4),
            **kwargs
        )

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        state_dict = self._apply_precision(state_dict)
        model.load_state_dict(state_dict)

        model = model.to(self.device)
        model.eval()

        if self.calibration_check:
            report = self.calibration_validator.validate_calibration(
                "classifier", model
            )
            if not report.within_tolerance:
                warnings.warn(
                    f"Classifier calibration drift detected: {report.calibration_drift:.4f}"
                )

        self._loaded_models["classifier"] = model
        return model

    def load_all(self) -> Dict[str, "nn.Module"]:
        """Load all model components."""
        return {
            "detector": self.load_detector(),
            "refinement": self.load_refinement(),
            "classifier": self.load_classifier(),
        }


def verify_environment():
    """
    Convenience function to verify environment compatibility.

    Usage:
        python -c "from src.models.model_loader import verify_environment; verify_environment()"
    """
    validator = EnvironmentValidator(strict=False)
    validator.validate()
    validator.print_report()


if __name__ == "__main__":
    verify_environment()
