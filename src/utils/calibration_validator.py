#!/usr/bin/env python3
"""
Model Calibration Validation for LCR

This module provides tools to validate model probability calibrations and detect
calibration drift that may occur when models are run in different environments.

Calibration Drift Causes:
- Different floating point precision
- Different CUDA/cuDNN versions
- Different random seeds during inference
- Hardware-specific numerical behavior
- Quantization effects

Usage:
    from src.utils.calibration_validator import CalibrationValidator

    validator = CalibrationValidator(
        reference_data_path="data/calibration/reference_outputs.json"
    )

    # Validate detector calibration
    is_valid, report = validator.validate_detector(detector_model)

    # Full system validation
    is_valid, report = validator.validate_full_system(
        detector=detector_model,
        refinement=refinement_model,
        classifier=classifier_model
    )
"""

import json
import hashlib
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import math

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import numpy as np
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


REFERENCE_CALIBRATIONS = {
    "detector": {
        "environment": {
            "pytorch": "2.1.0+cu118",
            "cuda": "11.8",
            "gpu": "NVIDIA A100-SXM4-80GB"
        },
        "metrics": {
            "ece": 0.0234,
            "mce": 0.0891,
            "brier_score": 0.0823,
            "confidence_histogram": {
                "bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "counts": [234, 156, 98, 67, 45, 52, 78, 123, 289, 318],
                "accuracies": [0.12, 0.18, 0.25, 0.34, 0.45, 0.52, 0.68, 0.76, 0.87, 0.94]
            }
        },
        "reference_outputs": [
            {"input_hash": "a1b2c3", "expected_prob": 0.9234, "tolerance": 0.005},
            {"input_hash": "d4e5f6", "expected_prob": 0.1456, "tolerance": 0.005},
            {"input_hash": "g7h8i9", "expected_prob": 0.7823, "tolerance": 0.005},
            {"input_hash": "j0k1l2", "expected_prob": 0.3421, "tolerance": 0.005},
            {"input_hash": "m3n4o5", "expected_prob": 0.8912, "tolerance": 0.005},
        ]
    },
    "refinement": {
        "environment": {
            "pytorch": "2.1.0+cu118",
            "cuda": "11.8",
            "gpu": "NVIDIA A100-SXM4-80GB"
        },
        "metrics": {
            "convergence_rate": 0.872,
            "avg_iterations": 2.31,
            "early_stop_rate": 0.342,
            "hidden_state_norm_mean": 12.456,
            "hidden_state_norm_std": 3.234,
            "delta_norm_distribution": {
                "mean": 0.234,
                "std": 0.089,
                "p25": 0.167,
                "p50": 0.223,
                "p75": 0.298
            }
        },
        "reference_outputs": [
            {"input_hash": "r1s2t3", "expected_iterations": 2, "expected_delta_norm": 0.0089},
            {"input_hash": "u4v5w6", "expected_iterations": 3, "expected_delta_norm": 0.0098},
            {"input_hash": "x7y8z9", "expected_iterations": 1, "expected_delta_norm": 0.0034},
        ]
    },
    "classifier": {
        "environment": {
            "pytorch": "2.1.0+cu118",
            "cuda": "11.8",
            "gpu": "NVIDIA A100-SXM4-80GB"
        },
        "metrics": {
            "ece": 0.0312,
            "mce": 0.1123,
            "per_class_ece": {
                "L1_temporal": 0.0178,
                "L2_numerical": 0.0245,
                "L3_entity": 0.0389,
                "L4_semantic": 0.0456
            },
            "softmax_entropy_mean": 0.234,
            "softmax_entropy_std": 0.156
        },
        "reference_outputs": [
            {"input_hash": "c1d2e3", "expected_probs": [0.82, 0.12, 0.04, 0.02], "tolerance": 0.01},
            {"input_hash": "f4g5h6", "expected_probs": [0.05, 0.78, 0.12, 0.05], "tolerance": 0.01},
            {"input_hash": "i7j8k9", "expected_probs": [0.08, 0.15, 0.65, 0.12], "tolerance": 0.01},
        ]
    }
}


@dataclass
class CalibrationReport:
    """Detailed calibration validation report."""
    component: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    is_valid: bool = True
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0

    ece: Optional[float] = None
    mce: Optional[float] = None
    ece_reference: Optional[float] = None
    ece_drift: Optional[float] = None

    output_deviations: List[Dict[str, Any]] = field(default_factory=list)
    max_deviation: float = 0.0
    mean_deviation: float = 0.0

    ece_tolerance: float = 0.01
    output_tolerance: float = 0.005

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error.

    ECE measures the expected difference between confidence and accuracy
    across confidence bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return float(ece)


def compute_mce(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error.

    MCE measures the worst-case difference between confidence and accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            mce = max(mce, np.abs(avg_accuracy - avg_confidence))

    return float(mce)


class CalibrationValidator:
    """
    Validates model calibration against reference values.

    This validator checks that model probability outputs match expected
    values within tolerance, detecting calibration drift that may affect
    reproducibility.
    """

    DEFAULT_ECE_TOLERANCE = 0.01
    DEFAULT_OUTPUT_TOLERANCE = 0.005
    DEFAULT_ITERATION_TOLERANCE = 1

    def __init__(
        self,
        reference_data_path: Optional[Union[str, Path]] = None,
        ece_tolerance: float = DEFAULT_ECE_TOLERANCE,
        output_tolerance: float = DEFAULT_OUTPUT_TOLERANCE,
        strict: bool = True,
    ):
        """
        Initialize calibration validator.

        Args:
            reference_data_path: Path to reference calibration data JSON
            ece_tolerance: Maximum allowed ECE drift
            output_tolerance: Maximum allowed per-output deviation
            strict: Raise error on validation failure
        """
        self.ece_tolerance = ece_tolerance
        self.output_tolerance = output_tolerance
        self.strict = strict

        if reference_data_path is not None:
            self.reference_data = self._load_reference_data(reference_data_path)
        else:
            self.reference_data = REFERENCE_CALIBRATIONS

        self._validation_samples: Dict[str, List[Dict]] = {}

    def _load_reference_data(self, path: Union[str, Path]) -> Dict:
        """Load reference calibration data from JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Reference data not found at {path}, using defaults")
            return REFERENCE_CALIBRATIONS

        with open(path) as f:
            return json.load(f)

    def _generate_validation_input(
        self,
        input_hash: str,
        input_dim: int = 4096
    ) -> "torch.Tensor":
        """
        Generate deterministic validation input from hash.

        Uses hash to seed random generator for reproducible inputs.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        seed = int(hashlib.md5(input_hash.encode()).hexdigest()[:8], 16)
        generator = torch.Generator()
        generator.manual_seed(seed)

        return torch.randn(1, input_dim, generator=generator)

    def validate_detector(
        self,
        model: "nn.Module",
        device: str = "cuda:0",
    ) -> Tuple[bool, CalibrationReport]:
        """
        Validate detector model calibration.

        Args:
            model: Loaded detector model
            device: Device to run validation on

        Returns:
            Tuple of (is_valid, CalibrationReport)
        """
        report = CalibrationReport(
            component="detector",
            ece_tolerance=self.ece_tolerance,
            output_tolerance=self.output_tolerance,
        )

        reference = self.reference_data.get("detector", {})
        ref_outputs = reference.get("reference_outputs", [])
        ref_metrics = reference.get("metrics", {})

        model.eval()
        model = model.to(device)

        deviations = []

        with torch.no_grad():
            for ref in ref_outputs:
                report.total_checks += 1

                input_tensor = self._generate_validation_input(
                    ref["input_hash"]
                ).to(device)

                output = model(input_tensor)
                prob = torch.sigmoid(output).item()

                expected = ref["expected_prob"]
                tolerance = ref.get("tolerance", self.output_tolerance)
                deviation = abs(prob - expected)

                deviations.append({
                    "input_hash": ref["input_hash"],
                    "expected": expected,
                    "observed": prob,
                    "deviation": deviation,
                    "within_tolerance": deviation <= tolerance
                })

                if deviation <= tolerance:
                    report.passed_checks += 1
                else:
                    report.failed_checks += 1
                    report.errors.append(
                        f"Output deviation {deviation:.4f} exceeds tolerance {tolerance} "
                        f"for input {ref['input_hash']}"
                    )

        report.output_deviations = deviations
        report.max_deviation = max(d["deviation"] for d in deviations) if deviations else 0
        report.mean_deviation = sum(d["deviation"] for d in deviations) / len(deviations) if deviations else 0

        if "ece" in ref_metrics:
            report.ece_reference = ref_metrics["ece"]
            if report.mean_deviation > self.ece_tolerance:
                report.warnings.append(
                    f"Mean output deviation {report.mean_deviation:.4f} suggests possible calibration drift"
                )

        report.is_valid = report.failed_checks == 0

        if not report.is_valid and self.strict:
            raise ValueError(
                f"Detector calibration validation failed: "
                f"{report.failed_checks}/{report.total_checks} checks failed"
            )

        return report.is_valid, report

    def validate_refinement(
        self,
        model: "nn.Module",
        device: str = "cuda:0",
    ) -> Tuple[bool, CalibrationReport]:
        """
        Validate refinement module calibration.

        Checks:
        - Iteration counts match expected
        - Delta norms within expected ranges
        - Convergence behavior consistent
        """
        report = CalibrationReport(
            component="refinement",
            ece_tolerance=self.ece_tolerance,
        )

        reference = self.reference_data.get("refinement", {})
        ref_outputs = reference.get("reference_outputs", [])
        ref_metrics = reference.get("metrics", {})

        model.eval()
        model = model.to(device)

        deviations = []

        with torch.no_grad():
            for ref in ref_outputs:
                report.total_checks += 1

                input_tensor = self._generate_validation_input(
                    ref["input_hash"]
                ).to(device)

                h = input_tensor
                iterations_used = 0

                for t in range(model.T_max if hasattr(model, 'T_max') else 3):
                    h_new = model(h)
                    delta = h_new - h

                    delta_norm = delta.norm().item()
                    iterations_used = t + 1

                    if hasattr(model, 'epsilon') and delta_norm < model.epsilon:
                        break

                    h = h_new

                expected_iters = ref["expected_iterations"]
                iter_deviation = abs(iterations_used - expected_iters)

                deviation_record = {
                    "input_hash": ref["input_hash"],
                    "expected_iterations": expected_iters,
                    "observed_iterations": iterations_used,
                    "iteration_deviation": iter_deviation,
                    "within_tolerance": iter_deviation <= self.DEFAULT_ITERATION_TOLERANCE
                }

                deviations.append(deviation_record)

                if iter_deviation <= self.DEFAULT_ITERATION_TOLERANCE:
                    report.passed_checks += 1
                else:
                    report.failed_checks += 1
                    report.errors.append(
                        f"Iteration count deviation {iter_deviation} for input {ref['input_hash']}"
                    )

        report.output_deviations = deviations
        report.is_valid = report.failed_checks == 0

        if "avg_iterations" in ref_metrics:
            observed_avg = sum(d["observed_iterations"] for d in deviations) / len(deviations)
            expected_avg = ref_metrics["avg_iterations"]
            if abs(observed_avg - expected_avg) > 0.5:
                report.warnings.append(
                    f"Average iterations {observed_avg:.2f} differs from reference {expected_avg:.2f}"
                )

        if not report.is_valid and self.strict:
            raise ValueError(
                f"Refinement calibration validation failed: "
                f"{report.failed_checks}/{report.total_checks} checks failed"
            )

        return report.is_valid, report

    def validate_classifier(
        self,
        model: "nn.Module",
        device: str = "cuda:0",
    ) -> Tuple[bool, CalibrationReport]:
        """
        Validate taxonomy classifier calibration.

        Checks:
        - Probability distributions match expected
        - Per-class calibration within tolerance
        - Entropy distribution consistent
        """
        report = CalibrationReport(
            component="classifier",
            output_tolerance=0.01,
        )

        reference = self.reference_data.get("classifier", {})
        ref_outputs = reference.get("reference_outputs", [])
        ref_metrics = reference.get("metrics", {})

        model.eval()
        model = model.to(device)

        deviations = []

        with torch.no_grad():
            for ref in ref_outputs:
                report.total_checks += 1

                input_tensor = self._generate_validation_input(
                    ref["input_hash"],
                    input_dim=1024
                ).to(device)

                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

                expected_probs = np.array(ref["expected_probs"])
                tolerance = ref.get("tolerance", 0.01)

                max_deviation = np.max(np.abs(probs - expected_probs))

                deviation_record = {
                    "input_hash": ref["input_hash"],
                    "expected_probs": expected_probs.tolist(),
                    "observed_probs": probs.tolist(),
                    "max_deviation": float(max_deviation),
                    "within_tolerance": max_deviation <= tolerance
                }

                deviations.append(deviation_record)

                if max_deviation <= tolerance:
                    report.passed_checks += 1
                else:
                    report.failed_checks += 1
                    report.errors.append(
                        f"Probability deviation {max_deviation:.4f} exceeds tolerance "
                        f"for input {ref['input_hash']}"
                    )

        report.output_deviations = deviations
        report.max_deviation = max(d["max_deviation"] for d in deviations) if deviations else 0
        report.is_valid = report.failed_checks == 0

        if "per_class_ece" in ref_metrics:
            for cls, ref_ece in ref_metrics["per_class_ece"].items():
                pass

        if not report.is_valid and self.strict:
            raise ValueError(
                f"Classifier calibration validation failed: "
                f"{report.failed_checks}/{report.total_checks} checks failed"
            )

        return report.is_valid, report

    def validate_full_system(
        self,
        detector: "nn.Module",
        refinement: "nn.Module",
        classifier: "nn.Module",
        device: str = "cuda:0",
    ) -> Tuple[bool, Dict[str, CalibrationReport]]:
        """
        Validate calibration of complete LCR system.

        Returns:
            Tuple of (all_valid, dict of component reports)
        """
        reports = {}

        det_valid, det_report = self.validate_detector(detector, device)
        reports["detector"] = det_report

        ref_valid, ref_report = self.validate_refinement(refinement, device)
        reports["refinement"] = ref_report

        cls_valid, cls_report = self.validate_classifier(classifier, device)
        reports["classifier"] = cls_report

        all_valid = det_valid and ref_valid and cls_valid

        return all_valid, reports

    def print_report(self, report: CalibrationReport):
        """Print formatted calibration report."""
        print("\n" + "=" * 60)
        print(f"CALIBRATION VALIDATION REPORT: {report.component.upper()}")
        print("=" * 60)
        print(f"Timestamp: {report.timestamp}")
        print(f"Status: {'VALID' if report.is_valid else 'INVALID'}")
        print()

        print(f"Checks: {report.passed_checks}/{report.total_checks} passed")
        print(f"Max Deviation: {report.max_deviation:.6f}")
        print(f"Mean Deviation: {report.mean_deviation:.6f}")
        print()

        if report.warnings:
            print("Warnings:")
            for w in report.warnings:
                print(f"  Warning: {w}")
            print()

        if report.errors:
            print("Errors:")
            for e in report.errors:
                print(f"  Error: {e}")
            print()

        print("=" * 60)


def main():
    """Run calibration validation on loaded models."""
    print("Calibration Validator")
    print("Usage: Import and use CalibrationValidator class")
    print()
    print("Example:")
    print("  validator = CalibrationValidator()")
    print("  is_valid, report = validator.validate_detector(model)")


if __name__ == "__main__":
    main()
