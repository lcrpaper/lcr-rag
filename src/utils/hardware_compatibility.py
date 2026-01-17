#!/usr/bin/env python3
"""
Hardware Compatibility Layer for LCR Models

This module provides hardware abstraction and compatibility checking for
running LCR models across different GPU architectures and configurations.

The LCR models were developed and validated on NVIDIA A100 GPUs. Running on
different hardware may require specific adaptations to maintain numerical
consistency and performance.

Key Considerations:
- Tensor Core availability and precision
- Memory bandwidth and capacity constraints
- Compute capability requirements
- Multi-GPU scaling behavior

Usage:
    from src.utils.hardware_compatibility import HardwareManager

    hw_manager = HardwareManager()
    hw_manager.validate_hardware()
    hw_manager.configure_optimal_settings()
"""

import os
import sys
import json
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.cuda as cuda
    import torch.backends.cudnn as cudnn
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class GPUArchitecture(Enum):
    """NVIDIA GPU architecture generations."""
    VOLTA = "volta"
    TURING = "turing"
    AMPERE = "ampere"
    ADA_LOVELACE = "ada"
    HOPPER = "hopper"
    UNKNOWN = "unknown"


@dataclass
class GPUCapabilities:
    """Detailed GPU capability information."""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    architecture: GPUArchitecture
    total_memory_gb: float
    memory_bandwidth_gbps: float
    tensor_cores: bool
    fp16_support: bool
    bf16_support: bool
    tf32_support: bool
    multi_instance_gpu: bool
    nvlink_available: bool

    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    int8_tops: float = 0.0

    recommended_batch_size: int = 32
    recommended_precision: str = "fp16"
    memory_fraction_limit: float = 0.9


GPU_SPECIFICATIONS = {
    "Tesla V100-SXM2-16GB": {
        "architecture": GPUArchitecture.VOLTA,
        "memory_bandwidth_gbps": 900,
        "fp32_tflops": 15.7,
        "fp16_tflops": 125,
        "tensor_cores": True,
        "bf16_support": False,
        "tf32_support": False,
        "recommended_batch_size": 24,
    },
    "Tesla V100-SXM2-32GB": {
        "architecture": GPUArchitecture.VOLTA,
        "memory_bandwidth_gbps": 900,
        "fp32_tflops": 15.7,
        "fp16_tflops": 125,
        "tensor_cores": True,
        "bf16_support": False,
        "tf32_support": False,
        "recommended_batch_size": 48,
    },

    "Tesla T4": {
        "architecture": GPUArchitecture.TURING,
        "memory_bandwidth_gbps": 320,
        "fp32_tflops": 8.1,
        "fp16_tflops": 65,
        "tensor_cores": True,
        "bf16_support": False,
        "tf32_support": False,
        "recommended_batch_size": 8,
    },
    "NVIDIA GeForce RTX 2080 Ti": {
        "architecture": GPUArchitecture.TURING,
        "memory_bandwidth_gbps": 616,
        "fp32_tflops": 13.4,
        "fp16_tflops": 26.9,
        "tensor_cores": True,
        "bf16_support": False,
        "tf32_support": False,
        "recommended_batch_size": 8,
    },

    "NVIDIA A100-SXM4-40GB": {
        "architecture": GPUArchitecture.AMPERE,
        "memory_bandwidth_gbps": 1555,
        "fp32_tflops": 19.5,
        "fp16_tflops": 312,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 64,
    },
    "NVIDIA A100-SXM4-80GB": {
        "architecture": GPUArchitecture.AMPERE,
        "memory_bandwidth_gbps": 2039,
        "fp32_tflops": 19.5,
        "fp16_tflops": 312,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 128,
    },
    "NVIDIA A100-PCIE-40GB": {
        "architecture": GPUArchitecture.AMPERE,
        "memory_bandwidth_gbps": 1555,
        "fp32_tflops": 19.5,
        "fp16_tflops": 312,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 64,
    },
    "NVIDIA GeForce RTX 3090": {
        "architecture": GPUArchitecture.AMPERE,
        "memory_bandwidth_gbps": 936,
        "fp32_tflops": 35.6,
        "fp16_tflops": 71,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 32,
    },
    "NVIDIA GeForce RTX 3080": {
        "architecture": GPUArchitecture.AMPERE,
        "memory_bandwidth_gbps": 760,
        "fp32_tflops": 29.8,
        "fp16_tflops": 59.5,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 16,
    },
    "NVIDIA A10": {
        "architecture": GPUArchitecture.AMPERE,
        "memory_bandwidth_gbps": 600,
        "fp32_tflops": 31.2,
        "fp16_tflops": 125,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 32,
    },

    "NVIDIA GeForce RTX 4090": {
        "architecture": GPUArchitecture.ADA_LOVELACE,
        "memory_bandwidth_gbps": 1008,
        "fp32_tflops": 82.6,
        "fp16_tflops": 165,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 48,
    },
    "NVIDIA GeForce RTX 4080": {
        "architecture": GPUArchitecture.ADA_LOVELACE,
        "memory_bandwidth_gbps": 716,
        "fp32_tflops": 48.7,
        "fp16_tflops": 97.5,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 24,
    },
    "NVIDIA L40": {
        "architecture": GPUArchitecture.ADA_LOVELACE,
        "memory_bandwidth_gbps": 864,
        "fp32_tflops": 90.5,
        "fp16_tflops": 181,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 64,
    },

    "NVIDIA H100-SXM5-80GB": {
        "architecture": GPUArchitecture.HOPPER,
        "memory_bandwidth_gbps": 3350,
        "fp32_tflops": 67,
        "fp16_tflops": 1979,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 256,
    },
    "NVIDIA H100-PCIE-80GB": {
        "architecture": GPUArchitecture.HOPPER,
        "memory_bandwidth_gbps": 2000,
        "fp32_tflops": 51,
        "fp16_tflops": 1513,
        "tensor_cores": True,
        "bf16_support": True,
        "tf32_support": True,
        "recommended_batch_size": 192,
    },
}


def get_architecture_from_compute_capability(major: int, minor: int) -> GPUArchitecture:
    """Determine GPU architecture from compute capability."""
    cc = (major, minor)

    if cc >= (9, 0):
        return GPUArchitecture.HOPPER
    elif cc >= (8, 9):
        return GPUArchitecture.ADA_LOVELACE
    elif cc >= (8, 0):
        return GPUArchitecture.AMPERE
    elif cc >= (7, 5):
        return GPUArchitecture.TURING
    elif cc >= (7, 0):
        return GPUArchitecture.VOLTA
    else:
        return GPUArchitecture.UNKNOWN


class HardwareCompatibilityError(Exception):
    """Raised when hardware is incompatible with LCR requirements."""
    pass


class HardwareManager:
    """
    Manages hardware detection, validation, and optimization for LCR models.

    This class handles:
    - GPU detection and capability assessment
    - Hardware compatibility validation
    - Optimal configuration for different GPU types
    - Multi-GPU setup and memory management
    - Precision and performance tuning
    """

    MIN_COMPUTE_CAPABILITY = (7, 0)
    MIN_MEMORY_GB = 8.0
    RECOMMENDED_MEMORY_GB = 24.0

    def __init__(
        self,
        device_ids: Optional[List[int]] = None,
        allow_cpu_fallback: bool = False,
        strict_validation: bool = True,
    ):
        """
        Initialize hardware manager.

        Args:
            device_ids: Specific GPU device IDs to use (None for auto-detect)
            allow_cpu_fallback: Allow CPU execution if GPU unavailable
            strict_validation: Raise errors on compatibility issues
        """
        self.device_ids = device_ids
        self.allow_cpu_fallback = allow_cpu_fallback
        self.strict_validation = strict_validation

        self._gpus: List[GPUCapabilities] = []
        self._validated = False
        self._configured = False

    def detect_gpus(self) -> List[GPUCapabilities]:
        """
        Detect and characterize available GPUs.

        Returns:
            List of GPUCapabilities for each detected GPU
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        if not torch.cuda.is_available():
            if self.allow_cpu_fallback:
                logger.warning("No GPU available, falling back to CPU")
                return []
            raise HardwareCompatibilityError("CUDA not available")

        device_count = torch.cuda.device_count()

        if self.device_ids is not None:
            for did in self.device_ids:
                if did >= device_count:
                    raise HardwareCompatibilityError(
                        f"Device ID {did} not available. Only {device_count} GPUs detected."
                    )
            devices = self.device_ids
        else:
            devices = list(range(device_count))

        gpus = []
        for device_id in devices:
            props = torch.cuda.get_device_properties(device_id)

            cc = (props.major, props.minor)
            arch = get_architecture_from_compute_capability(*cc)
            memory_gb = props.total_memory / (1024 ** 3)

            specs = GPU_SPECIFICATIONS.get(props.name, {})

            gpu = GPUCapabilities(
                device_id=device_id,
                name=props.name,
                compute_capability=cc,
                architecture=arch,
                total_memory_gb=memory_gb,
                memory_bandwidth_gbps=specs.get("memory_bandwidth_gbps", 0),
                tensor_cores=specs.get("tensor_cores", cc >= (7, 0)),
                fp16_support=cc >= (7, 0),
                bf16_support=specs.get("bf16_support", cc >= (8, 0)),
                tf32_support=specs.get("tf32_support", cc >= (8, 0)),
                multi_instance_gpu=props.multi_processor_count > 0,
                nvlink_available=False,
                fp32_tflops=specs.get("fp32_tflops", 0),
                fp16_tflops=specs.get("fp16_tflops", 0),
                recommended_batch_size=specs.get("recommended_batch_size", 32),
            )

            gpus.append(gpu)

        self._gpus = gpus
        return gpus

    def validate_hardware(self) -> Dict[str, Any]:
        """
        Validate hardware meets LCR requirements.

        Returns:
            Validation report dictionary

        Raises:
            HardwareCompatibilityError: If strict_validation and critical issues
        """
        if not self._gpus:
            self.detect_gpus()

        report = {
            "valid": True,
            "gpu_count": len(self._gpus),
            "gpus": [],
            "warnings": [],
            "errors": [],
        }

        if not self._gpus:
            if self.allow_cpu_fallback:
                report["warnings"].append("No GPUs detected, using CPU (slow)")
            else:
                report["valid"] = False
                report["errors"].append("No compatible GPUs detected")

        for gpu in self._gpus:
            gpu_report = {
                "device_id": gpu.device_id,
                "name": gpu.name,
                "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
                "memory_gb": round(gpu.total_memory_gb, 1),
                "architecture": gpu.architecture.value,
                "issues": [],
            }

            if gpu.compute_capability < self.MIN_COMPUTE_CAPABILITY:
                gpu_report["issues"].append(
                    f"Compute capability {gpu.compute_capability} below minimum "
                    f"{self.MIN_COMPUTE_CAPABILITY}"
                )
                report["errors"].append(
                    f"GPU {gpu.device_id} ({gpu.name}) compute capability too low"
                )
                report["valid"] = False

            if gpu.total_memory_gb < self.MIN_MEMORY_GB:
                gpu_report["issues"].append(
                    f"Memory {gpu.total_memory_gb:.1f}GB below minimum {self.MIN_MEMORY_GB}GB"
                )
                report["errors"].append(
                    f"GPU {gpu.device_id} ({gpu.name}) insufficient memory"
                )
                report["valid"] = False
            elif gpu.total_memory_gb < self.RECOMMENDED_MEMORY_GB:
                gpu_report["issues"].append(
                    f"Memory {gpu.total_memory_gb:.1f}GB below recommended {self.RECOMMENDED_MEMORY_GB}GB"
                )
                report["warnings"].append(
                    f"GPU {gpu.device_id} ({gpu.name}) may require reduced batch sizes"
                )

            if gpu.architecture == GPUArchitecture.VOLTA:
                report["warnings"].append(
                    f"GPU {gpu.device_id} uses Volta architecture. "
                    "Paper results obtained on Ampere (A100). "
                    "Minor numerical differences possible."
                )
            elif gpu.architecture == GPUArchitecture.UNKNOWN:
                report["warnings"].append(
                    f"GPU {gpu.device_id} ({gpu.name}) architecture not recognized. "
                    "Compatibility not guaranteed."
                )

            report["gpus"].append(gpu_report)

        self._validated = True

        if self.strict_validation and not report["valid"]:
            error_msg = "Hardware validation failed:\n" + "\n".join(report["errors"])
            raise HardwareCompatibilityError(error_msg)

        return report

    def configure_optimal_settings(self) -> Dict[str, Any]:
        """
        Configure PyTorch for optimal performance on detected hardware.

        Returns:
            Dictionary of applied settings
        """
        if not self._validated:
            self.validate_hardware()

        settings = {
            "cudnn_enabled": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32_enabled": False,
            "fp16_available": False,
            "bf16_available": False,
            "memory_efficient_attention": False,
        }

        if not _TORCH_AVAILABLE or not self._gpus:
            return settings

        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True

        has_tf32 = any(gpu.tf32_support for gpu in self._gpus)
        if has_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            settings["tf32_available"] = True

        settings["fp16_available"] = any(gpu.fp16_support for gpu in self._gpus)
        settings["bf16_available"] = any(gpu.bf16_support for gpu in self._gpus)

        min_memory = min(gpu.total_memory_gb for gpu in self._gpus)
        if min_memory < 16:
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                settings["memory_efficient_attention"] = True

        self._configured = True
        return settings

    def get_recommended_batch_size(self, model_component: str = "refinement") -> int:
        """
        Get recommended batch size for hardware configuration.

        Args:
            model_component: One of "detector", "refinement", "classifier"

        Returns:
            Recommended batch size
        """
        if not self._gpus:
            return 1

        memory_per_sample = {
            "detector": 0.1,
            "refinement": 0.5,
            "classifier": 0.8,
        }

        min_memory = min(gpu.total_memory_gb for gpu in self._gpus)
        available_memory = min_memory * 0.8

        mem_per_sample = memory_per_sample.get(model_component, 0.5)
        batch_size = int(available_memory / mem_per_sample)

        batch_size = 2 ** (batch_size.bit_length() - 1)

        return max(1, min(batch_size, 128))

    def get_precision_recommendation(self) -> str:
        """
        Get recommended precision based on hardware.

        Returns:
            One of "fp32", "fp16", "bf16"
        """
        if not self._gpus:
            return "fp32"

        if any(gpu.bf16_support for gpu in self._gpus):
            return "bf16"

        if any(gpu.fp16_support and gpu.tensor_cores for gpu in self._gpus):
            return "fp16"

        return "fp32"

    def print_hardware_report(self):
        """Print formatted hardware report."""
        if not self._gpus:
            self.detect_gpus()

        print("\n" + "=" * 70)
        print("LCR HARDWARE CONFIGURATION REPORT")
        print("=" * 70)

        if not self._gpus:
            print("No GPUs detected. CPU mode only.")
            print("=" * 70)
            return

        for gpu in self._gpus:
            print(f"\nGPU {gpu.device_id}: {gpu.name}")
            print("-" * 50)
            print(f"  Architecture: {gpu.architecture.value.upper()}")
            print(f"  Compute Capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
            print(f"  Memory: {gpu.total_memory_gb:.1f} GB")
            print(f"  Memory Bandwidth: {gpu.memory_bandwidth_gbps} GB/s")
            print(f"  Tensor Cores: {'Yes' if gpu.tensor_cores else 'No'}")
            print(f"  FP16 Support: {'Yes' if gpu.fp16_support else 'No'}")
            print(f"  BF16 Support: {'Yes' if gpu.bf16_support else 'No'}")
            print(f"  TF32 Support: {'Yes' if gpu.tf32_support else 'No'}")

            if gpu.fp32_tflops > 0:
                print(f"  FP32 Performance: {gpu.fp32_tflops} TFLOPS")
                print(f"  FP16 Performance: {gpu.fp16_tflops} TFLOPS")

            print(f"  Recommended Batch Size: {gpu.recommended_batch_size}")

        print("\n" + "-" * 70)
        print("RECOMMENDATIONS")
        print("-" * 70)
        print(f"  Precision: {self.get_precision_recommendation()}")
        print(f"  Detector Batch Size: {self.get_recommended_batch_size('detector')}")
        print(f"  Refinement Batch Size: {self.get_recommended_batch_size('refinement')}")
        print(f"  Classifier Batch Size: {self.get_recommended_batch_size('classifier')}")

        print("=" * 70)


def main():
    """Run hardware detection and validation."""
    manager = HardwareManager(strict_validation=False)

    try:
        manager.detect_gpus()
        report = manager.validate_hardware()
        settings = manager.configure_optimal_settings()

        manager.print_hardware_report()

        print("\nApplied Settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")

        if report["errors"]:
            print("\nERRORS:")
            for error in report["errors"]:
                print(f"  Error: {error}")

        if report["warnings"]:
            print("\nWARNINGS:")
            for warning in report["warnings"]:
                print(f"  Warning: {warning}")

    except HardwareCompatibilityError as e:
        print(f"\nHardware Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
