#!/usr/bin/env python3
"""
LCR Environment Verification Script

This script performs comprehensive verification of the execution environment
before running any LCR experiments or inference.

IMPORTANT: Run this script BEFORE attempting to load models or reproduce results.
Environment mismatches can cause silent numerical discrepancies.

Usage:
    python scripts/utils/verify_environment.py
    python scripts/utils/verify_environment.py --strict
    python scripts/utils/verify_environment.py --output report.json
    python scripts/utils/verify_environment.py --check-gpu-memory 24

Exit Codes:
    0: Environment fully compatible
    1: Environment has warnings but may work
    2: Environment incompatible (critical errors)
"""

import os
import sys
import json
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


REQUIRED_PYTHON_VERSION = (3, 10)
REQUIRED_PACKAGES = {
    "torch": {"min": "2.1.0", "max": "2.3.0", "critical": True},
    "transformers": {"min": "4.35.0", "max": "4.40.0", "critical": True},
    "numpy": {"min": "1.24.0", "max": "2.0.0", "critical": True},
    "scipy": {"min": "1.11.0", "max": "1.13.0", "critical": False},
    "accelerate": {"min": "0.24.0", "max": "0.28.0", "critical": False},
    "safetensors": {"min": "0.4.0", "max": "0.5.0", "critical": False},
    "tokenizers": {"min": "0.15.0", "max": "0.16.0", "critical": False},
}

CUDA_REQUIREMENTS = {
    "min_version": "11.8",
    "max_version": "12.3",
    "min_compute_capability": 7.0,
    "min_memory_gb": 8.0,
    "recommended_memory_gb": 24.0,
}

CUDNN_REQUIREMENTS = {
    "min_version": 8700,
}


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to comparable tuple."""
    version_str = version_str.split("+")[0].split(".post")[0]
    parts = []
    for part in version_str.split(".")[:3]:
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)


def version_compare(v1: str, v2: str) -> int:
    """Compare two version strings. Returns -1, 0, or 1."""
    t1, t2 = parse_version(v1), parse_version(v2)
    if t1 < t2:
        return -1
    elif t1 > t2:
        return 1
    return 0


def check_python_version() -> Dict[str, Any]:
    """Check Python version compatibility."""
    current = sys.version_info[:2]
    required = REQUIRED_PYTHON_VERSION

    result = {
        "check": "Python Version",
        "current": f"{current[0]}.{current[1]}",
        "required": f">= {required[0]}.{required[1]}",
        "status": "ok" if current >= required else "error",
    }

    if current < required:
        result["message"] = f"Python {required[0]}.{required[1]}+ required"

    return result


def check_package_version(package: str, requirements: Dict) -> Dict[str, Any]:
    """Check if a package meets version requirements."""
    result = {
        "check": f"Package: {package}",
        "required": f"{requirements['min']} - {requirements['max']}",
        "critical": requirements.get("critical", False),
    }

    try:
        if package == "torch":
            import torch
            version = torch.__version__
        elif package == "transformers":
            import transformers
            version = transformers.__version__
        elif package == "numpy":
            import numpy
            version = numpy.__version__
        elif package == "scipy":
            import scipy
            version = scipy.__version__
        elif package == "accelerate":
            import accelerate
            version = accelerate.__version__
        elif package == "safetensors":
            import safetensors
            version = safetensors.__version__
        elif package == "tokenizers":
            import tokenizers
            version = tokenizers.__version__
        else:
            import importlib
            mod = importlib.import_module(package)
            version = getattr(mod, "__version__", "unknown")

        result["current"] = version

        if version_compare(version, requirements["min"]) < 0:
            result["status"] = "error" if requirements["critical"] else "warning"
            result["message"] = f"Version {version} below minimum {requirements['min']}"
        elif version_compare(version, requirements["max"]) >= 0:
            result["status"] = "warning"
            result["message"] = f"Version {version} above tested range (max {requirements['max']})"
        else:
            result["status"] = "ok"

    except ImportError:
        result["current"] = "not installed"
        result["status"] = "error" if requirements["critical"] else "warning"
        result["message"] = f"Package {package} not installed"

    return result


def check_cuda() -> Dict[str, Any]:
    """Check CUDA availability and version."""
    result = {
        "check": "CUDA",
        "required": f"{CUDA_REQUIREMENTS['min_version']} - {CUDA_REQUIREMENTS['max_version']}",
    }

    try:
        import torch

        if not torch.cuda.is_available():
            result["current"] = "not available"
            result["status"] = "error"
            result["message"] = "CUDA not available. GPU required for LCR models."
            return result

        cuda_version = torch.version.cuda
        result["current"] = cuda_version

        if version_compare(cuda_version, CUDA_REQUIREMENTS["min_version"]) < 0:
            result["status"] = "error"
            result["message"] = f"CUDA {cuda_version} below minimum {CUDA_REQUIREMENTS['min_version']}"
        elif version_compare(cuda_version, CUDA_REQUIREMENTS["max_version"]) > 0:
            result["status"] = "warning"
            result["message"] = f"CUDA {cuda_version} above tested versions"
        else:
            result["status"] = "ok"

    except ImportError:
        result["current"] = "torch not installed"
        result["status"] = "error"

    return result


def check_cudnn() -> Dict[str, Any]:
    """Check cuDNN version."""
    result = {
        "check": "cuDNN",
        "required": f">= {CUDNN_REQUIREMENTS['min_version']}",
    }

    try:
        import torch

        if not torch.backends.cudnn.is_available():
            result["current"] = "not available"
            result["status"] = "warning"
            result["message"] = "cuDNN not available. Performance may be degraded."
            return result

        cudnn_version = torch.backends.cudnn.version()
        result["current"] = str(cudnn_version)

        if cudnn_version < CUDNN_REQUIREMENTS["min_version"]:
            result["status"] = "warning"
            result["message"] = f"cuDNN version below recommended"
        else:
            result["status"] = "ok"

    except ImportError:
        result["current"] = "torch not installed"
        result["status"] = "error"

    return result


def check_gpu_hardware() -> List[Dict[str, Any]]:
    """Check GPU hardware compatibility."""
    results = []

    try:
        import torch

        if not torch.cuda.is_available():
            return [{
                "check": "GPU Hardware",
                "status": "error",
                "message": "No GPU available"
            }]

        device_count = torch.cuda.device_count()

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)

            compute_cap = float(f"{props.major}.{props.minor}")
            memory_gb = props.total_memory / (1024 ** 3)

            result = {
                "check": f"GPU {i}: {props.name}",
                "compute_capability": compute_cap,
                "memory_gb": round(memory_gb, 1),
            }

            issues = []

            if compute_cap < CUDA_REQUIREMENTS["min_compute_capability"]:
                issues.append(
                    f"Compute capability {compute_cap} below minimum "
                    f"{CUDA_REQUIREMENTS['min_compute_capability']}"
                )

            if memory_gb < CUDA_REQUIREMENTS["min_memory_gb"]:
                issues.append(
                    f"GPU memory {memory_gb:.1f}GB below minimum "
                    f"{CUDA_REQUIREMENTS['min_memory_gb']}GB"
                )
            elif memory_gb < CUDA_REQUIREMENTS["recommended_memory_gb"]:
                result["warning"] = (
                    f"GPU memory {memory_gb:.1f}GB below recommended "
                    f"{CUDA_REQUIREMENTS['recommended_memory_gb']}GB. "
                    f"May need to reduce batch sizes."
                )

            if issues:
                result["status"] = "error"
                result["message"] = "; ".join(issues)
            else:
                result["status"] = "ok"

            results.append(result)

    except ImportError:
        results.append({
            "check": "GPU Hardware",
            "status": "error",
            "message": "torch not installed"
        })

    return results


def check_blas_libraries() -> Dict[str, Any]:
    """Check BLAS/LAPACK configuration for numerical consistency."""
    result = {
        "check": "BLAS/LAPACK Configuration",
    }

    try:
        import numpy as np

        blas_info = np.__config__.get_info("blas")
        lapack_info = np.__config__.get_info("lapack")

        result["blas"] = str(blas_info.get("libraries", ["unknown"]))
        result["lapack"] = str(lapack_info.get("libraries", ["unknown"]))
        result["status"] = "ok"

        blas_libs = str(blas_info.get("libraries", []))
        if "mkl" in blas_libs.lower():
            result["note"] = "Intel MKL detected (good for reproducibility)"
        elif "openblas" in blas_libs.lower():
            result["note"] = "OpenBLAS detected (may have slight numerical differences)"
            result["status"] = "warning"

    except Exception as e:
        result["status"] = "warning"
        result["message"] = f"Could not determine BLAS configuration: {e}"

    return result


def check_torch_backends() -> Dict[str, Any]:
    """Check PyTorch backend settings."""
    result = {
        "check": "PyTorch Backends",
    }

    try:
        import torch

        result["cudnn_enabled"] = torch.backends.cudnn.enabled
        result["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        result["cudnn_benchmark"] = torch.backends.cudnn.benchmark

        if not torch.backends.cudnn.enabled:
            result["status"] = "warning"
            result["message"] = "cuDNN disabled. Enable for better performance."
        elif torch.backends.cudnn.benchmark and not torch.backends.cudnn.deterministic:
            result["status"] = "warning"
            result["message"] = (
                "cuDNN benchmark enabled but not deterministic. "
                "Results may vary between runs."
            )
        else:
            result["status"] = "ok"

    except ImportError:
        result["status"] = "error"
        result["message"] = "torch not installed"

    return result


def check_environment_variables() -> Dict[str, Any]:
    """Check relevant environment variables."""
    result = {
        "check": "Environment Variables",
        "variables": {},
    }

    relevant_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "TORCH_HOME",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]

    for var in relevant_vars:
        value = os.environ.get(var)
        if value:
            result["variables"][var] = value

    result["status"] = "ok"
    return result


def check_disk_space() -> Dict[str, Any]:
    """Check available disk space for checkpoints and data."""
    result = {
        "check": "Disk Space",
    }

    try:
        import shutil

        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024 ** 3)

        result["free_gb"] = round(free_gb, 1)

        if free_gb < 10:
            result["status"] = "error"
            result["message"] = f"Only {free_gb:.1f}GB free. Need at least 10GB for checkpoints."
        elif free_gb < 50:
            result["status"] = "warning"
            result["message"] = f"{free_gb:.1f}GB free. Recommend 50GB+ for full experiments."
        else:
            result["status"] = "ok"

    except Exception as e:
        result["status"] = "warning"
        result["message"] = f"Could not check disk space: {e}"

    return result


def check_checkpoint_integrity() -> List[Dict[str, Any]]:
    """Verify checkpoint files exist and have expected sizes."""
    results = []

    expected_checkpoints = {
        "checkpoints/detector/detector_llama3_8b.pt": {"min_size_mb": 5, "max_size_mb": 15},
        "checkpoints/refinement/refinement_llama3_8b.pt": {"min_size_mb": 20, "max_size_mb": 35},
        "checkpoints/classifier/classifier_deberta_v3_large.pt": {"min_size_mb": 1000, "max_size_mb": 1500},
    }

    for checkpoint, size_req in expected_checkpoints.items():
        result = {"check": f"Checkpoint: {checkpoint}"}

        path = Path(checkpoint)
        if not path.exists():
            result["status"] = "error"
            result["message"] = "File not found"
        else:
            size_mb = path.stat().st_size / (1024 ** 2)
            result["size_mb"] = round(size_mb, 1)

            if size_mb < size_req["min_size_mb"]:
                result["status"] = "error"
                result["message"] = f"Size {size_mb:.1f}MB below expected minimum {size_req['min_size_mb']}MB"
            elif size_mb > size_req["max_size_mb"]:
                result["status"] = "warning"
                result["message"] = f"Size {size_mb:.1f}MB above expected maximum"
            else:
                result["status"] = "ok"

        results.append(result)

    return results


def run_all_checks(args) -> Dict[str, Any]:
    """Run all environment checks."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "checks": [],
        "summary": {
            "total": 0,
            "ok": 0,
            "warnings": 0,
            "errors": 0,
        }
    }

    checks = []

    checks.append(check_python_version())

    for package, requirements in REQUIRED_PACKAGES.items():
        checks.append(check_package_version(package, requirements))

    checks.append(check_cuda())
    checks.append(check_cudnn())

    checks.extend(check_gpu_hardware())

    checks.append(check_blas_libraries())

    checks.append(check_torch_backends())

    checks.append(check_environment_variables())

    checks.append(check_disk_space())

    checks.extend(check_checkpoint_integrity())

    report["checks"] = checks

    for check in checks:
        report["summary"]["total"] += 1
        status = check.get("status", "ok")
        if status == "ok":
            report["summary"]["ok"] += 1
        elif status == "warning":
            report["summary"]["warnings"] += 1
        else:
            report["summary"]["errors"] += 1

    return report


def print_report(report: Dict[str, Any], verbose: bool = False):
    """Print formatted report to console."""
    print("\n" + "=" * 70)
    print("LCR ENVIRONMENT VERIFICATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Platform: {report['platform']}")
    print()

    for check in report["checks"]:
        status = check.get("status", "ok")

        if status == "ok":
            icon = "✓"
        elif status == "warning":
            icon = "⚠"
        else:
            icon = "✗"

        name = check.get("check", "Unknown")
        current = check.get("current", "")

        print(f"  {icon} {name}")
        if current and verbose:
            print(f"      Current: {current}")
        if "message" in check:
            print(f"      {check['message']}")

    print()
    print("-" * 70)
    summary = report["summary"]
    print(f"Summary: {summary['ok']}/{summary['total']} OK, "
          f"{summary['warnings']} warnings, {summary['errors']} errors")

    if summary["errors"] > 0:
        print("\n⚠ ENVIRONMENT HAS CRITICAL ISSUES")
        print("  Please resolve errors before running LCR models.")
        print("  See documentation: PREREQUISITES.md")
    elif summary["warnings"] > 0:
        print("\n⚠ Environment has warnings. Results may differ from paper.")
    else:
        print("\n✓ Environment is fully compatible")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Verify LCR execution environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/utils/verify_environment.py
    python scripts/utils/verify_environment.py --strict
    python scripts/utils/verify_environment.py --output report.json
        """
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code on any warnings"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save report to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--check-gpu-memory",
        type=float,
        help="Require specific GPU memory (GB)"
    )

    args = parser.parse_args()

    report = run_all_checks(args)

    print_report(report, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")

    summary = report["summary"]
    if summary["errors"] > 0:
        sys.exit(2)
    elif args.strict and summary["warnings"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
