"""
Environment Checker for LCR Repository

Usage:
    python scripts/utils/check_environment.py

Returns exit code 0 if all checks pass, 1 if critical failures
"""

import sys
import platform
import subprocess
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version() -> bool:
    """Check Python version is 3.8-3.10."""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3:
        logger.error("✗ Python 3.x required")
        return False

    if version.minor < 8 or version.minor > 10:
        logger.warning(f"⚠  Python {version.major}.{version.minor} detected")
        logger.warning("   Recommended: Python 3.8-3.10 (3.11 untested)")
        return True

    logger.info("✓ Python version OK")
    return True


def check_cuda() -> bool:
    """Check CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPUs detected: {torch.cuda.device_count()}")
            return True
        else:
            logger.warning("⚠  CUDA not available - CPU mode only")
            logger.warning("   Training will be VERY slow (~40+ hours)")
            logger.warning("   Evaluation should work but be patient")
            return True

    except ImportError:
        logger.warning("⚠  PyTorch not installed - cannot check CUDA")
        return True


def check_packages() -> bool:
    """Check critical packages are installed."""
    logger.info("\nChecking critical packages...")

    critical_packages = [
        ('torch', '2.1.0'),
        ('transformers', '4.35.0'),
        ('numpy', '1.24.0'),
    ]

    all_ok = True

    for package, min_version in critical_packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            logger.info(f"✓ {package}: {version}")

            if version != 'unknown' and version < min_version:
                logger.warning(f"  ⚠  Version {version} < recommended {min_version}")

        except ImportError:
            logger.error(f"✗ {package} not installed")
            all_ok = False

    if all_ok:
        logger.info("✓ All critical packages installed")
    else:
        logger.error("\nMissing packages! Install with:")
        logger.error("  pip install -r requirements.txt")

    return all_ok


def check_disk_space() -> bool:
    """Check available disk space."""
    try:
        stat = shutil.disk_usage(Path.cwd())
        free_gb = stat.free / (1024**3)

        logger.info(f"\nDisk space: {free_gb:.1f} GB free")

        if free_gb < 50:
            logger.error(f"✗ Insufficient disk space ({free_gb:.1f} GB < 50 GB minimum)")
            return False
        elif free_gb < 100:
            logger.warning(f"⚠  Low disk space ({free_gb:.1f} GB < 100 GB recommended)")
            logger.warning("   Full reproduction requires ~100GB")
            return True
        else:
            logger.info("✓ Sufficient disk space")
            return True

    except Exception as e:
        logger.warning(f"⚠  Could not check disk space: {e}")
        return True


def check_data_files() -> bool:
    """Check that required data files exist."""
    logger.info("\nChecking data files...")

    required_paths = [
        'data/benchmarks/temp_conflict/train.jsonl',
        'data/benchmarks/num_conflict/train.jsonl',
        'data/benchmarks/entity_conflict/train.jsonl',
        'data/benchmarks/semantic_conflict/train.jsonl',
    ]

    all_exist = True
    for path in required_paths:
        if Path(path).exists():
            logger.info(f"✓ Found: {path}")
        else:
            logger.info(f"✗ Missing: {path}")
            all_exist = False

    if not all_exist:
        logger.warning("\n⚠  Data files missing. Generate with:")
        logger.warning("  python scripts/generate_real_datasets.py")
        logger.warning("  OR: make data-paper")
    else:
        logger.info("✓ All benchmark files present")

    return True


def main():
    logger.info("="*60)
    logger.info("LCR ENVIRONMENT CHECK")
    logger.info("="*60)

    checks = [
        ("Python Version", check_python_version),
        ("CUDA / GPU", check_cuda),
        ("Required Packages", check_packages),
        ("Disk Space", check_disk_space),
        ("Data Files", check_data_files),
    ]

    results = {}
    critical_failure = False

    for name, check_func in checks:
        logger.info(f"\n[{name}]")
        try:
            passed = check_func()
            results[name] = passed
            if not passed and name in ["Python Version", "Required Packages"]:
                critical_failure = True
        except Exception as e:
            logger.error(f"✗ Check failed with error: {e}")
            results[name] = False
            critical_failure = True

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    passed_count = sum(results.values())
    total_count = len(results)

    logger.info(f"Checks passed: {passed_count}/{total_count}")

    if critical_failure:
        logger.error("\n✗ CRITICAL FAILURES DETECTED")
        logger.error("  Fix errors above before proceeding")
        return 1
    elif passed_count == total_count:
        logger.info("\n✓ ALL CHECKS PASSED")
        logger.info("  Environment ready for LCR")
    else:
        logger.warning("\n⚠  WARNINGS DETECTED")
        logger.warning("  Review warnings above")
        logger.warning("  You can proceed but may encounter issues")

    return 0 if not critical_failure else 1


if __name__ == "__main__":
    sys.exit(main())
