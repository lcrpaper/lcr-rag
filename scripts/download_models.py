#!/usr/bin/env python3
"""
Model Checkpoint Downloader with MD5 Verification

Downloads pre-trained model checkpoints from anonymous hosting with:
- Progress bars for large files
- Resume capability for interrupted downloads
- Automatic MD5 checksum verification
- Support for individual or bulk downloads

Paper Reference: Appendix L.1, Lines 1746-1770

Usage:
    python scripts/download_models.py --all
    python scripts/download_models.py --detector --refinement
    python scripts/download_models.py --classifier distilbert
    python scripts/download_models.py --test-set expanded
    python scripts/download_models.py --verify-only
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


CHECKPOINTS = {
    "detector": {
        "name": "Conflict Detector",
        "filename": "detector_llama3_8b.pt",
        "size_mb": 8.4,
        "parameters": "2.1M",
        "output_dir": "checkpoints/detector/",
        "description": "Binary conflict detector trained on 14.6K examples"
    },
    "refinement": {
        "name": "Refinement Module",
        "filename": "refinement_llama3_8b.pt",
        "size_mb": 25.2,
        "parameters": "6.3M",
        "output_dir": "checkpoints/refinement/",
        "description": "Iterative hidden state refinement module"
    },
    "classifier_deberta": {
        "name": "Taxonomy Classifier (DeBERTa-v3-large)",
        "filename": "classifier_deberta_v3_large.pt",
        "size_mb": 1200,
        "parameters": "304M",
        "output_dir": "checkpoints/classifier/deberta_v3_large/",
        "description": "Main classifier - Macro F1=0.89"
    },
    "classifier_distilbert": {
        "name": "Taxonomy Classifier (DistilBERT-base)",
        "filename": "classifier_distilbert_base.pt",
        "size_mb": 264,
        "parameters": "66M",
        "output_dir": "checkpoints/classifier/distilbert_base/",
        "description": "Efficient classifier variant - Macro F1=0.85"
    },
    "classifier_tinybert": {
        "name": "Taxonomy Classifier (TinyBERT)",
        "filename": "classifier_tinybert.pt",
        "size_mb": 56,
        "parameters": "14M",
        "output_dir": "checkpoints/classifier/tinybert/",
        "description": "Lightweight classifier - Macro F1=0.78"
    },
    "natural_test_v1": {
        "name": "Natural Test Set v1 (Original)",
        "filename": "natural_test_set_v1.jsonl",
        "size_mb": 2.3,
        "parameters": "176 examples",
        "output_dir": "data/benchmarks/",
        "description": "Original natural-only test set"
    },
    "natural_test_v2": {
        "name": "Natural Test Set v2 (Expanded)",
        "filename": "natural_test_set_v2.jsonl",
        "size_mb": 5.1,
        "parameters": "400 examples",
        "output_dir": "data/benchmarks/",
        "description": "Expanded natural-only validation (Appendix H)"
    },
    "demo_detector": {
        "name": "Demo Detector (Quick Validation)",
        "filename": "demo_detector.pt",
        "size_mb": 3.2,
        "parameters": "2.1M (undertrained)",
        "output_dir": "checkpoints/demo/",
        "description": "Quick demo checkpoint - 92% of full performance on L1-L2"
    },
    "demo_refinement": {
        "name": "Demo Refinement (Quick Validation)",
        "filename": "demo_refinement.pt",
        "size_mb": 9.6,
        "parameters": "6.3M (undertrained)",
        "output_dir": "checkpoints/demo/",
        "description": "Quick demo checkpoint - 92% of full performance on L1-L2"
    }
}

BASE_URL = "https://anonymous.4open.science/r/lcr-rag-models/files/"

MIRROR_URLS = [
    "https://anonymous.4open.science/r/lcr-rag-models/files/",
]


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def download_with_progress(url: str, filepath: Path, expected_size_mb: float) -> bool:
    """
    Download file with progress bar and resume capability.

    Args:
        url: URL to download from
        filepath: Local path to save file
        expected_size_mb: Expected file size in MB for progress bar

    Returns:
        True if download successful, False otherwise
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    resume_header = {}
    initial_size = 0
    if filepath.exists():
        initial_size = filepath.stat().st_size
        resume_header = {"Range": f"bytes={initial_size}-"}
        print(f"  Resuming from {initial_size / 1024 / 1024:.1f} MB...")

    try:
        request = urllib.request.Request(url, headers=resume_header)

        with urllib.request.urlopen(request, timeout=30) as response:
            total_size = int(response.headers.get("content-length", 0)) + initial_size

            mode = "ab" if initial_size > 0 else "wb"
            with open(filepath, mode) as f:
                if HAS_TQDM:
                    with tqdm(total=total_size, initial=initial_size,
                             unit="B", unit_scale=True, desc="  Downloading") as pbar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    downloaded = initial_size
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = downloaded / (expected_size_mb * 1024 * 1024) * 100
                        print(f"\r  Progress: {progress:.1f}%", end="", flush=True)
                    print()

        return True

    except urllib.error.HTTPError as e:
        print(f"  HTTP Error: {e.code} - {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"  URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def verify_checkpoint(filepath: Path, expected_md5: str) -> Tuple[bool, str]:
    """
    Verify checkpoint integrity via MD5 checksum.

    Returns:
        Tuple of (passed, actual_md5)
    """
    if not filepath.exists():
        return False, "FILE_NOT_FOUND"

    actual_md5 = compute_md5(filepath)
    passed = actual_md5 == expected_md5
    return passed, actual_md5


def download_checkpoint(name: str, checkpoint_info: Dict,
                       base_url: str, verify: bool = True) -> bool:
    """
    Download and verify a single checkpoint.

    Args:
        name: Checkpoint identifier
        checkpoint_info: Checkpoint metadata dict
        base_url: Base URL for downloads
        verify: Whether to verify MD5 after download

    Returns:
        True if download and verification successful
    """
    print(f"\n{'='*60}")
    print(f"Checkpoint: {checkpoint_info['name']}")
    print(f"{'='*60}")
    print(f"  File: {checkpoint_info['filename']}")
    print(f"  Size: {checkpoint_info['size_mb']} MB")
    print(f"  Parameters: {checkpoint_info['parameters']}")
    print(f"  Description: {checkpoint_info['description']}")

    output_dir = Path(checkpoint_info['output_dir'])
    filepath = output_dir / checkpoint_info['filename']
    url = base_url + checkpoint_info['filename']

    if filepath.exists():
        print(f"\n  File exists at {filepath}")
        if verify:
            print("  Verifying existing file...")
            passed, actual_md5 = verify_checkpoint(filepath, checkpoint_info['md5'])
            if passed:
                print(f"  MD5: {actual_md5}")
                print("  Status: VERIFIED")
                return True
            else:
                print(f"  MD5 mismatch! Expected: {checkpoint_info['md5']}")
                print(f"  Got: {actual_md5}")
                print("  Re-downloading...")

    print(f"\n  Downloading from: {url}")
    success = download_with_progress(url, filepath, checkpoint_info['size_mb'])

    if not success:
        print("  DOWNLOAD FAILED")

        for mirror in MIRROR_URLS[1:]:
            print(f"  Trying mirror: {mirror}")
            mirror_url = mirror + checkpoint_info['filename']
            success = download_with_progress(mirror_url, filepath, checkpoint_info['size_mb'])
            if success:
                break

        if not success:
            return False

    if verify:
        print("\n  Verifying download...")
        passed, actual_md5 = verify_checkpoint(filepath, checkpoint_info['md5'])
        print(f"  Expected MD5: {checkpoint_info['md5']}")
        print(f"  Actual MD5:   {actual_md5}")

        if passed:
            print("  Status: PASS")
            return True
        else:
            print("  Status: FAIL - MD5 MISMATCH")
            print("\n  The downloaded file may be corrupted.")
            print("  Try deleting it and re-running this script.")
            print(f"  File location: {filepath}")
            return False

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download LCR model checkpoints with MD5 verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all checkpoints (recommended)
  python scripts/download_models.py --all

  # Download core models only (detector + refinement)
  python scripts/download_models.py --detector --refinement

  # Download specific classifier variant
  python scripts/download_models.py --classifier distilbert

  # Download test sets for evaluation
  python scripts/download_models.py --test-set original
  python scripts/download_models.py --test-set expanded

  # Verify existing checkpoints without downloading
  python scripts/download_models.py --verify-only

  # Download demo checkpoints for quick testing
  python scripts/download_models.py --demo
        """
    )

    parser.add_argument("--all", action="store_true",
                       help="Download all checkpoints (recommended)")
    parser.add_argument("--detector", action="store_true",
                       help="Download conflict detector checkpoint")
    parser.add_argument("--refinement", action="store_true",
                       help="Download refinement module checkpoint")
    parser.add_argument("--classifier", type=str,
                       choices=["deberta", "distilbert", "tinybert", "all"],
                       help="Download classifier checkpoint (default: deberta)")
    parser.add_argument("--test-set", type=str,
                       choices=["original", "expanded", "all"],
                       help="Download natural test sets")
    parser.add_argument("--demo", action="store_true",
                       help="Download demo checkpoints for quick validation")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing checkpoints (no download)")
    parser.add_argument("--no-verify", action="store_true",
                       help="Skip MD5 verification after download")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Base output directory (default: current directory)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LCR Model Checkpoint Downloader")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

    to_download = []

    if args.all:
        to_download = [k for k in CHECKPOINTS.keys() if not k.startswith("demo")]
    else:
        if args.detector:
            to_download.append("detector")
        if args.refinement:
            to_download.append("refinement")
        if args.classifier:
            if args.classifier == "all":
                to_download.extend(["classifier_deberta", "classifier_distilbert",
                                   "classifier_tinybert"])
            else:
                to_download.append(f"classifier_{args.classifier}")
        if args.test_set:
            if args.test_set == "original":
                to_download.append("natural_test_v1")
            elif args.test_set == "expanded":
                to_download.append("natural_test_v2")
            elif args.test_set == "all":
                to_download.extend(["natural_test_v1", "natural_test_v2"])
        if args.demo:
            to_download.extend(["demo_detector", "demo_refinement"])

    if args.verify_only:
        to_download = list(CHECKPOINTS.keys())

    if not to_download:
        print("\nNo checkpoints specified. Use --help for usage information.")
        print("Quick start: python scripts/download_models.py --all")
        return 1

    results = {}
    for name in to_download:
        if name not in CHECKPOINTS:
            print(f"\nWarning: Unknown checkpoint '{name}', skipping")
            continue

        info = CHECKPOINTS[name]

        if args.output_dir != ".":
            info = info.copy()
            info['output_dir'] = os.path.join(args.output_dir, info['output_dir'])

        if args.verify_only:
            filepath = Path(info['output_dir']) / info['filename']
            if filepath.exists():
                passed, actual_md5 = verify_checkpoint(filepath, info['md5'])
                status = "PASS" if passed else "FAIL"
                results[name] = passed
                print(f"\n{info['name']}: {status}")
                print(f"  File: {filepath}")
                print(f"  MD5: {actual_md5}")
            else:
                results[name] = None
                print(f"\n{info['name']}: NOT FOUND")
        else:
            success = download_checkpoint(name, info, BASE_URL,
                                         verify=not args.no_verify)
            results[name] = success

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for v in results.values() if v is True)
    fail_count = sum(1 for v in results.values() if v is False)
    missing_count = sum(1 for v in results.values() if v is None)

    for name, status in results.items():
        if status is True:
            symbol = "PASS"
        elif status is False:
            symbol = "FAIL"
        else:
            symbol = "N/A"
        print(f"  {CHECKPOINTS[name]['name']}: {symbol}")

    print(f"\nTotal: {success_count} passed, {fail_count} failed, {missing_count} not found")

    if fail_count > 0:
        print("\nSome downloads failed. Please check your network connection")
        print("and try again. If the problem persists, please report an issue.")
        return 1

    if success_count > 0:
        print("\nCheckpoints are ready. You can now run:")
        print("  python examples/quick_start.py")
        print("  make eval-quick")

    return 0


if __name__ == "__main__":
    sys.exit(main())
