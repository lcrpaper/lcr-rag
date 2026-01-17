#!/usr/bin/env python3
"""
Memory Profiling Utilities

Tools for profiling GPU and CPU memory usage during training and inference.

Usage:
    python scripts/profiling/memory_profiler.py --model detector --batch-size 32
    python scripts/profiling/memory_profiler.py --trace training_trace.json
"""

import gc
import os
import sys
import time
import json
import argparse
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.cuda as cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import nvidia_ml_py3.pynvml as pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except:
    HAS_NVML = False

try:
    from memory_profiler import profile as mem_profile
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage."""
    timestamp: float
    label: str

    gpu_allocated: int = 0
    gpu_reserved: int = 0
    gpu_max_allocated: int = 0
    gpu_total: int = 0

    cpu_used: int = 0
    cpu_available: int = 0
    cpu_percent: float = 0.0

    process_rss: int = 0
    process_vms: int = 0

    cuda_device: int = 0
    notes: str = ""


class MemoryProfiler:
    """
    Memory profiler for deep learning workloads.

    Tracks GPU and CPU memory usage over time.
    """

    def __init__(
        self,
        device: int = 0,
        track_gpu: bool = True,
        track_cpu: bool = True,
        track_process: bool = True,
    ):
        self.device = device
        self.track_gpu = track_gpu and HAS_TORCH and torch.cuda.is_available()
        self.track_cpu = track_cpu and HAS_PSUTIL
        self.track_process = track_process and HAS_PSUTIL

        self.snapshots: List[MemorySnapshot] = []
        self.start_time = time.time()

        if self.track_gpu:
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()

    def _get_gpu_memory(self) -> Dict[str, int]:
        """Get GPU memory statistics."""
        if not self.track_gpu:
            return {}

        return {
            'gpu_allocated': torch.cuda.memory_allocated(self.device),
            'gpu_reserved': torch.cuda.memory_reserved(self.device),
            'gpu_max_allocated': torch.cuda.max_memory_allocated(self.device),
            'gpu_total': torch.cuda.get_device_properties(self.device).total_memory,
        }

    def _get_cpu_memory(self) -> Dict[str, Any]:
        """Get CPU memory statistics."""
        if not self.track_cpu:
            return {}

        mem = psutil.virtual_memory()
        return {
            'cpu_used': mem.used,
            'cpu_available': mem.available,
            'cpu_percent': mem.percent,
        }

    def _get_process_memory(self) -> Dict[str, int]:
        """Get process memory statistics."""
        if not self.track_process:
            return {}

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            'process_rss': mem_info.rss,
            'process_vms': mem_info.vms,
        }

    def snapshot(self, label: str = "", notes: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        snap = MemorySnapshot(
            timestamp=time.time() - self.start_time,
            label=label,
            cuda_device=self.device,
            notes=notes,
            **self._get_gpu_memory(),
            **self._get_cpu_memory(),
            **self._get_process_memory(),
        )

        self.snapshots.append(snap)
        return snap

    @contextmanager
    def track(self, label: str):
        """Context manager for tracking a code block."""
        self.snapshot(f"{label}_start")
        yield
        self.snapshot(f"{label}_end")

    def reset_peak_memory(self):
        """Reset peak memory tracking."""
        if self.track_gpu:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_peak_gpu_memory(self) -> int:
        """Get peak GPU memory usage."""
        if not self.track_gpu:
            return 0
        return torch.cuda.max_memory_allocated(self.device)

    def summarize(self) -> Dict[str, Any]:
        """Summarize profiling results."""
        if not self.snapshots:
            return {}

        gpu_allocated = [s.gpu_allocated for s in self.snapshots]
        process_rss = [s.process_rss for s in self.snapshots]

        return {
            'num_snapshots': len(self.snapshots),
            'duration_seconds': self.snapshots[-1].timestamp,
            'gpu_peak_allocated_mb': max(gpu_allocated) / 1024**2 if gpu_allocated else 0,
            'gpu_peak_reserved_mb': max(s.gpu_reserved for s in self.snapshots) / 1024**2,
            'process_peak_rss_mb': max(process_rss) / 1024**2 if process_rss else 0,
        }

    def to_json(self, path: str):
        """Export snapshots to JSON."""
        data = {
            'metadata': {
                'device': self.device,
                'start_time': self.start_time,
            },
            'snapshots': [asdict(s) for s in self.snapshots],
            'summary': self.summarize(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        """Print summary to console."""
        summary = self.summarize()

        print("\n" + "=" * 50)
        print("MEMORY PROFILING SUMMARY")
        print("=" * 50)
        print(f"Duration: {summary.get('duration_seconds', 0):.2f}s")
        print(f"GPU Peak Allocated: {summary.get('gpu_peak_allocated_mb', 0):.1f} MB")
        print(f"GPU Peak Reserved: {summary.get('gpu_peak_reserved_mb', 0):.1f} MB")
        print(f"Process Peak RSS: {summary.get('process_peak_rss_mb', 0):.1f} MB")
        print("=" * 50)


def profile_model_memory(
    model_name: str,
    batch_size: int = 32,
    seq_length: int = 512,
    num_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Profile memory usage for a model.

    Args:
        model_name: Name of model component (detector, refinement, classifier)
        batch_size: Batch size
        seq_length: Sequence length
        num_iterations: Number of forward passes

    Returns:
        Profiling results
    """
    profiler = MemoryProfiler()

    profiler.snapshot("before_import")

    if model_name == "detector":
        from src.models.conflict_detector import ConflictDetector
        model = ConflictDetector().cuda()
        input_shape = (batch_size, 4096)
    elif model_name == "refinement":
        from src.models.refinement_module import RefinementModule
        model = RefinementModule().cuda()
        input_shape = (batch_size, 4096)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    profiler.snapshot("after_model_load")

    x = torch.randn(*input_shape, device='cuda')
    _ = model(x)
    torch.cuda.synchronize()
    profiler.snapshot("after_warmup")

    profiler.reset_peak_memory()

    for i in range(num_iterations):
        x = torch.randn(*input_shape, device='cuda')

        with profiler.track(f"forward_{i}"):
            _ = model(x)
            torch.cuda.synchronize()

    profiler.snapshot("final")
    profiler.print_summary()

    return profiler.summarize()


def main():
    parser = argparse.ArgumentParser(description='Memory Profiler')
    parser.add_argument('--model', choices=['detector', 'refinement', 'classifier'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-length', type=int, default=512)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--trace', help='Load and analyze existing trace')

    args = parser.parse_args()

    if args.trace:
        with open(args.trace) as f:
            data = json.load(f)
        print(json.dumps(data['summary'], indent=2))
    elif args.model:
        results = profile_model_memory(
            args.model,
            args.batch_size,
            args.seq_length,
            args.iterations
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
