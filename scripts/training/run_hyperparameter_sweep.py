#!/usr/bin/env python3
"""
Hyperparameter Sweep Script

Runs systematic hyperparameter search for detector/refinement modules.
Documents exploration process typical of research projects.

Usage:
    python scripts/training/run_hyperparameter_sweep.py --model detector
    python scripts/training/run_hyperparameter_sweep.py --model refinement --quick
"""

import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class HyperparameterSweep:
    """Manages hyperparameter search experiments."""

    DETECTOR_SPACE = {
        'learning_rate': [1e-5, 2e-5, 5e-5],
        'batch_size': [16, 32, 64],
        'dropout': [0.1, 0.2, 0.3],
        'hidden_multiplier': [1, 2, 4]
    }

    REFINEMENT_SPACE = {
        'learning_rate': [1e-5, 5e-5, 1e-4],
        'alpha': [0.1, 0.3, 0.5],
        't_max': [1, 3, 5],
        'bottleneck_dim': [256, 384, 512]
    }

    def __init__(self, model_type: str, output_dir: str = 'experiments/sweeps'):
        self.model_type = model_type
        self.output_dir = Path(output_dir) / f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.search_space = (
            self.DETECTOR_SPACE if model_type == 'detector'
            else self.REFINEMENT_SPACE
        )

    def generate_configs(self, quick: bool = False) -> List[Dict[str, Any]]:
        """
        Generate hyperparameter configurations.

        Args:
            quick: If True, use smaller search space for testing
        """
        if quick:
            if self.model_type == 'detector':
                space = {
                    'learning_rate': [1e-5, 2e-5],
                    'batch_size': [32],
                    'dropout': [0.1, 0.2],
                    'hidden_multiplier': [2]
                }
            else:
                space = {
                    'learning_rate': [1e-5, 5e-5],
                    'alpha': [0.3],
                    't_max': [1, 3],
                    'bottleneck_dim': [384]
                }
        else:
            space = self.search_space

        keys = list(space.keys())
        values = [space[k] for k in keys]
        configs = []

        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)

        logger.info(f"Generated {len(configs)} configurations to try")
        return configs

    def run_sweep(self, quick: bool = False, max_experiments: int = None):
        """Run full hyperparameter sweep."""
        configs = self.generate_configs(quick)

        if max_experiments:
            configs = configs[:max_experiments]

        logger.info("="*80)
        logger.info(f"HYPERPARAMETER SWEEP: {self.model_type.upper()}")
        logger.info(f"Total experiments: {len(configs)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*80)

        all_results = []

        for i, config in enumerate(configs, 1):

            experiment = {
                'experiment_id': i,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(experiment)

            exp_file = self.output_dir / f"experiment_{i:03d}.json"
            with open(exp_file, 'w') as f:
                json.dump(experiment, f, indent=2)

        if self.model_type == 'detector':
            metric_key = 'val_f1'
        else:
            metric_key = 'val_accuracy'

        best_exp = max(all_results, key=lambda x: x['results'][metric_key])

        summary = {
            'model_type': self.model_type,
            'total_experiments': len(all_results),
            'search_space': self.search_space,
            'best_experiment': best_exp,
            'all_results': all_results
        }

        summary_file = self.output_dir / 'sweep_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "="*80)
        logger.info("SWEEP COMPLETE")
        logger.info("="*80)
        logger.info(f"Best experiment: #{best_exp['experiment_id']}")
        logger.info(f"Best config: {best_exp['config']}")
        logger.info(f"Best {metric_key}: {best_exp['results'][metric_key]:.4f}")
        logger.info(f"\nResults saved to: {summary_file}")
        logger.info("="*80)

        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run hyperparameter sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full detector sweep (3^4 = 81 experiments)
  python scripts/training/run_hyperparameter_sweep.py --model detector

  # Quick refinement sweep (testing, ~8 experiments)
  python scripts/training/run_hyperparameter_sweep.py --model refinement --quick

  # Limited sweep
  python scripts/training/run_hyperparameter_sweep.py --model detector --max 10
        """
    )

    parser.add_argument(
        '--model',
        required=True,
        choices=['detector', 'refinement'],
        help='Which model to sweep'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: reduced search space for testing'
    )
    parser.add_argument(
        '--max',
        type=int,
        help='Maximum number of experiments to run'
    )
    parser.add_argument(
        '--output-dir',
        default='experiments/sweeps',
        help='Output directory for sweep results'
    )

    args = parser.parse_args()

    sweeper = HyperparameterSweep(
        model_type=args.model,
        output_dir=args.output_dir
    )

    summary = sweeper.run_sweep(
        quick=args.quick,
        max_experiments=args.max
    )

    logger.info("\n✅ Hyperparameter sweep complete!")


if __name__ == '__main__':
    main()
