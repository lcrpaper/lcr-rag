#!/usr/bin/env python3
"""
Detector Calibration Script

Calibrates the conflict detector's probability outputs to improve
reliability of the detection threshold (tau).

Implements:
1. Temperature scaling (post-hoc calibration)
2. Platt scaling (logistic regression on logits)
3. Isotonic regression (non-parametric)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore', category=UserWarning)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Where:
    - B_b is the set of samples in bin b
    - acc(B_b) is accuracy in bin b
    - conf(B_b) is average confidence in bin b

    Lower ECE is better. Perfect calibration = 0.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = probs[in_bin].mean()
            avg_accuracy = labels[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)


def maximum_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE = max_b |acc(B_b) - conf(B_b)|

    Measures worst-case calibration error.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])

        if in_bin.sum() > 0:
            avg_confidence = probs[in_bin].mean()
            avg_accuracy = labels[in_bin].mean()
            mce = max(mce, abs(avg_accuracy - avg_confidence))

    return float(mce)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of probabilities).

    Brier = (1/n) * sum((p_i - y_i)^2)

    Lower is better. Perfect predictions = 0.
    """
    return float(np.mean((probs - labels) ** 2))


class TemperatureScaling:
    """
    Temperature scaling for calibration.

    Scales logits by learned temperature T:
        p_calibrated = softmax(logits / T)

    For binary classification:
        p_calibrated = sigmoid(logits / T)
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 1000):
        """Fit temperature parameter using NLL loss."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)

        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits_t / temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(temperature.item())

        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))

    def __repr__(self):
        return f"TemperatureScaling(T={self.temperature:.4f})"


class PlattScaling:
    """
    Platt scaling for calibration.

    Fits a logistic regression on the logits:
        p_calibrated = sigmoid(a * logits + b)
    """

    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """Fit Platt scaling parameters."""
        self.model.fit(logits.reshape(-1, 1), labels)
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]

    def __repr__(self):
        return f"PlattScaling(a={self.model.coef_[0][0]:.4f}, b={self.model.intercept_[0]:.4f})"


class IsotonicCalibration:
    """
    Isotonic regression for calibration.

    Non-parametric method that learns a monotonic mapping
    from probabilities to calibrated probabilities.

    Advantage: No parametric assumptions
    Disadvantage: Can overfit on small calibration sets
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """Fit isotonic regression."""
        self.model.fit(probs, labels)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        return self.model.predict(probs)

    def __repr__(self):
        return "IsotonicCalibration()"


def load_detector_outputs(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load detector outputs from file.

    Expected format (JSONL):
    {"logit": float, "prob": float, "label": int}
    """
    logits = []
    probs = []
    labels = []

    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                logits.append(data['logit'])
                probs.append(data['prob'])
                labels.append(data['label'])

    return np.array(logits), np.array(probs), np.array(labels)


def calibrate_detector(
    logits: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    method: str = 'temperature'
) -> Tuple[np.ndarray, object, Dict]:
    """
    Calibrate detector outputs.

    Args:
        logits: Raw logits from detector
        probs: Probability outputs (sigmoid of logits)
        labels: Ground truth labels
        method: Calibration method ('temperature', 'platt', 'isotonic')

    Returns:
        calibrated_probs: Calibrated probability outputs
        calibrator: Fitted calibrator object
        metrics: Calibration metrics before and after
    """
    metrics_before = {
        'ece': expected_calibration_error(probs, labels),
        'mce': maximum_calibration_error(probs, labels),
        'brier': brier_score(probs, labels)
    }

    if method == 'temperature':
        calibrator = TemperatureScaling().fit(logits, labels)
        calibrated_probs = calibrator.calibrate(logits)
    elif method == 'platt':
        calibrator = PlattScaling().fit(logits, labels)
        calibrated_probs = calibrator.calibrate(logits)
    elif method == 'isotonic':
        calibrator = IsotonicCalibration().fit(probs, labels)
        calibrated_probs = calibrator.calibrate(probs)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    metrics_after = {
        'ece': expected_calibration_error(calibrated_probs, labels),
        'mce': maximum_calibration_error(calibrated_probs, labels),
        'brier': brier_score(calibrated_probs, labels)
    }

    metrics = {
        'before': metrics_before,
        'after': metrics_after,
        'improvement': {
            'ece_reduction': metrics_before['ece'] - metrics_after['ece'],
            'mce_reduction': metrics_before['mce'] - metrics_after['mce'],
            'brier_reduction': metrics_before['brier'] - metrics_after['brier']
        }
    }

    return calibrated_probs, calibrator, metrics


def find_optimal_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, Dict]:
    """
    Find optimal detection threshold.

    Paper uses tau = 0.6 (determined by this analysis).
    """
    thresholds = np.arange(0.3, 0.9, 0.02)
    best_threshold = 0.5
    best_score = 0.0
    all_scores = {}

    for tau in thresholds:
        preds = (probs >= tau).astype(int)

        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(labels)

        scores = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
        all_scores[float(tau)] = scores

        if scores[metric] > best_score:
            best_score = scores[metric]
            best_threshold = tau

    return float(best_threshold), all_scores


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate conflict detector outputs'
    )
    parser.add_argument('--input', type=Path, required=True,
                       help='Path to detector outputs (JSONL)')
    parser.add_argument('--output', type=Path, default=Path('calibration_results.json'),
                       help='Output path for calibration results')
    parser.add_argument('--method', choices=['temperature', 'platt', 'isotonic', 'all'],
                       default='temperature', help='Calibration method')

    args = parser.parse_args()

    print("Loading detector outputs...")
    logits, probs, labels = load_detector_outputs(args.input)
    print(f"Loaded {len(labels)} examples")

    results = {
        'input_file': str(args.input),
        'n_examples': len(labels),
        'positive_rate': float(labels.mean())
    }

    methods = ['temperature', 'platt', 'isotonic'] if args.method == 'all' else [args.method]

    for method in methods:
        print(f"\nApplying {method} calibration...")
        calibrated_probs, calibrator, metrics = calibrate_detector(
            logits, probs, labels, method
        )

        print(f"  {calibrator}")
        print(f"  ECE: {metrics['before']['ece']:.4f} -> {metrics['after']['ece']:.4f}")
        print(f"  MCE: {metrics['before']['mce']:.4f} -> {metrics['after']['mce']:.4f}")
        print(f"  Brier: {metrics['before']['brier']:.4f} -> {metrics['after']['brier']:.4f}")

        results[method] = {
            'calibrator': str(calibrator),
            'metrics': metrics
        }

        optimal_tau, threshold_scores = find_optimal_threshold(calibrated_probs, labels)
        print(f"  Optimal threshold: {optimal_tau:.2f}")
        results[method]['optimal_threshold'] = optimal_tau

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
