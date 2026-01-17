#!/usr/bin/env python3
"""
[LEGACY] Initial Baseline Comparison Script

=============================================================================
DEPRECATION NOTICE
=============================================================================
This script was used during early development for initial
baseline experiments. It has been SUPERSEDED by:
  - scripts/evaluation/eval_baselines.py
  - scripts/evaluation/eval_main_results.py

Key differences from current implementation:
1. Uses different random seeds (legacy: 12345, current: 42)
2. Different train/test split ratios (legacy: 80/20, current: uses fixed splits)
3. No statistical significance testing
4. Manual metric computation (now uses src/evaluation/metrics.py)

=============================================================================

Status: DEPRECATED (see above)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def random_baseline(documents: List[Dict]) -> str:
    """
    Random baseline: Select a random document's answer.

    Historical Note: This was our initial sanity check baseline.
    """
    if not documents:
        return ""
    selected = random.choice(documents)
    return selected.get('answer', selected.get('content', ''))


def majority_baseline(documents: List[Dict]) -> str:
    """
    Majority baseline: Select most common answer.

    Historical Note: We expected this to be weak, but it performed
    surprisingly well on L1/L2 conflicts where recency correlates
    with majority.

    This observation led to the recency heuristic baseline.
    """
    if not documents:
        return ""

    answers = [doc.get('answer', '') for doc in documents if doc.get('answer')]
    if not answers:
        return documents[0].get('content', '')[:100]

    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def recency_baseline(documents: List[Dict]) -> str:
    """
    Recency baseline: Select most recent document.

    Historical Note: Added after EDA revealed strong temporal patterns
    in L1 conflicts. This became our strongest rule-based baseline.

    Key insight: Wikipedia edits that introduce conflicts often
    reflect updates to outdated information.
    """
    if not documents:
        return ""

    docs_with_time = [
        (doc, doc.get('timestamp', '1970-01-01'))
        for doc in documents
    ]

    docs_with_time.sort(key=lambda x: x[1], reverse=True)
    return docs_with_time[0][0].get('answer', docs_with_time[0][0].get('content', ''))


def bm25_score_baseline(documents: List[Dict]) -> str:
    """
    BM25 score baseline: Select highest-scored document.

    Historical Note: Initial hypothesis was that retrieval score
    correlates with answer quality. Results showed this is NOT
    the case for conflicts - high BM25 just means relevance,
    not correctness.
    """
    if not documents:
        return ""

    docs_with_score = [
        (doc, doc.get('relevance_score', 0.0))
        for doc in documents
    ]

    docs_with_score.sort(key=lambda x: x[1], reverse=True)
    return docs_with_score[0][0].get('answer', docs_with_score[0][0].get('content', ''))


def exact_match(pred: str, gold: str) -> bool:
    """Simple exact match (case-insensitive)."""
    return pred.strip().lower() == gold.strip().lower()


def fuzzy_match(pred: str, gold: str, threshold: float = 0.8) -> bool:
    """
    Fuzzy match using character overlap.

    Historical Note: We initially used this for evaluation but found
    it too lenient. Production uses exact match + manual review.
    """
    pred_chars = set(pred.lower())
    gold_chars = set(gold.lower())

    if not gold_chars:
        return not pred_chars

    overlap = len(pred_chars & gold_chars) / len(gold_chars)
    return overlap >= threshold


def evaluate_baseline(
    baseline_fn,
    data: List[Dict],
    use_fuzzy: bool = False
) -> Dict:
    """Evaluate a baseline function on dataset."""
    correct = 0
    total = 0
    by_type = Counter()
    correct_by_type = Counter()

    for example in data:
        documents = example.get('documents', [])
        gold = example.get('gold_answer', '')
        conflict_type = example.get('conflict_type', 'unknown')

        if not gold:
            continue

        pred = baseline_fn(documents)

        match_fn = fuzzy_match if use_fuzzy else exact_match
        is_correct = match_fn(pred, gold)

        total += 1
        by_type[conflict_type] += 1

        if is_correct:
            correct += 1
            correct_by_type[conflict_type] += 1

    results = {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'by_type': {}
    }

    for ctype, count in by_type.items():
        correct_count = correct_by_type.get(ctype, 0)
        results['by_type'][ctype] = {
            'total': count,
            'correct': correct_count,
            'accuracy': correct_count / count if count > 0 else 0
        }

    return results


def load_data(path: Path) -> List[Dict]:
    """Load JSONL data."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    """
    Main entry point for legacy baseline evaluation.

    Usage (LEGACY):
        python initial_baseline_comparison_v1.py --data data/dev/ --output results/

    Note: For current baseline evaluation, use:
        python scripts/evaluation/eval_baselines.py
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='[LEGACY] Initial baseline comparison'
    )
    parser.add_argument('--data', type=Path, default=Path('data/dev'),
                       help='Data directory')
    parser.add_argument('--output', type=Path, default=Path('results'),
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=12345,
                       help='Random seed (legacy default: 12345)')

    args = parser.parse_args()

    random.seed(args.seed)

    print("="*60)
    print("[LEGACY] Initial Baseline Comparison Script")
    print("WARNING: This script is deprecated. Use eval_baselines.py instead.")
    print("="*60)

    data_files = list(args.data.glob('*.jsonl'))
    all_data = []
    for f in data_files:
        all_data.extend(load_data(f))

    print(f"Loaded {len(all_data)} examples")

    baselines = {
        'random': random_baseline,
        'majority': majority_baseline,
        'recency': recency_baseline,
        'bm25_score': bm25_score_baseline,
    }

    results = {}
    for name, fn in baselines.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_baseline(fn, all_data)
        print(f"  Accuracy: {results[name]['accuracy']:.3f}")

    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / 'legacy_baseline_results_v1.json'

    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'script': 'initial_baseline_comparison_v1.py',
                'status': 'DEPRECATED',
                'seed': args.seed
            },
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nNote: These results may differ from paper due to:")
    print("  - Different random seed")
    print("  - Legacy data format")
    print("  - No significance testing")


if __name__ == '__main__':
    main()
