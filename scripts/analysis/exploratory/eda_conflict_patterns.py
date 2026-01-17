#!/usr/bin/env python3
"""
Exploratory Data Analysis: Conflict Pattern Discovery

This script performs initial exploration of the conflict dataset to:
1. Identify common conflict patterns by type
2. Analyze lexical overlap between conflicting documents
3. Discover temporal patterns in L1 conflicts
4. Profile numerical distribution in L2 conflicts

Status: EXPLORATORY

Note: This is exploratory code - not optimized for production.
Some analyses were superseded by more rigorous methods.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import argparse

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, some analyses disabled")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("Warning: nltk not available, using basic tokenization")


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def simple_tokenize(text: str) -> List[str]:
    """Basic tokenization fallback."""
    return re.findall(r'\b\w+\b', text.lower())


def tokenize(text: str) -> List[str]:
    """Tokenize text, with fallback."""
    if HAS_NLTK:
        return word_tokenize(text.lower())
    return simple_tokenize(text)


def compute_jaccard_similarity(doc1: str, doc2: str) -> float:
    """Compute Jaccard similarity between two documents."""
    tokens1 = set(tokenize(doc1))
    tokens2 = set(tokenize(doc2))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def analyze_lexical_overlap(data: List[Dict]) -> Dict:
    """Analyze lexical overlap between conflicting documents."""
    print("Analyzing lexical overlap...")

    overlap_by_type = defaultdict(list)

    for example in data:
        if len(example.get('documents', [])) < 2:
            continue

        doc1_text = example['documents'][0].get('content', '')
        doc2_text = example['documents'][1].get('content', '')

        similarity = compute_jaccard_similarity(doc1_text, doc2_text)
        conflict_type = example.get('conflict_type', 'unknown')
        overlap_by_type[conflict_type].append(similarity)

    results = {}
    for ctype, similarities in overlap_by_type.items():
        if HAS_NUMPY:
            results[ctype] = {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'count': len(similarities)
            }
        else:
            results[ctype] = {
                'mean': sum(similarities) / len(similarities),
                'count': len(similarities)
            }

    return results


DATE_PATTERNS = [
    r'\b(19|20)\d{2}\b',
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2}\b',
    r'\b\d{1,2}[-/]\d{1,2}[-/](19|20)?\d{2}\b',
    r'\b(19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b',
]


def extract_years(text: str) -> List[int]:
    """Extract year mentions from text."""
    years = []
    for match in re.findall(r'\b(19|20)\d{2}\b', text):
        years.append(int(match + match[-2:]) if len(match) == 2 else int(text[text.find(match):text.find(match)+4]))

    year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
    return [int(y) for y in year_matches]


def analyze_temporal_patterns(data: List[Dict]) -> Dict:
    """Analyze temporal patterns in L1 conflicts."""
    print("Analyzing temporal patterns...")

    l1_data = [ex for ex in data if ex.get('conflict_type') == 'L1_temporal']

    year_spans = []
    decade_conflicts = Counter()

    for example in l1_data:
        all_years = []
        for doc in example.get('documents', []):
            years = extract_years(doc.get('content', ''))
            all_years.extend(years)

        if len(all_years) >= 2:
            span = max(all_years) - min(all_years)
            year_spans.append(span)

            for year in all_years:
                decade = (year // 10) * 10
                decade_conflicts[decade] += 1

    results = {
        'total_l1_examples': len(l1_data),
        'decade_distribution': dict(decade_conflicts.most_common(10)),
    }

    if HAS_NUMPY and year_spans:
        results['year_span'] = {
            'mean': float(np.mean(year_spans)),
            'median': float(np.median(year_spans)),
            'std': float(np.std(year_spans)),
            'max': int(max(year_spans))
        }

    return results


NUMBER_PATTERN = r'(?:[\$€£]?\s*)(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:million|billion|trillion|thousand|%|percent)?'


def extract_numbers(text: str) -> List[float]:
    """Extract numerical values from text."""
    numbers = []

    matches = re.findall(NUMBER_PATTERN, text, re.IGNORECASE)

    for match in matches:
        try:
            num_str = match.replace(',', '')
            num = float(num_str)

            numbers.append(num)
        except ValueError:
            continue

    return numbers


def analyze_numerical_patterns(data: List[Dict]) -> Dict:
    """Analyze numerical patterns in L2 conflicts."""
    print("Analyzing numerical patterns...")

    l2_data = [ex for ex in data if ex.get('conflict_type') == 'L2_numerical']

    value_diffs = []
    relative_diffs = []

    for example in l2_data:
        all_numbers = []
        for doc in example.get('documents', []):
            numbers = extract_numbers(doc.get('content', ''))
            all_numbers.append(numbers)

        if len(all_numbers) >= 2 and all_numbers[0] and all_numbers[1]:
            n1 = all_numbers[0][0]
            n2 = all_numbers[1][0]

            abs_diff = abs(n1 - n2)
            value_diffs.append(abs_diff)

            if max(n1, n2) > 0:
                rel_diff = abs_diff / max(n1, n2)
                relative_diffs.append(rel_diff)

    results = {
        'total_l2_examples': len(l2_data),
        'examples_with_numbers': len(value_diffs)
    }

    if HAS_NUMPY and relative_diffs:
        results['relative_difference'] = {
            'mean': float(np.mean(relative_diffs)),
            'median': float(np.median(relative_diffs)),
            'std': float(np.std(relative_diffs))
        }

        bins = [0, 0.1, 0.25, 0.5, 1.0, float('inf')]
        bin_labels = ['<10%', '10-25%', '25-50%', '50-100%', '>100%']
        hist, _ = np.histogram(relative_diffs, bins=bins)
        results['relative_diff_bins'] = dict(zip(bin_labels, hist.tolist()))

    return results


def analyze_query_patterns(data: List[Dict]) -> Dict:
    """Analyze common query patterns."""
    print("Analyzing query patterns...")

    query_starters = Counter()
    query_lengths = []
    wh_questions = Counter()

    for example in data:
        query = example.get('query', '')
        tokens = tokenize(query)

        if tokens:
            query_starters[tokens[0]] += 1
            query_lengths.append(len(tokens))

            first_word = tokens[0].lower()
            if first_word in ['what', 'when', 'where', 'who', 'which', 'how', 'why']:
                wh_questions[first_word] += 1

    results = {
        'top_query_starters': dict(query_starters.most_common(10)),
        'wh_question_distribution': dict(wh_questions.most_common()),
    }

    if HAS_NUMPY and query_lengths:
        results['query_length'] = {
            'mean': float(np.mean(query_lengths)),
            'median': float(np.median(query_lengths)),
            'std': float(np.std(query_lengths))
        }

    return results


def run_exploration(data_path: Path, output_path: Optional[Path] = None):
    """Run full exploratory analysis."""
    print(f"Loading data from {data_path}...")
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} examples")

    results = {
        'metadata': {
            'data_path': str(data_path),
            'total_examples': len(data)
        }
    }

    results['lexical_overlap'] = analyze_lexical_overlap(data)
    results['temporal_patterns'] = analyze_temporal_patterns(data)
    results['numerical_patterns'] = analyze_numerical_patterns(data)
    results['query_patterns'] = analyze_query_patterns(data)

    print("\n" + "="*60)
    print("EXPLORATORY ANALYSIS SUMMARY")
    print("="*60)

    print("\nLexical Overlap by Conflict Type:")
    for ctype, stats in results['lexical_overlap'].items():
        print(f"  {ctype}: mean={stats['mean']:.3f}")

    if 'temporal_patterns' in results:
        tp = results['temporal_patterns']
        print(f"\nTemporal Patterns (L1): {tp['total_l1_examples']} examples")
        if 'year_span' in tp:
            print(f"  Year span: mean={tp['year_span']['mean']:.1f}, median={tp['year_span']['median']:.1f}")

    if 'numerical_patterns' in results:
        np_res = results['numerical_patterns']
        print(f"\nNumerical Patterns (L2): {np_res['total_l2_examples']} examples")
        if 'relative_difference' in np_res:
            print(f"  Relative diff: mean={np_res['relative_difference']['mean']:.3f}")

    print("\nQuery Patterns:")
    qp = results['query_patterns']
    print(f"  Top starters: {list(qp['top_query_starters'].keys())[:5]}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Exploratory data analysis for conflict dataset'
    )
    parser.add_argument('--data', type=Path, required=True,
                       help='Path to JSONL data file')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output path for results JSON')

    args = parser.parse_args()
    run_exploration(args.data, args.output)


if __name__ == '__main__':
    main()
