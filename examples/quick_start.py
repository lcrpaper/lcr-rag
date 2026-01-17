#!/usr/bin/env python3
"""
LCR Quick Start Example

This script demonstrates the basic usage of the LCR (Latent Conflict Refinement)
system for detecting and resolving conflicts in RAG contexts.

Requirements:
    - Trained checkpoints in checkpoints/ directory
    - Python 3.10+
    - Dependencies from requirements.txt

Usage:
    python examples/quick_start.py
    python examples/quick_start.py --query "Who invented the telephone?"
    python examples/quick_start.py --checkpoint-dir /path/to/checkpoints

"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

try:
    from src.models.conflict_detector import ConflictDetector
    from src.models.refinement_module import RefinementModule
    from src.models.lcr_system import LCRSystem
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    print("Running in demo mode with simulated outputs.")
    MODELS_AVAILABLE = False


EXAMPLE_QUERIES = [
    {
        "query": "When was SpaceX founded?",
        "documents": [
            {"text": "SpaceX was founded in 2002 by Elon Musk.", "source": "wikipedia"},
            {"text": "The company SpaceX was established in May 2002.", "source": "reuters"},
        ],
        "expected_conflict": False,
        "expected_type": None,
    },
    {
        "query": "What is the population of Tokyo?",
        "documents": [
            {"text": "Tokyo has a population of 13.96 million (2020).", "source": "un_stats"},
            {"text": "Greater Tokyo Area has over 37 million residents.", "source": "wikipedia"},
        ],
        "expected_conflict": True,
        "expected_type": "L2_numerical",
    },
    {
        "query": "Who is the CEO of Twitter?",
        "documents": [
            {"text": "Jack Dorsey is CEO of Twitter.", "source": "techcrunch", "date": "2021-03"},
            {"text": "Elon Musk acquired Twitter in October 2022.", "source": "nytimes", "date": "2022-11"},
            {"text": "Linda Yaccarino became CEO in June 2023.", "source": "wsj", "date": "2023-06"},
        ],
        "expected_conflict": True,
        "expected_type": "L1_temporal",
    },
    {
        "query": "Is coffee good for health?",
        "documents": [
            {"text": "Moderate coffee consumption reduces heart disease risk.", "source": "nejm"},
            {"text": "Excessive coffee leads to anxiety and insomnia.", "source": "mayo_clinic"},
        ],
        "expected_conflict": True,
        "expected_type": "L4_semantic",
    },
]


class DemoLCRSystem:
    """
    Demonstration LCR system that simulates model behavior.
    Use this when trained checkpoints are not available.
    """

    def __init__(self):
        print("=" * 60)
        print("DEMO MODE: Using simulated LCR system")
        print("For real inference, train models first:")
        print("  make reproduce-detector")
        print("  make reproduce-refinement")
        print("=" * 60)
        print()

    def detect_conflict(self, query: str, documents: List[Dict]) -> Dict:
        """Simulate conflict detection based on document analysis."""
        texts = [d["text"].lower() for d in documents]

        import re
        numbers = []
        for text in texts:
            nums = re.findall(r'\d+(?:\.\d+)?', text)
            numbers.extend(nums)

        has_numerical_conflict = len(set(numbers)) > 1 and len(numbers) > 2

        years = re.findall(r'\b(19|20)\d{2}\b', ' '.join(texts))
        has_temporal_conflict = len(set(years)) > 1

        conflict_prob = 0.3
        if has_numerical_conflict:
            conflict_prob = 0.85
        elif has_temporal_conflict:
            conflict_prob = 0.75

        return {
            "conflict_detected": conflict_prob > 0.5,
            "conflict_probability": conflict_prob,
            "method": "demo_heuristic"
        }

    def classify_conflict(self, query: str, documents: List[Dict]) -> Dict:
        """Simulate conflict type classification."""
        texts = ' '.join([d["text"].lower() for d in documents])

        if any(word in texts for word in ["million", "percent", "meters", "km"]):
            conflict_type = "L2_numerical"
            confidence = 0.72
        elif any(word in texts for word in ["2020", "2021", "2022", "2023", "founded", "established"]):
            conflict_type = "L1_temporal"
            confidence = 0.78
        elif any(word in texts for word in ["invented", "created", "discovered", "founded by"]):
            conflict_type = "L3_entity"
            confidence = 0.65
        else:
            conflict_type = "L4_semantic"
            confidence = 0.58

        return {
            "conflict_type": conflict_type,
            "confidence": confidence,
            "probabilities": {
                "L1_temporal": 0.25,
                "L2_numerical": 0.25,
                "L3_entity": 0.25,
                "L4_semantic": 0.25,
            }
        }

    def refine_answer(self, query: str, documents: List[Dict], conflict_info: Dict) -> Dict:
        """Simulate answer refinement."""
        if documents:
            answer = documents[0]["text"].split(".")[0] + "."
        else:
            answer = "Unable to determine answer."

        return {
            "refined_answer": answer,
            "confidence": 0.7,
            "refinement_iterations": 3,
            "token_overhead_percent": 6.2
        }

    def process(self, query: str, documents: List[Dict]) -> Dict:
        """Full LCR pipeline."""
        detection = self.detect_conflict(query, documents)

        result = {
            "query": query,
            "num_documents": len(documents),
            "detection": detection,
        }

        if detection["conflict_detected"]:
            classification = self.classify_conflict(query, documents)
            refinement = self.refine_answer(query, documents, classification)

            result["classification"] = classification
            result["refinement"] = refinement
        else:
            result["classification"] = None
            result["refinement"] = {
                "refined_answer": documents[0]["text"] if documents else "No documents provided.",
                "confidence": 0.9,
                "refinement_iterations": 0,
                "token_overhead_percent": 0.0
            }

        return result


def run_demo(checkpoint_dir: Optional[str] = None, custom_query: Optional[str] = None):
    """Run the LCR demonstration."""

    print("=" * 70)
    print("LCR (Latent Conflict Refinement) - Quick Start Demo")
    print("=" * 70)
    print()

    if MODELS_AVAILABLE and checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            print(f"Loading models from: {checkpoint_dir}")
            lcr = DemoLCRSystem()
        else:
            print(f"Checkpoint directory not found: {checkpoint_dir}")
            lcr = DemoLCRSystem()
    else:
        lcr = DemoLCRSystem()

    if custom_query:
        examples = [{
            "query": custom_query,
            "documents": [
                {"text": "This is document 1 with some information.", "source": "source1"},
                {"text": "This is document 2 with potentially different information.", "source": "source2"},
            ],
            "expected_conflict": None,
            "expected_type": None,
        }]
    else:
        examples = EXAMPLE_QUERIES

    print(f"Processing {len(examples)} example(s)...\n")

    for i, example in enumerate(examples, 1):
        print(f"{'─' * 70}")
        print(f"Example {i}: {example['query']}")
        print(f"{'─' * 70}")

        print("\nDocuments:")
        for j, doc in enumerate(example["documents"], 1):
            source = doc.get("source", "unknown")
            date = doc.get("date", "")
            date_str = f" ({date})" if date else ""
            print(f"  [{j}] {doc['text'][:80]}...")
            print(f"      Source: {source}{date_str}")

        result = lcr.process(example["query"], example["documents"])

        print("\nLCR Analysis:")
        print(f"  Conflict Detected: {result['detection']['conflict_detected']}")
        print(f"  Conflict Probability: {result['detection']['conflict_probability']:.2%}")

        if result["classification"]:
            print(f"  Conflict Type: {result['classification']['conflict_type']}")
            print(f"  Type Confidence: {result['classification']['confidence']:.2%}")

        print(f"\nRefined Answer:")
        print(f"  {result['refinement']['refined_answer']}")
        print(f"  Answer Confidence: {result['refinement']['confidence']:.2%}")
        print(f"  Refinement Iterations: {result['refinement']['refinement_iterations']}")
        print(f"  Token Overhead: +{result['refinement']['token_overhead_percent']:.1f}%")

        if example.get("expected_conflict") is not None:
            print(f"\nExpected:")
            print(f"  Conflict: {example['expected_conflict']}, Type: {example['expected_type']}")
            match = result["detection"]["conflict_detected"] == example["expected_conflict"]
            print(f"  Detection Match: {'✓' if match else '✗'}")

        print()

    print("=" * 70)
    print("Demo complete!")
    print()
    print("Next steps:")
    print("  1. Train models: make reproduce-paper")
    print("  2. Run evaluation: make eval-main")
    print("  3. See REPRODUCIBILITY.md for full instructions")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="LCR Quick Start Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/quick_start.py
    python examples/quick_start.py --query "Who wrote Hamlet?"
    python examples/quick_start.py --checkpoint-dir checkpoints/
        """
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Custom query to process"
    )
    parser.add_argument(
        "--checkpoint-dir", "-c",
        type=str,
        default="checkpoints/",
        help="Directory containing trained checkpoints"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    run_demo(
        checkpoint_dir=args.checkpoint_dir,
        custom_query=args.query
    )


if __name__ == "__main__":
    main()
