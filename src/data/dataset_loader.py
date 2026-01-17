"""
Dataset Loaders for LCR Benchmarks

Loads JSONL datasets from data/benchmarks/ directory.
Supports all four conflict types (L1-L4) with train/dev/test splits.
"""

import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class ConflictExample:
    """Single example from LCR benchmarks."""
    query: str
    documents: List[str]
    answer: str
    conflict_type: str
    metadata: Dict

    def __post_init__(self):
        """Validate example structure."""
        assert len(self.documents) > 0, "Must have at least one document"
        assert self.conflict_type in ["L1", "L2", "L3", "L4"], \
            f"Invalid conflict type: {self.conflict_type}"


class LCRDataset(Dataset):
    """
    PyTorch dataset for LCR conflict benchmarks.

    Args:
        data_path: Path to JSONL file
        conflict_type: One of ['temporal', 'numerical', 'entity', 'semantic']
        max_docs: Maximum number of documents to use (default: 5)
    """

    def __init__(
        self,
        data_path: str,
        conflict_type: str,
        max_docs: int = 5
    ):
        self.data_path = Path(data_path)
        self.conflict_type = conflict_type
        self.max_docs = max_docs

        self.type_to_level = {
            'temporal': 'L1',
            'temp_conflict': 'L1',
            'numerical': 'L2',
            'num_conflict': 'L2',
            'entity': 'L3',
            'entity_conflict': 'L3',
            'semantic': 'L4',
            'semantic_conflict': 'L4'
        }

        self.level = self.type_to_level.get(conflict_type, conflict_type)

        self.examples = self._load_examples()

    def _load_examples(self) -> List[ConflictExample]:
        """Load examples from JSONL file."""
        examples = []

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        with jsonlines.open(self.data_path) as reader:
            for obj in reader:
                query = obj['query']
                documents = obj['documents'][:self.max_docs]
                answer = obj['answer']

                metadata = {
                    'conflict_level': obj.get('conflict_level', self.level),
                    'source': obj.get('source', 'unknown'),
                    'natural': obj.get('natural', True),
                    'augmented': obj.get('augmented', False)
                }

                example = ConflictExample(
                    query=query,
                    documents=documents,
                    answer=answer,
                    conflict_type=self.level,
                    metadata=metadata
                )
                examples.append(example)

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ConflictExample:
        return self.examples[idx]

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        natural_count = sum(1 for ex in self.examples if ex.metadata['natural'])
        augmented_count = sum(1 for ex in self.examples if ex.metadata['augmented'])

        return {
            'total_examples': len(self.examples),
            'natural_examples': natural_count,
            'augmented_examples': augmented_count,
            'natural_percentage': 100 * natural_count / len(self.examples),
            'conflict_type': self.conflict_type,
            'conflict_level': self.level,
            'avg_documents_per_query': sum(len(ex.documents) for ex in self.examples) / len(self.examples)
        }


class LCRDataModule:
    """
    Data module for managing train/dev/test splits across all conflict types.

    Args:
        data_root: Root directory containing benchmarks (default: data/benchmarks/)
        batch_size: Batch size for dataloaders
        max_docs: Maximum documents per query
    """

    def __init__(
        self,
        data_root: str = "data/benchmarks",
        batch_size: int = 16,
        max_docs: int = 5
    ):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.max_docs = max_docs

        self.conflict_types = [
            'temp_conflict',
            'num_conflict',
            'entity_conflict',
            'semantic_conflict'
        ]

        self.datasets = {}

    def setup(self, stage: Optional[str] = None):
        """
        Load datasets for specified stage.

        Args:
            stage: 'train', 'dev', 'test', or None for all
        """
        stages = ['train', 'dev', 'test'] if stage is None else [stage]

        for conflict_type in self.conflict_types:
            for split in stages:
                key = f"{conflict_type}_{split}"
                data_path = self.data_root / conflict_type / f"{split}.jsonl"

                if data_path.exists():
                    self.datasets[key] = LCRDataset(
                        data_path=str(data_path),
                        conflict_type=conflict_type,
                        max_docs=self.max_docs
                    )

    def get_dataloader(
        self,
        conflict_type: str,
        split: str,
        shuffle: bool = None
    ) -> DataLoader:
        """
        Get dataloader for specific conflict type and split.

        Args:
            conflict_type: One of temp_conflict, num_conflict, entity_conflict, semantic_conflict
            split: 'train', 'dev', or 'test'
            shuffle: Whether to shuffle (defaults: True for train, False otherwise)

        Returns:
            DataLoader
        """
        key = f"{conflict_type}_{split}"

        if key not in self.datasets:
            raise KeyError(f"Dataset {key} not loaded. Call setup() first.")

        if shuffle is None:
            shuffle = (split == 'train')

        return DataLoader(
            self.datasets[key],
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch: List[ConflictExample]) -> Dict:
        """Collate batch of examples."""
        return {
            'queries': [ex.query for ex in batch],
            'documents': [ex.documents for ex in batch],
            'answers': [ex.answer for ex in batch],
            'conflict_types': [ex.conflict_type for ex in batch],
            'metadata': [ex.metadata for ex in batch]
        }

    def get_all_statistics(self) -> Dict:
        """Get statistics for all loaded datasets."""
        stats = {}
        for key, dataset in self.datasets.items():
            stats[key] = dataset.get_statistics()

        total_stats = {
            'total_examples': sum(s['total_examples'] for s in stats.values()),
            'natural_examples': sum(s['natural_examples'] for s in stats.values()),
            'augmented_examples': sum(s['augmented_examples'] for s in stats.values()),
        }
        total_stats['natural_percentage'] = (
            100 * total_stats['natural_examples'] / total_stats['total_examples']
        )

        stats['overall'] = total_stats
        return stats

    def verify_dataset_sizes(self) -> Dict:
        """
        Verify dataset sizes match paper specifications.

        Expected from paper (Table, line 317-323):
            TempConflict:     train=4120, dev=412, test=823
            NumConflict:      train=2780, dev=278, test=687
            EntityConflict:   train=4540, dev=454, test=1142
            SemanticConflict: train=3210, dev=321, test=612
        """
        expected = {
            'temp_conflict_train': 4120,
            'temp_conflict_dev': 412,
            'temp_conflict_test': 823,
            'num_conflict_train': 2780,
            'num_conflict_dev': 278,
            'num_conflict_test': 687,
            'entity_conflict_train': 4540,
            'entity_conflict_dev': 454,
            'entity_conflict_test': 1142,
            'semantic_conflict_train': 3210,
            'semantic_conflict_dev': 321,
            'semantic_conflict_test': 612,
        }

        results = {}
        all_match = True

        for key, expected_size in expected.items():
            if key in self.datasets:
                actual_size = len(self.datasets[key])
                match = (actual_size == expected_size)
                all_match = all_match and match

                results[key] = {
                    'expected': expected_size,
                    'actual': actual_size,
                    'match': match,
                    'diff': actual_size - expected_size
                }
            else:
                results[key] = {
                    'expected': expected_size,
                    'actual': None,
                    'match': False,
                    'status': 'not_loaded'
                }
                all_match = False

        results['all_match'] = all_match
        return results


def demo_dataset_loader():
    """Demonstrate dataset loading and verification."""
    print("=" * 70)
    print("LCR Dataset Loader Demo")
    print("=" * 70)

    data_module = LCRDataModule(data_root="data/benchmarks")

    print("\nLoading all datasets...")
    data_module.setup()

    print("\nDataset Statistics:")
    print("-" * 70)
    stats = data_module.get_all_statistics()

    for key, stat in stats.items():
        if key == 'overall':
            print(f"\n{'Overall':20s}: {stat['total_examples']:5d} examples "
                  f"({stat['natural_percentage']:.1f}% natural)")
        elif 'total_examples' in stat:
            print(f"{key:20s}: {stat['total_examples']:5d} examples "
                  f"({stat['natural_percentage']:.1f}% natural)")

    print("\n\nDataset Size Verification:")
    print("-" * 70)
    verification = data_module.verify_dataset_sizes()

    for key, result in verification.items():
        if key != 'all_match' and result.get('expected'):
            status = "✓" if result['match'] else "✗"
            print(f"{status} {key:25s}: {result['actual']:5d} / {result['expected']:5d} "
                  f"(diff: {result['diff']:+4d})")

    print(f"\n{'All sizes match paper:':30s} {'✓ YES' if verification['all_match'] else '✗ NO'}")

    print("\n\nExample Batch:")
    print("-" * 70)
    dataloader = data_module.get_dataloader('temp_conflict', 'test', shuffle=False)
    batch = next(iter(dataloader))

    print(f"Batch size: {len(batch['queries'])}")
    print(f"\nFirst example:")
    print(f"  Query: {batch['queries'][0]}")
    print(f"  Documents: {len(batch['documents'][0])} docs")
    print(f"  Answer: {batch['answers'][0]}")
    print(f"  Conflict type: {batch['conflict_types'][0]}")
    print(f"  Natural: {batch['metadata'][0]['natural']}")


if __name__ == "__main__":
    demo_dataset_loader()
