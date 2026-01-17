# API Reference

This document provides the API reference for all public classes and functions in the LCR system.

## Core Models

### ConflictDetector

Binary classifier for detecting conflicts in retrieved documents.

```python
from src.models import ConflictDetector

class ConflictDetector(nn.Module):
    """
    2-layer MLP for binary conflict detection.

    Args:
        hidden_dim (int): Input dimension from LLM. Default: 4096
        intermediate_dim (int): Hidden layer dimension. Default: 512
        dropout (float): Dropout probability. Default: 0.1

    Attributes:
        fc1 (nn.Linear): First linear layer
        fc2 (nn.Linear): Output layer
        dropout (nn.Dropout): Dropout layer

    Example:
        >>> detector = ConflictDetector(hidden_dim=4096)
        >>> hidden_states = torch.randn(32, 4096)
        >>> probs = detector(hidden_states)
        >>> probs.shape
        torch.Size([32, 1])
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for conflict detection.

        Args:
            hidden_states: Tensor of shape (batch_size, hidden_dim)

        Returns:
            Tensor of shape (batch_size, 1) with conflict probabilities
        """
        pass

    @classmethod
    def from_pretrained(cls, path: str) -> "ConflictDetector":
        """
        Load a pretrained detector from checkpoint.

        Args:
            path: Path to checkpoint directory or file

        Returns:
            Loaded ConflictDetector instance
        """
        pass

    def save_pretrained(self, path: str) -> None:
        """
        Save detector to checkpoint.

        Args:
            path: Output path for checkpoint
        """
        pass
```

### RefinementModule

Iterative bottleneck refinement for conflict resolution.

```python
from src.models import RefinementModule

class RefinementModule(nn.Module):
    """
    Bottleneck MLP for iterative hidden state refinement.

    Args:
        hidden_dim (int): LLM hidden dimension. Default: 4096
        bottleneck_dim (int): Bottleneck dimension. Default: 732
        alpha (float): Interpolation coefficient. Default: 0.3
        num_iterations (int): Refinement iterations. Default: 3

    Example:
        >>> refiner = RefinementModule(hidden_dim=4096, alpha=0.3)
        >>> h0 = torch.randn(32, 4096)
        >>> h_refined = refiner(h0)
        >>> h_refined.shape
        torch.Size([32, 4096])
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_iterations: Optional[int] = None,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Perform iterative refinement.

        Args:
            hidden_states: Input hidden states (batch_size, hidden_dim)
            num_iterations: Override default iterations. Default: None
            return_intermediates: Return all intermediate states. Default: False

        Returns:
            If return_intermediates=False:
                Refined hidden states (batch_size, hidden_dim)
            If return_intermediates=True:
                Tuple of (final_states, list of intermediate states)
        """
        pass

    def single_iteration(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform single refinement iteration.

        Args:
            hidden_states: Current hidden states

        Returns:
            Updated hidden states after one iteration
        """
        pass
```

### TaxonomyClassifier

4-way conflict type classifier based on DeBERTa.

```python
from src.models import TaxonomyClassifier

class TaxonomyClassifier(nn.Module):
    """
    DeBERTa-based classifier for conflict taxonomy.

    Args:
        model_name (str): HuggingFace model identifier.
            Default: "microsoft/deberta-v3-large"
        num_classes (int): Number of conflict types. Default: 4
        max_length (int): Maximum sequence length. Default: 512

    Example:
        >>> classifier = TaxonomyClassifier()
        >>> query = "When was the treaty signed?"
        >>> docs = ["Doc 1 text...", "Doc 2 text..."]
        >>> pred = classifier.predict(query, docs)
        >>> pred
        {'class': 'L1_temporal', 'confidence': 0.89}
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass returning logits.

        Args:
            input_ids: Tokenized input (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Logits (batch_size, num_classes)
        """
        pass

    def predict(
        self,
        query: str,
        documents: List[str]
    ) -> Dict[str, Any]:
        """
        Predict conflict type for query-document pair.

        Args:
            query: User query string
            documents: List of retrieved document texts

        Returns:
            Dict with 'class' and 'confidence' keys
        """
        pass
```

### LCRSystem

Complete end-to-end system combining all components.

```python
from src.models import LCRSystem

class LCRSystem:
    """
    Complete LCR pipeline for conflict detection and resolution.

    Args:
        detector_path (str): Path to detector checkpoint
        classifier_path (str): Path to classifier checkpoint
        refinement_path (str): Path to refinement checkpoint
        base_model (str): Base LLM identifier
        device (str): Device to use. Default: "cuda"

    Example:
        >>> lcr = LCRSystem.from_pretrained("checkpoints/")
        >>> result = lcr.process(
        ...     query="Who is the CEO?",
        ...     documents=[doc1, doc2, doc3]
        ... )
        >>> result
        {
            'conflict_detected': True,
            'conflict_type': 'L1_temporal',
            'original_answer': '...',
            'refined_answer': '...',
            'confidence': 0.87
        }
    """

    def process(
        self,
        query: str,
        documents: List[str],
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Process query with retrieved documents.

        Args:
            query: User query
            documents: Retrieved documents
            return_details: Include intermediate outputs. Default: False

        Returns:
            Dict containing conflict info and answers
        """
        pass

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        **kwargs
    ) -> "LCRSystem":
        """
        Load complete system from checkpoint directory.

        Args:
            checkpoint_dir: Directory containing all checkpoints
            **kwargs: Additional arguments passed to components

        Returns:
            Loaded LCRSystem instance
        """
        pass
```

## Data Loading

### ConflictDataset

```python
from src.data import ConflictDataset

class ConflictDataset(torch.utils.data.Dataset):
    """
    Dataset for conflict detection and classification.

    Args:
        data_path (str): Path to JSONL data file
        tokenizer: HuggingFace tokenizer for text encoding
        max_length (int): Maximum sequence length. Default: 512
        include_labels (bool): Include labels for training. Default: True

    Example:
        >>> dataset = ConflictDataset(
        ...     "data/train/conflicts.jsonl",
        ...     tokenizer=tokenizer
        ... )
        >>> len(dataset)
        10200
        >>> sample = dataset[0]
        >>> sample.keys()
        dict_keys(['input_ids', 'attention_mask', 'labels'])
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single example."""
        pass

    def __len__(self) -> int:
        """Return dataset size."""
        pass
```

### DataCollator

```python
from src.data import ConflictDataCollator

class ConflictDataCollator:
    """
    Data collator for batching conflict examples.

    Args:
        tokenizer: Tokenizer for padding
        max_length (int): Maximum sequence length

    Example:
        >>> collator = ConflictDataCollator(tokenizer)
        >>> batch = collator([dataset[0], dataset[1]])
    """

    def __call__(
        self,
        examples: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Collate examples into batch."""
        pass
```

## Evaluation

### Metrics

```python
from src.evaluation import compute_metrics

def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: str = "detection"
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task: One of "detection", "classification", "refinement"

    Returns:
        Dict of metric names to values

    Example:
        >>> metrics = compute_metrics(preds, labels, task="detection")
        >>> metrics
        {'f1': 0.871, 'precision': 0.883, 'recall': 0.859, 'accuracy': 0.874}
    """
    pass

def compute_per_class_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics for multi-class classification.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        class_names: List of class names

    Returns:
        Nested dict of class -> metric -> value

    Example:
        >>> per_class = compute_per_class_metrics(preds, labels, ['L1', 'L2', 'L3', 'L4'])
        >>> per_class['L1']['f1']
        0.921
    """
    pass
```

## Training Utilities

### Trainer

```python
from src.training import Trainer

class Trainer:
    """
    Generic trainer for LCR components.

    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Training configuration dict
        output_dir (str): Output directory for checkpoints

    Example:
        >>> trainer = Trainer(
        ...     model=detector,
        ...     train_dataset=train_ds,
        ...     eval_dataset=dev_ds,
        ...     config=config,
        ...     output_dir="checkpoints/detector/"
        ... )
        >>> trainer.train()
    """

    def train(self) -> Dict[str, Any]:
        """
        Run training loop.

        Returns:
            Dict with training history and best metrics
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on eval_dataset.

        Returns:
            Dict of evaluation metrics
        """
        pass
```

## Configuration

### Config Loading

```python
from src.utils import load_config, merge_configs

def load_config(path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML config

    Returns:
        Configuration dict
    """
    pass

def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Merge two configurations with override taking precedence.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    pass
```

## Error Classes

```python
class LCRError(Exception):
    """Base exception for LCR errors."""
    pass

class ModelNotFoundError(LCRError):
    """Raised when checkpoint is not found."""
    pass

class ConfigurationError(LCRError):
    """Raised for invalid configuration."""
    pass

class DataError(LCRError):
    """Raised for data loading errors."""
    pass
```
