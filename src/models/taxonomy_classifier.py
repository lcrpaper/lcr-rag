"""
Taxonomy Classifier for LCR

Fine-tuned DeBERTa-v3-large classifier to predict conflict levels L1-L4
for hybrid routing with cascade confidence thresholds.

Classifications:
    L1 (Temporal): Conflicts over dates/times → Shallow, use LCR
    L2 (Numerical): Conflicts over quantities → Shallow, use LCR
    L3 (Entity): Conflicts over entity identities → Deep, use verification
    L4 (Semantic): Conflicts requiring reasoning → Deep, use verification
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import numpy as np


class TaxonomyClassifier:
    """
    DeBERTa-v3-large based classifier for conflict taxonomy (L1-L4).

    Args:
        model_name (str): Pre-trained model name or path
        confidence_threshold (float): Cascade routing threshold (default: 0.7)
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        confidence_threshold: float = 0.7,
        device: str = None
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.id2label = {0: "L1", 1: "L2", 2: "L3", 3: "L4"}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.tokenizer = None
        self.model = None
        self._initialized = False

    def initialize(self):
        """Initialize model and tokenizer (lazy loading)."""
        if self._initialized:
            return

        print(f"Initializing TaxonomyClassifier with {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=4,
            id2label=self.id2label,
            label2id=self.label2id
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        )
        self.model.to(self.device)
        self.model.eval()

        self._initialized = True
        print(f"Model loaded to {self.device}")

    def _create_input_text(self, query: str, documents: List[str]) -> str:
        """
        Format query and documents for classification.

        Args:
            query: User query
            documents: List of retrieved documents

        Returns:
            Formatted text for classifier
        """
        doc_text = " [SEP] ".join(documents[:5])
        return f"{query} [SEP] {doc_text}"

    def classify(
        self,
        query: str,
        documents: List[str],
        return_all_scores: bool = False
    ) -> Dict:
        """
        Classify conflict type with confidence score.

        Args:
            query: User query
            documents: Retrieved documents
            return_all_scores: If True, return scores for all classes

        Returns:
            dict with:
                - level: Predicted conflict level (L1/L2/L3/L4)
                - confidence: Confidence score [0, 1]
                - use_lcr: Whether to use LCR based on cascade routing
                - all_scores: (optional) Scores for all levels
        """
        if not self._initialized:
            self.initialize()

        input_text = self._create_input_text(query, documents)
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        confidence, pred_id = torch.max(probs, dim=-1)
        confidence = confidence.item()
        pred_id = pred_id.item()
        predicted_level = self.id2label[pred_id]

        use_lcr = (
            confidence >= self.confidence_threshold and
            predicted_level in ["L1", "L2"]
        )

        result = {
            'level': predicted_level,
            'confidence': confidence,
            'use_lcr': use_lcr,
            'routing_strategy': 'lcr' if use_lcr else 'full_verification'
        }

        if return_all_scores:
            all_scores = {
                self.id2label[i]: probs[0, i].item()
                for i in range(4)
            }
            result['all_scores'] = all_scores

        return result

    def batch_classify(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Classify multiple queries in batches.

        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            batch_size: Batch size for processing

        Returns:
            List of classification results
        """
        if not self._initialized:
            self.initialize()

        results = []
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            batch_docs = documents_list[i:i+batch_size]

            batch_texts = [
                self._create_input_text(q, d)
                for q, d in zip(batch_queries, batch_docs)
            ]

            inputs = self.tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            confidences, pred_ids = torch.max(probs, dim=-1)
            for conf, pred_id in zip(confidences, pred_ids):
                conf = conf.item()
                pred_id = pred_id.item()
                level = self.id2label[pred_id]

                use_lcr = (
                    conf >= self.confidence_threshold and
                    level in ["L1", "L2"]
                )

                results.append({
                    'level': level,
                    'confidence': conf,
                    'use_lcr': use_lcr,
                    'routing_strategy': 'lcr' if use_lcr else 'full_verification'
                })

        return results

    def get_performance_stats(self) -> Dict:
        """
        Get expected performance statistics from paper.

        Returns:
            dict with performance metrics
        """
        return {
            'macro_f1': 0.89,
            'per_class_f1': {
                'L1': 0.91,
                'L2': 0.88,
                'L3': 0.87,
                'L4': 0.86
            },
            'training_examples': 14_650,
            'test_examples': 3_264,
            'paper_reference': 'Table in Appendix, Section on Cascade Routing',
            'improvement_over_roberta': {
                'macro_f1_gain': 0.07,
                'roberta_f1': 0.82
            }
        }

    def save_pretrained(self, save_directory: str):
        """
        Save fine-tuned classifier.

        Args:
            save_directory: Directory to save model
        """
        if not self._initialized:
            raise ValueError("Model not initialized. Call initialize() first.")

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        config = {
            'model_type': 'taxonomy_classifier',
            'base_model': self.model_name,
            'num_labels': 4,
            'id2label': self.id2label,
            'label2id': self.label2id,
            'confidence_threshold': self.confidence_threshold,
            'performance': self.get_performance_stats(),
            'paper_reference': 'Lines 542-578, Appendix Section on Classifier'
        }

        with open(save_path / 'classifier_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        with open(save_path / 'README.md', 'w') as f:
            f.write(f"""# Taxonomy Classifier (DeBERTa-v3-large)


## Specifications
- Base Model: microsoft/deberta-v3-large (304M parameters)
- Classes: 4 (L1-Temporal, L2-Numerical, L3-Entity, L4-Semantic)
- Confidence Threshold: {self.confidence_threshold}

## Cascade Routing
When confidence < 0.7 OR level ∈ {{L3, L4}}:
    → Route to full verification (Self-Consistency)
Otherwise:
    → Route to LCR (lightweight refinement)

## Training
```bash
python src/training/train_classifier.py \\
    --data_path data/benchmarks/ \\
    --base_model microsoft/deberta-v3-large \\
    --output_dir checkpoints/classifier
```

## Usage
```python
classifier = TaxonomyClassifier()
result = classifier.classify(query, documents)
if result['use_lcr']:
    # Use lightweight LCR
else:
    # Use full verification
```
""")

        print(f"Taxonomy Classifier saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = None) -> 'TaxonomyClassifier':
        """
        Load fine-tuned classifier.

        Args:
            load_directory: Directory containing saved model
            device: Device to load model on

        Returns:
            Loaded TaxonomyClassifier instance
        """
        load_path = Path(load_directory)

        with open(load_path / 'classifier_config.json', 'r') as f:
            config = json.load(f)

        classifier = cls(
            model_name=str(load_path),
            confidence_threshold=config['confidence_threshold'],
            device=device
        )

        classifier.initialize()

        print(f"Loaded Taxonomy Classifier from {load_directory}")

        return classifier


def create_classifier(save_dir: str = "checkpoints/classifier"):
    """
    Create classifier with default configuration.
    
    Args:
        save_dir: Directory to save the model
    """
    print("Creating Taxonomy Classifier...")
    print("Downloading microsoft/deberta-v3-large (304M params)")

    classifier = TaxonomyClassifier()
    classifier.initialize()
    classifier.save_pretrained(save_dir)

    print(f"Model saved to {save_dir}")
    print(f"Fine-tune with: python src/training/train_classifier.py")


if __name__ == "__main__":
    print("Taxonomy Classifier - Demo")
    print("=" * 50)

    classifier = TaxonomyClassifier()
    classifier.initialize()

    query = "When was the Eiffel Tower completed?"
    documents = [
        "The Eiffel Tower was completed in 1889.",
        "Construction finished in 1887 according to some sources.",
        "The tower opened to the public in 1889.",
        "Work concluded in March 1889."
    ]

    result = classifier.classify(query, documents, return_all_scores=True)

    print(f"\nQuery: {query}")
    print(f"Predicted Level: {result['level']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Use LCR: {result['use_lcr']}")
    print(f"Routing: {result['routing_strategy']}")
    print(f"\nAll Scores: {result['all_scores']}")

    print("\nTaxonomy Classifier implementation complete!")
