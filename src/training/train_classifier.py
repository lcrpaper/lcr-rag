"""
Train Taxonomy Classifier (DeBERTa-v3-large)

This script fine-tunes DeBERTa-v3-large to classify conflict types:
- L1: Temporal conflicts (dates, times, sequences)
- L2: Numerical conflicts (quantities, measurements)
- L3: Entity conflicts (names, places, identities)
- L4: Semantic conflicts (claims, facts, interpretations)

Usage:
    python src/training/train_classifier.py \\
        --config configs/classifier_config.yaml \\
        --output checkpoints/classifier/
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConflictDataset(Dataset):
    """Dataset for conflict type classification."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data(data_path)

        self.label_map = {
            'L1': 0,
            'L2': 1,
            'L3': 2,
            'L4': 3
        }
        self.id2label = {v: k for k, v in self.label_map.items()}

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load JSONL data file."""
        examples = []
        data_file = Path(data_path)

        if data_file.is_dir():
            for split in ['train', 'dev', 'test']:
                for conflict_type in ['temp_conflict', 'num_conflict', 'entity_conflict', 'semantic_conflict']:
                    file_path = data_file / conflict_type / f"{split}.jsonl"
                    if file_path.exists():
                        examples.extend(self._read_jsonl(file_path))
        else:
            examples = self._read_jsonl(data_file)

        logger.info(f"Loaded {len(examples)} examples from {data_path}")
        return examples

    def _read_jsonl(self, file_path: Path) -> List[Dict]:
        """Read JSONL file."""
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]

        query = example['query']
        docs_text = " ".join([doc['text'][:200] for doc in example['documents'][:5]])
        text = f"{query} [SEP] {docs_text}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        conflict_level = example.get('conflict_level', 'L1')
        label = self.label_map.get(conflict_level, 0)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device) -> Dict:
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    accuracy = (all_preds == all_labels).mean()
    avg_loss = total_loss / len(dataloader)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': f1,
        'classification_report': classification_report(
            all_labels,
            all_preds,
            target_names=['L1_Temporal', 'L2_Numerical', 'L3_Entity', 'L4_Semantic'],
            digits=4
        )
    }


def main():
    parser = argparse.ArgumentParser(description='Train Taxonomy Classifier')
    parser.add_argument('--config', type=str, default='configs/classifier_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/benchmarks',
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='checkpoints/classifier',
                       help='Output directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']

    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Classes: {num_classes}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )
    model.to(device)

    logger.info("Loading datasets...")
    train_dataset = ConflictDataset(args.data_path, tokenizer)

    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Val examples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_metrics': []}

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        history['train_loss'].append(train_loss)
        logger.info(f"  Train loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        history['val_metrics'].append(val_metrics)

        logger.info(f"  Val loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Val Macro-F1: {val_metrics['macro_f1']:.4f}")
        logger.info(f"  Val Precision: {val_metrics['precision']:.4f}")
        logger.info(f"  Val Recall: {val_metrics['recall']:.4f}")

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            with open(output_dir / 'training_config.json', 'w') as f:
                json.dump({
                    'model_name': model_name,
                    'num_classes': num_classes,
                    'best_val_f1': float(best_val_f1),
                    'best_epoch': epoch + 1,
                    'seed': args.seed
                }, f, indent=2)

            logger.info(f"  Saved best model (F1: {best_val_f1:.4f})")

    logger.info("\nFinal Evaluation:")
    logger.info(val_metrics['classification_report'])

    with open(Path(args.output) / 'history.json', 'w') as f:
        serializable_history = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_macro_f1': [float(m['macro_f1']) for m in history['val_metrics']],
            'val_accuracy': [float(m['accuracy']) for m in history['val_metrics']]
        }
        json.dump(serializable_history, f, indent=2)

    logger.info(f"\nTraining complete!")
    logger.info(f"  Best Macro-F1: {best_val_f1:.4f}")
    logger.info(f"  Expected (paper): 0.89 +/- 0.02")
    logger.info(f"  Model saved to: {args.output}")


if __name__ == "__main__":
    main()
