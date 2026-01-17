# LCR Examples

This directory contains example scripts demonstrating how to use the LCR system.

## Available Examples

### 1. Quick Start (`quick_start.py`)

Basic demonstration of LCR capabilities with built-in example queries.

```bash
# Run with default examples
python examples/quick_start.py

# Run with custom query
python examples/quick_start.py --query "Who invented the telephone?"

# Specify checkpoint directory
python examples/quick_start.py --checkpoint-dir /path/to/checkpoints
```

### 2. Batch Inference (`batch_inference.py`)

Process multiple queries from JSONL files efficiently.

```bash
# Basic usage
python examples/batch_inference.py --input data/test/l1_temporal.jsonl --output results/output.jsonl

# With batch size and GPU selection
python examples/batch_inference.py --input queries.jsonl --batch-size 32 --gpu 0

# Limit number of examples
python examples/batch_inference.py --input data/test/l2_numerical.jsonl --max-examples 100
```

## Input Format

Examples expect JSONL files with the following schema:

```json
{
  "id": "example_001",
  "query": "What is the population of Tokyo?",
  "documents": [
    {"text": "Document 1 text...", "source": "wikipedia"},
    {"text": "Document 2 text...", "source": "news"}
  ]
}
```

## Output Format

Results are saved as JSONL with:

```json
{
  "id": "example_001",
  "query": "What is the population of Tokyo?",
  "conflict_detected": true,
  "conflict_probability": 0.85,
  "conflict_type": "L2_numerical",
  "type_confidence": 0.78,
  "refined_answer": "Tokyo has 13.96 million residents (city proper).",
  "answer_confidence": 0.72,
  "processing_time_ms": 45.2,
  "token_count": 156
}
```
