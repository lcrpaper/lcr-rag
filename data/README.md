# LCR Dataset & Data Pipeline

## Overview

This directory contains all datasets and data pipeline for the LCR system.

## Dataset Summary

### Dataset

| Split | Total | L1 | L2 | L3 | L4 | Natural % |
|-------|-------|-----|-----|-----|-----|-----------|
| train | 10,200 | 2,849 | 2,520 | 2,380 | 2,451 | 66% |
| dev | 1,460 | 407 | 360 | 340 | 353 | 66% |
| test | 2,940 | 814 | 720 | 680 | 726 | 66% |
| natural_v1 | 176 | 44 | 42 | 48 | 42 | 100% |
| natural_v2 | 400 | 100 | 95 | 110 | 95 | 100% |
| **Total** | **15,176** | | | | | |

### Composition
- **66% Natural**: Real-world conflicts from web sources
- **34% Augmented**: Synthetic conflicts for class balance
- **BM25 Retrieval**: k1=0.9, b=0.4 (tuned parameters)

### Data Sources and Licenses

| Source Dataset | License | Usage |
|----------------|---------|-------|
| NaturalQuestions | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | Base QA pairs |
| HotpotQA | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Multi-hop questions |
| Wikipedia | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | Document retrieval |
| ConflictQA | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | External benchmark |
| FreshQA | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Temporal subset |

**Our Benchmark License**: The LCR benchmark dataset is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), consistent with the most restrictive source license.

---

## Directory Structure

```
data/
├── README.md                          # This file
├── PROVENANCE.md                      # Complete data lineage
├── VERSION_HISTORY.md                 # Pipeline version changes
├── SCHEMA.md                          # Data format specification
│
├── train/                             # Training data
│   ├── l1_temporal.jsonl              # 2,295 examples
│   ├── l2_numerical.jsonl             # 2,448 examples
│   ├── l3_entity.jsonl                # 2,754 examples
│   └── l4_semantic.jsonl              # 2,703 examples
│
├── dev/                               # Validation data
│   ├── l1_temporal.jsonl              # 291 examples
│   ├── l2_numerical.jsonl             # 327 examples
│   ├── l3_entity.jsonl                # 413 examples
│   └── l4_semantic.jsonl              # 434 examples
│
├── test/                              # Test data
│   ├── l1_temporal.jsonl              # 645 examples
│   ├── l2_numerical.jsonl             # 737 examples
│   ├── l3_entity.jsonl                # 993 examples
│   └── l4_semantic.jsonl              # 889 examples
│
├── natural/                           # Natural conflict test sets
│   ├── natural_test_set_v1.jsonl      # 176 examples (initial curation)
│   ├── natural_test_set_v2.jsonl      # 400 examples (extended)
│   └── annotation_metadata.json       # IAA scores, annotator info
│
├── raw/                               # Stage 0: Source data
│   ├── README.md                      # Download instructions
│   ├── DOWNLOAD_MANIFEST.md           # URLs, dates, checksums
│   ├── natural_questions/             # NQ artifacts
│   ├── hotpotqa/                      # HotpotQA artifacts
│   └── wikipedia/                     # Wikipedia dump info
│
├── intermediate/                      # Processing stages (generated)
│   ├── stage1_extracted/              # Raw text extraction
│   ├── stage2_filtered/               # Quality filtering
│   ├── stage3_tokenized/              # Tokenization
│   └── processing_logs/               # Stage-by-stage logs
│
├── indices/                           # Retrieval indices
│   ├── bm25_v1/                       # Default params [DEPRECATED]
│   ├── bm25_v3/                       # Tuned params
│   └── INDEX_COMPARISON.md            # Performance comparison
│
├── augmented/                         # Synthetic augmentation
│   ├── gpt4_generated/                # GPT-4 conflict generation
│   ├── rule_based/                    # Template-based generation
│   └── mixing_ratios.json             # Augmentation configuration
│
├── analysis/                          # Dataset analysis
│   ├── l2_numerical_stratification.json
│   ├── class_distribution.json
│   └── difficulty_analysis.json
│
├── debug/                             # Development subsets
│   ├── sanity_10.jsonl                # 10 examples (quick test)
│   ├── quick_100.jsonl                # 100 examples (fast dev)
│   └── balanced_1k.jsonl              # 1K balanced subset
│
├── external/                          # External benchmarks
│   ├── conflictqa/                    # ConflictQA benchmark
│   ├── freshqa_subset/                # FreshQA temporal subset
│   └── popqa_conflicts/               # PopQA extracted conflicts
│
└── deprecated/                        # Old pipeline outputs
    ├── v0_csv_format/                 # Pre-JSONL format
    ├── v1_natural_only/               # Before augmentation
    ├── v2_experimental/               # Experimental ratios
    └── MIGRATION.md                   # Format migration guide
```

---

## Data Pipeline

**Key Parameters**:
- BM25: k1=0.9, b=0.4 (tuned for conflict detection)
- Natural: 66% from web sources
- Augmented: 34% synthetic (GPT-4 extraction)
- Seed: 42 (reproducibility)

## Data Format

### Schema

```json
{
  "id": "l1_temp_00001",
  "query": "When was the company founded?",
  "documents": [
    "Document A: The company was founded in 2018...",
    "Document B: Established in 2019, the company..."
  ],
  "answer": "2018",
  "conflict_type": "L1_temporal",
  "conflict_level": "Temporal",
  "source": "NaturalQuestions",
  "natural": true,
  "augmented": false,
  "metadata": {
    "source_a": "wikipedia",
    "source_b": "news_article",
    "retrieval_method": "bm25_v3",
    "difficulty": "medium",
    "annotator_agreement": 0.85
  }
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `query` | string | Yes | Natural language question |
| `documents` | list[str] | Yes | Retrieved documents (2-5) |
| `answer` | string | Yes | Ground truth answer |
| `conflict_type` | string | Yes | L1/L2/L3/L4 taxonomy code |
| `conflict_level` | string | Yes | Human-readable conflict type |
| `source` | string | Yes | Data source (NQ, HotpotQA, etc.) |
| `natural` | bool | Yes | True if naturally occurring |
| `augmented` | bool | Yes | True if synthetically generated |
| `metadata` | dict | No | Additional metadata |

---

## Conflict Taxonomy

### L1: Temporal Conflicts

**Description**: Disagreements about dates, times, or temporal ordering.

**Examples**:
- "Founded in 2018" vs "Established in 2019"
- "Won the championship in 2015" vs "First championship in 2016"

**Paper Results**: LCR achieves **72.4%** accuracy (vs 73.1% Self-Consistency)

### L2: Numerical Conflicts

**Description**: Disagreements about quantities, measurements, or statistics.

**Examples**:
- "Revenue of $5 million" vs "Revenue of $8 million"
- "Population of 50,000" vs "Population of 52,000"

**Paper Results**: LCR achieves **70.1%** accuracy (vs 71.2% Self-Consistency)

### L3: Entity Conflicts

**Description**: Ambiguous entity references or attribution conflicts.

**Examples**:
- Different people with same name
- Conflicting attributions of achievements

**Paper Results**: LCR achieves **59.7%** accuracy (requires explicit reasoning)

### L4: Semantic Conflicts

**Description**: Deep meaning or interpretation conflicts.

**Examples**:
- Contradictory causal claims
- Conflicting expert opinions

**Paper Results**: LCR achieves **57.8%** accuracy (requires explicit reasoning)

---

## Preprocessing Pipeline

### Stage 1: Raw Data Download

```bash
# Download source datasets
python scripts/download_datasets.py --all

# Verify downloads
python scripts/utils/verify_checksums.py --data
```

**Dependencies**:
- Internet connection
- ~50GB disk space
- wget or curl

### Stage 2: Text Extraction

```bash
python scripts/preprocessing/extract_text.py \
    --input data/raw/ \
    --output data/intermediate/stage1_extracted/
```

**Dependencies**:
- Python regex >= 2023.10.3 (NOT built-in re)
- System locale: en_US.UTF-8

### Stage 3: Quality Filtering

```bash
python scripts/preprocessing/filter_quality.py \
    --input data/intermediate/stage1_extracted/ \
    --output data/intermediate/stage2_filtered/
```

**Dependencies**:
- rapidfuzz >= 3.0.0
- spaCy 3.7.x with en_core_web_trf

### Stage 4: BM25 Index Building

```bash
# Use v3 with tuned parameters
python scripts/preprocessing/build_bm25_v3.py \
    --k1 0.9 --b 0.4

# [DEPRECATED] Original parameters
python scripts/preprocessing/build_bm25_v1.py
```

**Dependencies**:
- rank_bm25 == 0.2.2 (EXACTLY - 0.2.3 changed scoring)
- nltk with punkt_tab tokenizer

### Stage 5: Conflict Extraction

```bash
# ML-based extraction (GPT-4)
python scripts/preprocessing/gpt4_extraction_prompts.py

# [DEPRECATED] Rule-based extraction
python scripts/preprocessing/extract_conflicts_rule_based.py
```

**Performance Comparison**:
| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Rule-based | 0.68 | 0.45 | 0.54 |
| GPT-4 | 0.91 | 0.87 | 0.89 |

---

## Data Quality

### Inter-Annotator Agreement

Natural test sets were annotated by 3-5 annotators:

| Dataset | Fleiss' κ | Agreement % |
|---------|-----------|-------------|
| natural_v1 | 0.82 | 89% |
| natural_v2 | 0.86 | 92% |

### Quality Checks

```bash
# Verify data integrity
python scripts/utils/verify_data_quality.py

# Check for duplicates
python scripts/utils/check_duplicates.py

# Validate schema
python scripts/utils/validate_schema.py --split all
```

---

## External Benchmarks

### ConflictQA

```bash
# Download and prepare
python scripts/download_datasets.py --external conflictqa

# Location: data/external/conflictqa/
```

### FreshQA (Temporal Subset)

```bash
# Download and prepare
python scripts/download_datasets.py --external freshqa

# Location: data/external/freshqa_subset/
```

---

## Known Issues

### Issue 1: BM25 Version Mismatch

**Symptom**: Different recall values

**Solution**: Ensure rank_bm25 version is exactly 0.2.2

```bash
pip install rank_bm25==0.2.2
```

### Issue 2: Encoding Issues

**Symptom**: Unicode errors during preprocessing

**Solution**: Set system locale

```bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Issue 3: spaCy Model Mismatch

**Symptom**: Different tokenization results

**Solution**: Use exact spaCy version and model

```bash
pip install spacy==3.7.2
python -m spacy download en_core_web_trf
```


