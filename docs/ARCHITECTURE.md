# LCR System Architecture

This document provides a detailed technical description of the Latent Conflict Refinement (LCR) system architecture.

## System Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │                  LCR System                      │
                    │                                                  │
Query + Documents   │  ┌──────────┐   ┌────────────┐   ┌───────────┐ │  Refined
─────────────────►  │  │ Conflict │──►│  Taxonomy  │──►│ Iterative │ │  Answer
                    │  │ Detector │   │ Classifier │   │ Refinement│─┼────────►
                    │  └──────────┘   └────────────┘   └───────────┘ │
                    │                                                  │
                    └─────────────────────────────────────────────────┘
```

## Component Details

### 1. Conflict Detector

**Purpose**: Binary classification to detect whether retrieved documents contain conflicting information.

**Architecture**:
```
Input: h ∈ R^4096 (LLM hidden state)
       ↓
    Linear(4096, 512) + ReLU + Dropout(0.1)
       ↓
    Linear(512, 1) + Sigmoid
       ↓
Output: p(conflict) ∈ [0, 1]
```

**Parameters**: 2,098,177
- Layer 1: 4096 × 512 + 512 = 2,097,664
- Layer 2: 512 × 1 + 1 = 513

**Training Configuration**:
- Batch size: 64
- Learning rate: 2e-5
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: Linear warmup (500 steps)
- Early stopping: patience=5

**Expected Performance**:
- F1: 0.871
- Precision: 0.883
- Recall: 0.859

### 2. Taxonomy Classifier

**Purpose**: 4-way classification of conflict type (L1-L4).

**Architecture**: Fine-tuned DeBERTa-v3-large
```
Input: Query + [SEP] + Doc1 + [SEP] + Doc2
       ↓
    DeBERTa-v3-large encoder
       ↓
    [CLS] token embedding
       ↓
    Classification head (768 → 4)
       ↓
Output: class ∈ {L1, L2, L3, L4}
```

**Parameters**: ~434M (DeBERTa) + 3,076 (head)

**Training Configuration**:
- Batch size: 16
- Learning rate: 2e-5
- Max length: 512 tokens
- Epochs: 10

**Expected Performance**:
- Macro F1: 0.888
- Per-class F1: L1=0.921, L2=0.903, L3=0.867, L4=0.861

### 3. Iterative Refinement Module

**Purpose**: Iteratively refine LLM hidden states to resolve detected conflicts.

**Architecture** (Bottleneck MLP):
```
Input: h ∈ R^4096 (hidden state)

For t = 1 to T:
    # Down-projection
    z_t = ReLU(W_down · h_t + b_down)     # z_t ∈ R^732

    # Up-projection
    Δh_t = W_up · z_t + b_up              # Δh_t ∈ R^4096

    # Interpolated update
    h_{t+1} = (1 - α) · h_t + α · Δh_t

Output: h_T (refined hidden state)
```

**Key Hyperparameters**:
- Bottleneck dimension r = 732 (= 4096 / 5.6)
- Interpolation coefficient α = 0.3
- Number of iterations T = 3

**Parameters**: 5,996,544
- W_down: 4096 × 732 = 2,998,272
- b_down: 732
- W_up: 732 × 4096 = 2,998,272
- b_up: 0 (no bias for up-projection)

**Training Configuration**:
- Batch size: 32
- Learning rate: 1e-5
- Optimizer: AdamW
- Scheduler: Cosine
- KL divergence weight: 0.1

## Data Flow

### Training Pipeline

```
1. Data Preparation
   Raw Wikipedia/News → BM25 Retrieval → Deduplication → Annotation → Splits

2. Detector Training
   Train split → Extract LLM hidden states → Train binary classifier

3. Classifier Training
   Train split → Tokenize text → Fine-tune DeBERTa

4. Refinement Training
   Train split → Extract hidden states → Train bottleneck MLP
                                        (with KL regularization)
```

### Inference Pipeline

```
1. Input Processing
   Query + Documents → LLM forward pass → Extract hidden state h_0

2. Conflict Detection
   h_0 → Detector → p(conflict)
   If p < 0.5: return original answer

3. Conflict Classification
   Query + Docs → Classifier → conflict_type

4. Iterative Refinement
   h_0 → Refinement(T=3) → h_T

5. Answer Generation
   h_T → LLM decoder → Refined answer
```

## Mathematical Formulation

### Refinement Objective

The refinement module is trained to minimize:

```
L = L_task + λ · L_KL

where:
- L_task = CrossEntropy(LLM(h_T), gold_answer)
- L_KL = KL(p(h_T) || p(h_0))  # Regularization
- λ = 0.1
```

### Alpha Selection Criterion

Optimal α balances correction strength vs. information preservation:

```
α* = argmax_α [ Accuracy(α) ]
   = argmax_α [ E[1{decode(h_T) = gold}] ]
```

Empirically: α* = 0.3 (see ablation studies)

## Layer Selection

The refinement module operates on hidden states from layer 16 (of 32) in LLaMA-3-8B. This was determined through probing experiments:

| Layer | Probing Accuracy | Notes |
|-------|------------------|-------|
| 8     | 0.712 | Too early - insufficient context |
| 12    | 0.756 | Improving |
| 16    | 0.823 | **Optimal** - best conflict signal |
| 20    | 0.798 | Decreasing - too committed |
| 24    | 0.767 | Late layers less malleable |

## Extension Points

The architecture supports several extensions:

1. **Multi-model refinement**: Train separate refinement modules per LLM
2. **Adaptive iterations**: Learn T based on conflict difficulty
3. **Hierarchical refinement**: Different α per conflict type
4. **Attention-based routing**: Route to type-specific refinement heads
