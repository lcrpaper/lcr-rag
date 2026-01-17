# Unused Baseline Implementations

**Note**: These baselines were explored during development but **NOT INCLUDED** in the final paper.

## Why Not Included

### Contrastive Decoding (`contrastive_decoding.py`)
- **Reason**: Performance too similar to standard decoding
- **Results**: 41.2% accuracy (vs 38.6% baseline)
- **Verdict**: Marginal gains don't justify complexity

### Retrieval Re-ranking (`retrieval_reranking.py`)
- **Reason**: Requires external reliability metadata
- **Results**: 52.3% when metadata available
- **Verdict**: Not generalizable; metadata rarely available in practice

## Preliminary Results

| Baseline | Accuracy | Notes |
|----------|----------|-------|
| Contrastive Decoding | 41.2% | α=0.5 amateur/expert |
| Retrieval Re-ranking | 52.3% | With full metadata |
| LCR (paper) | 61.9% | No external dependencies |

## Usage

These scripts can still be run for exploratory purposes:

```bash
# Contrastive decoding
python scripts/baselines/unused/contrastive_decoding.py \
    --alpha 0.5 \
    --data data/test/

# Retrieval re-ranking (requires metadata)
python scripts/baselines/unused/retrieval_reranking.py \
    --data data/test/ \
    --metadata data/source_reliability.json
```

## Citation

If you use these baselines in your work, please cite appropriately. Note that they are not part of our paper's contributions.
