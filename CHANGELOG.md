# Changelog

All notable changes to the LCR project are documented in this file.

This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format
and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Multi-scale detector (see `src/models/experimental/multi_scale_detector.py`)
- Contrastive classifier training objective
- Additional cross-model evaluations (Phi-3, Qwen-2)

---

## [1.0.0] - Paper Submission [PAPER]

**This is the official paper release. All experiments in the paper were run using this version.**

### Added
- Complete LCR system implementation
- Conflict Detector
- Refinement Module
- Taxonomy Classifier
- Full training pipelines for all components
- Comprehensive evaluation scripts
- Natural conflict test sets
- Baseline implementations
- Ablation study configurations

### Changed
- Refinement architecture changed from additive to interpolation (α=0.3)
- Detector reduced from 3-layer to 2-layer MLP
- Default T_max set to 3 based on ablation results

### Deprecated
- `src/models/legacy/detector_v0.py` - Original 3-layer detector
- `src/models/legacy/refinement_v0.py` - Additive refinement
- `scripts/preprocessing/v1_deprecated/` - Old preprocessing pipeline

### Removed
- Contrastive decoding baseline (not included in paper)
- Retrieval re-ranking baseline (not included in paper)

## [0.9.0] - Internal

### Added
- KL divergence regularization to refinement training
- Natural conflict dataset v2
- Cross-model generalization experiments

### Changed
- Updated hyperparameters based on sweep results
- Improved data loading for large-scale training

### Fixed
- Memory leak in refinement iteration loop
- Incorrect L4 accuracy computation

## [0.8.0] - Internal

### Added
- Hyperparameter sweep infrastructure
- W&B integration for experiment tracking
- Multi-GPU training support

### Changed
- Moved to bottleneck MLP architecture for refinement
- Standardized configuration format (YAML)

## [0.7.0] - Internal

### Added
- Taxonomy classifier with DeBERTa
- L1-L4 conflict categorization

### Changed
- Dataset restructured with category-specific splits

## [0.5.0] - Initial

### Added
- Initial conflict detector implementation
- Basic refinement module
- Preprocessing pipeline v1
- Evaluation metrics

---

## Version Numbering

- Major: Paper submission milestones
- Minor: Feature additions
- Patch: Bug fixes

## Related Documents

- `REPRODUCIBILITY.md`: Full reproduction instructions
- `docs/LEGACY_NOTES.md`: Historical development notes
