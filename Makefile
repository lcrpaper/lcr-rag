# ==============================================================================
# LCR Project Makefile
# ==============================================================================
#
# Master Makefile for the Latent Conflict Refinement research project.
# This Makefile has accumulated targets over the project lifecycle.
#
# QUICK START:
#   make help                    # Show this help
#   make verify                  # Verify environment
#   make demo                    # Run quick demo
#   make reproduce-paper         # Full paper reproduction
#
# ==============================================================================

.PHONY: help verify demo clean all

# Default shell
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# Project configuration
PROJECT_NAME := lcr
PYTHON := python
PIP := pip
PYTEST := pytest
CONFIG_DIR := configs
DATA_DIR := data
CHECKPOINT_DIR := checkpoints
RESULTS_DIR := results
LOGS_DIR := logs

# Environment detection
CUDA_AVAILABLE := $(shell $(PYTHON) -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
GPU_COUNT := $(shell $(PYTHON) -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

# Version detection
TORCH_VERSION := $(shell $(PYTHON) -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
TRANSFORMERS_VERSION := $(shell $(PYTHON) -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")

# ==============================================================================
# HELP
# ==============================================================================

help:
	@echo "=============================================================================="
	@echo "LCR Project Makefile - Available Targets"
	@echo "=============================================================================="
	@echo ""
	@echo "PAPER REPRODUCTION:"
	@echo "  reproduce-paper      All experiments"
	@echo "  reproduce-detector   Train detector"
	@echo "  reproduce-refinement Train refinement"
	@echo "  reproduce-classifier Train classifier"
	@echo "  reproduce-eval       Run all evaluations"
	@echo ""
	@echo "QUICK START:"
	@echo "  verify               Verify environment setup"
	@echo "  demo                 Run quick demo"
	@echo "  sanity               Quick sanity check"
	@echo ""
	@echo "DATA PIPELINE:"
	@echo "  data-download        Download raw data sources"
	@echo "  data-preprocess      Run full preprocessing"
	@echo "  data-verify          Verify data integrity"
	@echo ""
	@echo "TRAINING:"
	@echo "  train-detector       Train conflict detector"
	@echo "  train-refinement     Train refinement module"
	@echo "  train-classifier     Train taxonomy classifier"
	@echo "  train-all            Train all components"
	@echo ""
	@echo "Use 'make <target> VERBOSE=1' for verbose output"
	@echo "Use 'make <target> DRY_RUN=1' to see commands without executing"
	@echo "=============================================================================="

# ==============================================================================
# PAPER REPRODUCTION TARGETS
# ==============================================================================

.PHONY: reproduce-paper reproduce-detector reproduce-refinement reproduce-classifier

# Full reproduction pipeline
reproduce-paper: verify data-paper reproduce-detector reproduce-refinement reproduce-classifier reproduce-eval
	@echo "Full reproduction complete"
	@echo "Results saved to $(RESULTS_DIR)/paper/"

# Train detector
reproduce-detector:
	@echo "Training conflict detector..."
	$(PYTHON) src/training/train_detector.py \
		--config $(CONFIG_DIR)/detector_config.yaml \
		--output $(CHECKPOINT_DIR)/detector/ \
		--seed 42 \
		--wandb-project lcr-paper
	@echo "Detector training complete"

# Train refinement module
reproduce-refinement:
	@echo "Training refinement module..."
	$(PYTHON) src/training/train_refinement.py \
		--config $(CONFIG_DIR)/refinement_config.yaml \
		--output $(CHECKPOINT_DIR)/refinement/ \
		--seed 42 \
		--wandb-project lcr-paper
	@echo "Refinement training complete"

# Train taxonomy classifier
reproduce-classifier:
	@echo "Training taxonomy classifier..."
	$(PYTHON) src/training/train_classifier.py \
		--config $(CONFIG_DIR)/classifier_config.yaml \
		--output $(CHECKPOINT_DIR)/classifier/ \
		--seed 42 \
		--wandb-project lcr-paper
	@echo "Classifier training complete"

# Run all evaluations
reproduce-eval: eval-main eval-ablations eval-cross-model
	@echo "All evaluations complete"

# ==============================================================================
# DATA PIPELINE TARGETS
# ==============================================================================

.PHONY: data-download data-preprocess data-verify data-clean

# Download raw data sources
data-download:
	@echo "Downloading raw data sources..."
	$(PYTHON) scripts/download_datasets.py --all
	@echo "Download complete"

# Download specific sources
data-download-nq:
	$(PYTHON) scripts/download_datasets.py --source natural_questions

data-download-hotpot:
	$(PYTHON) scripts/download_datasets.py --source hotpotqa

data-download-wiki:
	$(PYTHON) scripts/download_datasets.py --source wikipedia

# Full preprocessing pipeline
data-preprocess: data-extract data-filter data-tokenize data-index
	@echo "Preprocessing complete"

# Individual preprocessing stages
data-extract:
	@echo "Stage 1: Text extraction..."
	$(PYTHON) scripts/preprocessing/extract_text.py \
		--input $(DATA_DIR)/raw/ \
		--output $(DATA_DIR)/intermediate/stage1_extracted/

data-filter:
	@echo "Stage 2: Quality filtering..."
	$(PYTHON) scripts/preprocessing/filter_quality.py \
		--input $(DATA_DIR)/intermediate/stage1_extracted/ \
		--output $(DATA_DIR)/intermediate/stage2_filtered/

data-tokenize:
	@echo "Stage 3: Tokenization..."
	$(PYTHON) scripts/preprocessing/tokenize_data.py \
		--input $(DATA_DIR)/intermediate/stage2_filtered/ \
		--output $(DATA_DIR)/intermediate/stage3_tokenized/

data-index:
	@echo "Stage 4: Building BM25 index..."
	$(PYTHON) scripts/preprocessing/build_bm25_v3.py \
		--k1 0.9 --b 0.4

# Verify data integrity
data-verify:
	@echo "Verifying data integrity..."
	$(PYTHON) scripts/utils/verify_paper_data.py
	$(PYTHON) scripts/utils/check_duplicates.py
	$(PYTHON) scripts/utils/validate_schema.py --split all

# Clean generated data (preserves raw)
data-clean:
	rm -rf $(DATA_DIR)/intermediate/
	rm -rf $(DATA_DIR)/indices/
	@echo "Intermediate data cleaned"

# ==============================================================================
# TRAINING TARGETS
# ==============================================================================

.PHONY: train-detector train-refinement train-classifier train-all
.PHONY: train-detector-debug train-refinement-debug train-classifier-debug
.PHONY: train-resume train-distributed

# Train conflict detector
train-detector:
	$(PYTHON) src/training/train_detector.py \
		--config $(CONFIG_DIR)/detector_config.yaml

# Train with debug settings
train-detector-debug:
	$(PYTHON) src/training/train_detector.py \
		--config $(CONFIG_DIR)/detector_config.yaml \
		--data $(DATA_DIR)/debug/sanity_10.jsonl \
		--epochs 2 \
		--no-wandb

# Train refinement module
train-refinement:
	$(PYTHON) src/training/train_refinement.py \
		--config $(CONFIG_DIR)/refinement_config.yaml

train-refinement-debug:
	$(PYTHON) src/training/train_refinement.py \
		--config $(CONFIG_DIR)/refinement_config.yaml \
		--data $(DATA_DIR)/debug/sanity_10.jsonl \
		--iterations 2 \
		--no-wandb

# Train taxonomy classifier
train-classifier:
	$(PYTHON) src/training/train_classifier.py \
		--config $(CONFIG_DIR)/classifier_config.yaml

train-classifier-debug:
	$(PYTHON) src/training/train_classifier.py \
		--config $(CONFIG_DIR)/classifier_config.yaml \
		--data $(DATA_DIR)/debug/sanity_10.jsonl \
		--epochs 1 \
		--no-wandb

# Train all components
train-all: train-detector train-refinement train-classifier

# Resume training from checkpoint
train-resume:
	@echo "Resuming training..."
	@if [ -z "$(CHECKPOINT)" ]; then echo "Error: CHECKPOINT not specified"; exit 1; fi
	./scripts/training/resume_training.sh $(CHECKPOINT)

# Distributed training (multi-GPU)
train-distributed:
	@echo "Starting distributed training..."
	@if [ "$(GPU_COUNT)" -lt "2" ]; then echo "Warning: <2 GPUs detected"; fi
	torchrun --nproc_per_node=$(GPU_COUNT) \
		src/training/train_refinement.py \
		--config $(CONFIG_DIR)/environments/cloud_4xa100.yaml \
		--distributed

# Hyperparameter sweep
train-sweep:
	$(PYTHON) scripts/training/run_hyperparameter_sweep.py \
		--config $(CONFIG_DIR)/experiments/sweeps/detector_lr.yaml

# ==============================================================================
# BASELINE TARGETS
# ==============================================================================

.PHONY: baselines-all baselines-context baselines-llm baselines-self-consistency

# Run all baselines
baselines-all: baselines-context baselines-llm baselines-self-consistency
	@echo "All baselines complete"

# Context-based baselines
baselines-context:
	@echo "Running context-based baselines..."
	$(PYTHON) scripts/baselines/context_majority.py
	$(PYTHON) scripts/baselines/context_recency.py
	$(PYTHON) scripts/baselines/context_authority.py

# LLM-based baselines
baselines-llm:
	@echo "Running LLM baselines..."
	$(PYTHON) scripts/baselines/gpt4_baseline.py
	$(PYTHON) scripts/baselines/claude_baseline.py
	$(PYTHON) scripts/baselines/llama_baseline.py

# Self-consistency baselines
baselines-self-consistency:
	@echo "Running self-consistency baselines..."
	$(PYTHON) scripts/baselines/self_consistency.py --samples 5
	$(PYTHON) scripts/baselines/self_consistency.py --samples 10
	$(PYTHON) scripts/baselines/self_consistency.py --samples 20

# ==============================================================================
# ANALYSIS
# ==============================================================================

.PHONY: analysis-all analysis-error analysis-probing analysis-significance

# Full analysis pipeline
analysis-all: analysis-error analysis-probing analysis-significance
	@echo "Analysis complete"

# Error analysis
analysis-error:
	$(PYTHON) scripts/analysis/error_analysis.py \
		--results $(RESULTS_DIR)/main/ \
		--output $(RESULTS_DIR)/analysis/

# Probing experiments (Appendix)
analysis-probing:
	$(PYTHON) scripts/analysis/run_probing_experiments.py

# Statistical significance
analysis-significance:
	$(PYTHON) scripts/analysis/compute_significance.py \
		--results $(RESULTS_DIR)/main/

# ==============================================================================
# ENVIRONMENT & SETUP
# ==============================================================================

.PHONY: verify setup setup-dev setup-minimal install-deps
.PHONY: check-cuda check-versions

# Verify environment
verify:
	@echo "Verifying environment..."
	@echo "CUDA Available: $(CUDA_AVAILABLE)"
	@echo "GPU Count: $(GPU_COUNT)"
	@echo "PyTorch: $(TORCH_VERSION)"
	@echo "Transformers: $(TRANSFORMERS_VERSION)"
	$(PYTHON) scripts/utils/verify_environment.py

# Strict verification (for paper reproduction)
verify-strict:
	$(PYTHON) scripts/utils/verify_environment.py --strict

# Full setup
setup: install-deps download-models verify
	@echo "Setup complete"

# Development setup
setup-dev: install-deps-dev download-models verify
	@echo "Development setup complete"

# Minimal setup (inference only)
setup-minimal:
	$(PIP) install -r requirements-minimal.txt
	$(PYTHON) scripts/download_models.py --minimal

# Install dependencies
install-deps:
	$(PIP) install -r requirements.txt

install-deps-exact:
	$(PIP) install -r requirements-exact.txt --extra-index-url https://download.pytorch.org/whl/cu118

install-deps-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

# Download models
download-models:
	$(PYTHON) scripts/download_models.py

download-models-minimal:
	$(PYTHON) scripts/download_models.py --minimal

# Check CUDA
check-cuda:
	@$(PYTHON) -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Check all versions
check-versions:
	@$(PYTHON) scripts/utils/check_versions.py
