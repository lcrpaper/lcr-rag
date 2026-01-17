#!/bin/bash

set -e

MODEL_TYPE=$1
CHECKPOINT_PATH=$2

if [ -z "$MODEL_TYPE" ] || [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: $0 <model_type> <checkpoint_path>"
    echo "  model_type: detector, refinement, or classifier"
    echo "  checkpoint_path: Path to checkpoint file"
    echo ""
    echo "Examples:"
    echo "  $0 detector checkpoints/detector/checkpoint_epoch_3.pt"
    echo "  $0 refinement checkpoints/refinement/latest.pt"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "=========================================="
echo "RESUMING TRAINING"
echo "=========================================="
echo "Model type: $MODEL_TYPE"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "=========================================="

case $MODEL_TYPE in
    detector)
        python src/training/train_detector.py \
            --config configs/detector_config.yaml \
            --resume "$CHECKPOINT_PATH" \
            --output checkpoints/detector_resumed
        ;;
    refinement)
        python src/training/train_refinement.py \
            --config configs/refinement_config.yaml \
            --resume "$CHECKPOINT_PATH" \
            --output checkpoints/refinement_resumed
        ;;
    classifier)
        python src/training/train_classifier.py \
            --config configs/classifier_config.yaml \
            --resume "$CHECKPOINT_PATH" \
            --output checkpoints/classifier_resumed
        ;;
    *)
        echo "Error: Unknown model type: $MODEL_TYPE"
        echo "Valid types: detector, refinement, classifier"
        exit 1
        ;;
esac

echo ""
echo "Training resumed successfully!"
