#!/bin/bash

# Error Analysis Script for BERT vs ModernBERT
# This script performs t-SNE boundary error analysis and attention map comparison

cd "$(dirname "$0")/.."

# Default checkpoint paths (adjust if needed)
BERT_CHECKPOINT="checkpoints/bert_base_uncased/epoch_2_acc_0.8810.pt"
MODERNBERT_CHECKPOINT="checkpoints/answerdotai/modernbert_base/epoch_2_acc_0.9092.pt"

# Parse arguments
MODE="both"  # tsne, attention, or both
N_SAMPLES=15

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --bert-checkpoint)
            BERT_CHECKPOINT="$2"
            shift 2
            ;;
        --modernbert-checkpoint)
            MODERNBERT_CHECKPOINT="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --text)
            TEXT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--mode tsne|attention|both] [--bert-checkpoint PATH] [--modernbert-checkpoint PATH] [--n-samples N] [--text TEXT]"
            exit 1
            ;;
    esac
done

# Run analysis
echo "=========================================="
echo "Error Analysis: BERT vs ModernBERT"
echo "=========================================="
echo "Mode: $MODE"
echo "BERT Checkpoint: $BERT_CHECKPOINT"
echo "ModernBERT Checkpoint: $MODERNBERT_CHECKPOINT"
echo "=========================================="
echo ""

if [ -z "$TEXT" ]; then
    python src/analyze_errors.py \
        --mode "$MODE" \
        --bert_checkpoint "$BERT_CHECKPOINT" \
        --modernbert_checkpoint "$MODERNBERT_CHECKPOINT" \
        --n_samples "$N_SAMPLES"
else
    python src/analyze_errors.py \
        --mode "$MODE" \
        --bert_checkpoint "$BERT_CHECKPOINT" \
        --modernbert_checkpoint "$MODERNBERT_CHECKPOINT" \
        --n_samples "$N_SAMPLES" \
        --text "$TEXT"
fi

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "Results saved to: analysis_outputs/"
echo "=========================================="




