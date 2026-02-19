#!/bin/bash

# Error Analysis Script for BERT vs ModernBERT across Multiple Batch Sizes

cd "$(dirname "$0")/.."

MODE="both"  # tsne, attention, or both
N_SAMPLES=15
TEXT=""

# parameter parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --text) TEXT="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Batch Size 64 폴더 경로
BERT_DIR_64="checkpoints/bert_base_uncased/TargetBS_64_bert-base-uncased"
MODERN_DIR_64="checkpoints/answerdotai/modernbert_base/TargetBS_64_answerdotai/ModernBERT-base"

# Batch Size 256 폴더 경로
BERT_DIR_256="checkpoints/bert_base_uncased/TargetBS_256_bert-base-uncased"
MODERN_DIR_256="checkpoints/answerdotai/modernbert_base/TargetBS_256_answerdotai/ModernBERT-base"

# Batch Size 1024 폴더 경로
BERT_DIR_1024="checkpoints/bert_base_uncased/TargetBS_1024_bert-base-uncased"
MODERN_DIR_1024="checkpoints/answerdotai/modernbert_base/TargetBS_1024_answerdotai/ModernBERT-base"


echo "=========================================="
echo "🚀 Multi-Batch Error Analysis Starting..."
echo "Mode: $MODE"
echo "=========================================="

for BS in 64 256 1024; do
    echo ""
    echo "=========================================="
    echo "🎯 Running Analysis for Target Batch Size: $BS"
    echo "=========================================="
    
    # 1. 타겟 배치 사이즈에 맞는 폴더 경로 선택
    if [ "$BS" -eq 64 ]; then
        B_DIR=$BERT_DIR_64
        M_DIR=$MODERN_DIR_64
    elif [ "$BS" -eq 256 ]; then
        B_DIR=$BERT_DIR_256
        M_DIR=$MODERN_DIR_256
    elif [ "$BS" -eq 1024 ]; then
        B_DIR=$BERT_DIR_1024
        M_DIR=$MODERN_DIR_1024
    fi

    # 2. 해당 폴더에서 epoch 숫자가 가장 큰 .pt 파일을 자동으로 찾기
    B_CKPT=$(find "$B_DIR" -maxdepth 1 -name "*.pt" | sort -V | tail -n 1)
    M_CKPT=$(find "$M_DIR" -maxdepth 1 -name "*.pt" | sort -V | tail -n 1)

    # 3. 파일을 제대로 찾았는지 안전 장치(예외 처리)
    if [ -z "$B_CKPT" ] || [ -z "$M_CKPT" ]; then
        echo "❌ Error: Could not find any .pt files in $B_DIR or $M_DIR!"
        echo "Skipping Batch Size $BS..."
        continue
    fi

    echo "✅ Found BERT Checkpoint: $(basename "$B_CKPT")"
    echo "✅ Found ModernBERT Checkpoint: $(basename "$M_CKPT")"

    if [ -z "$TEXT" ]; then
        python src/analyze_models.py \
            --bert_ckpt "$B_CKPT" \
            --modern_ckpt "$M_CKPT"
    else
        python src/analyze_models.py \
            --bert_ckpt "$B_CKPT" \
            --modern_ckpt "$M_CKPT" \
            --text "$TEXT"
    fi
    
    SAVE_DIR="analysis_outputs/BS_${BS}"
    mkdir -p "$SAVE_DIR"
    
    mv analysis_outputs/*.png "$SAVE_DIR/" 2>/dev/null
    
    echo "✅ Saved results for BS $BS into $SAVE_DIR/"

done

echo ""
echo "=========================================="
echo "🎉 All Analysis Complete!"
echo "Check the 'analysis_outputs/' directory for BS_64, BS_256, BS_1024 folders."
echo "=========================================="