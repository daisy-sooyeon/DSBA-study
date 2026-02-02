#!/bin/bash

# NLP 실험 자동 실행 스크립트 - 두 모델(BERT, ModernBERT) 모두 실행

set -e

# Change to exp_1 directory (parent of scripts)
cd "$(dirname "$0")/.."

echo "========================================="
echo "🚀 NLP IMDB Classification Experiments"
echo "========================================="
echo ""

# GPU 설정
GPU_ID=0

# ==========================================
# 1. BERT-base-uncased 실험
# ==========================================
echo "📌 Experiment 1: BERT-base-uncased"
echo "========================================="
CUDA_VISIBLE_DEVICES=$GPU_ID python src/main.py model=bert

echo "✅ BERT Experiment Finished!"
echo ""

# ==========================================
# 2. ModernBERT-base 실험
# ==========================================
echo "📌 Experiment 2: ModernBERT-base"
echo "========================================="
CUDA_VISIBLE_DEVICES=$GPU_ID python src/main.py model=modernbert

echo "✅ ModernBERT Experiment Finished!"
echo ""

# ==========================================
# 종료
# ==========================================
echo "🎉 All Experiments Completed!"
