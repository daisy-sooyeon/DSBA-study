#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

DATA_ROOT="./data/ImageNet9/bg_challenge"
EPOCHS=5  

# ==========================================
# 1. ResNet50 실험 실행
# ==========================================
echo "----------------------------------------------------------------"
echo "🚀 [1/2] Starting Experiment: ResNet50 (CNN)"
echo "----------------------------------------------------------------"

python eval_background_robustness.py \
    --config ./configs/models/resnet50_pretrained_in9.yaml \
    --data_root "$DATA_ROOT" \
    --epochs $EPOCHS

echo "✅ ResNet50 Experiment Completed!"
echo ""

# ==========================================
# 2. ViT-Small 실험 실행
# ==========================================
echo "----------------------------------------------------------------"
echo "🚀 [2/2] Starting Experiment: ViT-Small (Transformer)"
echo "----------------------------------------------------------------"

# timm 설치 확인
pip install timm --quiet

python eval_background_robustness.py \
    --config ./configs/models/vit_small_pretrained_in9.yaml \
    --data_root "$DATA_ROOT" \
    --epochs $EPOCHS

echo "✅ ViT-Small Experiment Completed!"
echo ""

# ==========================================
# 종료
# ==========================================
echo "🎉 All Experiments Finished Successfully!"