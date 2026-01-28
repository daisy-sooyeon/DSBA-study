#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

DATA_ROOT="./data/CIFAR-10-C"
EPOCHS=5

RESNET_SCRATCH="./checkpoints/resnet50_cifar10/best_model.pth"
VIT_SCRATCH="./checkpoints/vit_small_cifar10/best_model.pth"

RESNET_PRETRAINED="./checkpoints/resnet50_pretrained_cifar10/best_model.pth"
VIT_PRETRAINED="./checkpoints/vit_small_pretrained_cifar10/best_model.pth"

# ==========================================
# 1. ResNet50 (Scratch)
# ==========================================
echo "----------------------------------------------------------------"
echo "🚀 [1/4] Fine-tuning: ResNet50 (Scratch)"
echo "   - Weights: $RESNET_SCRATCH"
echo "----------------------------------------------------------------"

if [ -f "$RESNET_SCRATCH" ]; then
    python finetune.py \
        --config ./configs/models/resnet50_pretrained.yaml \
        --weights "$RESNET_SCRATCH" \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS
else
    echo "⚠️ Error: Weights not found at $RESNET_SCRATCH"
    echo "   Skipping..."
fi

REAL_MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model_name'])")
IS_PRETRAINED=$(python -c "import yaml; print(yaml.safe_load(open('./configs/models/resnet50.yaml')).get('pretrained', False))")
SAVE_DIR="./checkpoints/resnet50_cifar10"

if [ -f "./logs/finetune_results_${REAL_MODEL_NAME}_${IS_PRETRAINED}.csv" ]; then
    mv "./logs/finetune_results_${REAL_MODEL_NAME}_${IS_PRETRAINED}.csv" "$SAVE_DIR/finetune_results.csv"
fi

echo ""

# ==========================================
# 2. ResNet50 (Pretrained)
# ==========================================
echo "----------------------------------------------------------------"
echo "🚀 [2/4] Fine-tuning: ResNet50 (Pretrained)"
echo "   - Weights: $RESNET_PRETRAINED"
echo "----------------------------------------------------------------"

if [ -f "$RESNET_PRETRAINED" ]; then
    python finetune.py \
        --config ./configs/models/resnet50_pretrained.yaml \
        --weights "$RESNET_PRETRAINED" \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS
else
    echo "⚠️ Error: Weights not found at $RESNET_PRETRAINED"
    echo "   Skipping..."
fi

REAL_MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model_name'])")
IS_PRETRAINED=$(python -c "import yaml; print(yaml.safe_load(open('./configs/models/resnet50_pretrained.yaml')).get('pretrained', False))")
SAVE_DIR="./checkpoints/resnet50_pretrained_cifar10"

if [ -f "./logs/finetune_results_${REAL_MODEL_NAME}_${IS_PRETRAINED}.csv" ]; then
    mv "./logs/finetune_results_${REAL_MODEL_NAME}_${IS_PRETRAINED}.csv" "$SAVE_DIR/finetune_results.csv"
fi

echo ""

# ==========================================
# 3. ViT-Small (Scratch)
# ==========================================
echo "----------------------------------------------------------------"
echo "🚀 [3/4] Fine-tuning: ViT-Small (Scratch)"
echo "   - Weights: $VIT_SCRATCH"
echo "----------------------------------------------------------------"

if [ -f "$VIT_SCRATCH" ]; then
    python finetune.py \
        --config ./configs/models/vit_small_pretrained.yaml \
        --weights "$VIT_SCRATCH" \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS
else
    echo "⚠️ Error: Weights not found at $VIT_SCRATCH"
    echo "   Skipping..."
fi

REAL_MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model_name'])")
IS_PRETRAINED=$(python -c "import yaml; print(yaml.safe_load(open('./configs/models/vit_small.yaml')).get('pretrained', False))")
SAVE_DIR="./checkpoints/vit_small_cifar10"

if [ -f "./logs/finetune_results_${REAL_MODEL_NAME}_${IS_PRETRAINED}.csv" ]; then
    mv "./logs/finetune_results_${REAL_MODEL_NAME}_${IS_PRETRAINED}.csv" "$SAVE_DIR/finetune_results.csv"
fi

echo ""

# ==========================================
# 4. ViT-Small (Pretrained)
# ==========================================
echo "----------------------------------------------------------------"
echo "🚀 [4/4] Fine-tuning: ViT-Small (Pretrained)"
echo "   - Weights: $VIT_PRETRAINED"
echo "----------------------------------------------------------------"

if [ -f "$VIT_PRETRAINED" ]; then
    python finetune.py \
        --config ./configs/models/vit_small_pretrained.yaml \
        --weights "$VIT_PRETRAINED" \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS
else
    echo "⚠️ Error: Weights not found at $VIT_PRETRAINED"
    echo "   Skipping..."
fi

REAL_MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model_name'])")
IS_PRETRAINED=$(python -c "import yaml; print(yaml.safe_load(open('./configs/models/vit_small_pretrained.yaml')).get('pretrained', False))")
SAVE_DIR="./checkpoints/vit_small_pretrained_cifar10"

if [ -f "./logs/finetune_results_${REAL_MODEL_NAME}_${IS_PRETRAINED}.csv" ]; then
    mv "./logs/finetune_results_${REAL_MODEL_NAME}_${IS_PRETRAINED}.csv" "$SAVE_DIR/finetune_results.csv"
fi

# ==========================================
# 종료
# ==========================================
echo ""
echo "🎉 All 4 Fine-tuning Experiments Completed!"