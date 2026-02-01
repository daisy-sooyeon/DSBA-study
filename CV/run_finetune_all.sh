#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

DATA_ROOT="./data/CIFAR-10-C"
EPOCHS=5
GPU_ID=0

# 체크포인트 경로들
RESNET_SCRATCH="./checkpoints/resnet50_cifar10/best_model.pth"
VIT_SCRATCH="./checkpoints/vit_small_cifar10/best_model.pth"

RESNET_PRETRAINED="./checkpoints/resnet50_pretrained_cifar10/best_model.pth"
VIT_PRETRAINED="./checkpoints/vit_small_pretrained_cifar10/best_model.pth"

echo "🔧 Fine-tuning 실험 시작..."
echo ""

# ==========================================
# 1. ResNet50 (Scratch)
# ==========================================
echo "================================================================"
echo "🚀 [1/4] Fine-tuning: ResNet50 (Scratch)"
echo "   - Weights: $RESNET_SCRATCH"
echo "================================================================"

if [ -f "$RESNET_SCRATCH" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python -c "
from src.finetune import main
from hydra import initialize_config_dir, compose
import os

config_dir = os.path.abspath('./configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='config', overrides=['model=resnet50', f'train.epochs=$EPOCHS', f'train.data_root=$DATA_ROOT'])
    main(cfg, weights_path='$RESNET_SCRATCH')
"
    echo "✅ ResNet50 (Scratch) Fine-tuning Completed!"
else
    echo "⚠️ Warning: Weights not found at $RESNET_SCRATCH"
    echo "   Skipping..."
fi

echo ""

# ==========================================
# 2. ResNet50 (Pretrained)
# ==========================================
echo "================================================================"
echo "🚀 [2/4] Fine-tuning: ResNet50 (Pretrained)"
echo "   - Weights: $RESNET_PRETRAINED"
echo "================================================================"

if [ -f "$RESNET_PRETRAINED" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python -c "
from src.finetune import main
from hydra import initialize_config_dir, compose
import os

config_dir = os.path.abspath('./configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='config', overrides=['model=resnet50_pretrained', f'train.epochs=$EPOCHS', f'train.data_root=$DATA_ROOT'])
    main(cfg, weights_path='$RESNET_PRETRAINED')
"
    echo "✅ ResNet50 (Pretrained) Fine-tuning Completed!"
else
    echo "⚠️ Warning: Weights not found at $RESNET_PRETRAINED"
    echo "   Skipping..."
fi

echo ""

# ==========================================
# 3. ViT-Small (Scratch)
# ==========================================
echo "================================================================"
echo "🚀 [3/4] Fine-tuning: ViT-Small (Scratch)"
echo "   - Weights: $VIT_SCRATCH"
echo "================================================================"

if [ -f "$VIT_SCRATCH" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python -c "
from src.finetune import main
from hydra import initialize_config_dir, compose
import os

config_dir = os.path.abspath('./configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='config', overrides=['model=vit_small', f'train.epochs=$EPOCHS', f'train.data_root=$DATA_ROOT'])
    main(cfg, weights_path='$VIT_SCRATCH')
"
    echo "✅ ViT-Small (Scratch) Fine-tuning Completed!"
else
    echo "⚠️ Warning: Weights not found at $VIT_SCRATCH"
    echo "   Skipping..."
fi

echo ""

# ==========================================
# 4. ViT-Small (Pretrained)
# ==========================================
echo "================================================================"
echo "🚀 [4/4] Fine-tuning: ViT-Small (Pretrained)"
echo "   - Weights: $VIT_PRETRAINED"
echo "================================================================"

if [ -f "$VIT_PRETRAINED" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python -c "
from src.finetune import main
from hydra import initialize_config_dir, compose
import os

config_dir = os.path.abspath('./configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='config', overrides=['model=vit_small_pretrained', f'train.epochs=$EPOCHS', f'train.data_root=$DATA_ROOT'])
    main(cfg, weights_path='$VIT_PRETRAINED')
"
    echo "✅ ViT-Small (Pretrained) Fine-tuning Completed!"
else
    echo "⚠️ Warning: Weights not found at $VIT_PRETRAINED"
    echo "   Skipping..."
fi

echo ""

# ==========================================
# 종료
# ==========================================
echo "🎉 All Fine-tuning Experiments Completed!"
