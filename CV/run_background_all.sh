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

CUDA_VISIBLE_DEVICES=0 python -c "
from src.eval_background_robustness import main
from hydra import initialize_config_dir, compose
import os

config_dir = os.path.abspath('./configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='config', overrides=['model=resnet50_pretrained_in9', 'background.epochs=$EPOCHS'])
    main(cfg)
"

echo "✅ ResNet50 Experiment Completed!"
echo ""

# ==========================================
# 2. ViT-Small 실험 실행
# ==========================================
echo "----------------------------------------------------------------"
echo "🚀 [2/2] Starting Experiment: ViT-Small (Transformer)"
echo "----------------------------------------------------------------"

CUDA_VISIBLE_DEVICES=0 python -c "
from src.eval_background_robustness import main
from hydra import initialize_config_dir, compose
import os

config_dir = os.path.abspath('./configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='config', overrides=['model=vit_small_pretrained_in9', 'background.epochs=$EPOCHS'])
    main(cfg)
"

echo "✅ ViT-Small Experiment Completed!"
echo ""

# ==========================================
# 종료
# ==========================================
echo "🎉 All Experiments Finished Successfully!"