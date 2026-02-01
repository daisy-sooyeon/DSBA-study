#!/bin/bash

# 에러가 나면 즉시 스크립트 중단
set -e

# ==========================================
# 1. 실험할 모델과 데이터셋 설정
# ==========================================
MODELS=("vit_small") 
DATASETS=("cifar10") 
GPU_ID=0  

# ==========================================
# 2. 이중 반복문으로 실험 자동 수행
# ==========================================
for MODEL in "${MODELS[@]}"
do
    for DATA in "${DATASETS[@]}"
    do
        echo "======================================================c=="
        echo "🚀 Starting Experiment: [ Model: $MODEL | Data: $DATA ]"
        echo "========================================================"

        if [ "$DATA" == "cifar10" ]; then
            DATA_ROOT_C="./data/CIFAR-10-C"
        else
            echo "❌ Error: Unknown dataset '$DATA'. Please check the path settings."
            exit 1
        fi

        # (1) 결과 저장할 폴더 만들기
        EXP_NAME="${MODEL}_${DATA}"
        SAVE_DIR="./checkpoints/${EXP_NAME}"
        mkdir -p "$SAVE_DIR"

        # (2) 학습 실행 (Train)
        echo "Step 1. Training ($MODEL on $DATA)..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py model=$MODEL dataset=$DATA 
        

        # (3) 학습 결과 파일 이동 및 정리
        # 파일명 형식: {DATA}_{MODEL}_{is_pretrained}_best.pth
        IS_PRETRAINED=$(python -c "import yaml; pretrained = yaml.safe_load(open('./configs/model/${MODEL}.yaml')).get('model', {}).get('pretrained', False); print(str(pretrained))")
        echo "   -> Detected Pretrained Status: $IS_PRETRAINED"

        # (3) 학습 결과 파일 이동 및 정리
        # 파일명 형식: {DATA}_{MODEL}_best.pth
        SOURCE_FILE="./logs/${DATA}_${MODEL}_best.pth"
        
        if [ -f "$SOURCE_FILE" ]; then
            mv "$SOURCE_FILE" "$SAVE_DIR/best_model.pth"
            echo "   -> Moved best model to: $SAVE_DIR/best_model.pth"
        else
            echo "⚠️ Warning: Best model file not found: $SOURCE_FILE"
        fi
        
        # 설정 파일 백업
        cp "./configs/model/${MODEL}.yaml" "$SAVE_DIR/model_config.yaml"
        cp "./configs/dataset/${DATA}.yaml" "$SAVE_DIR/data_config.yaml"

        # (4) 강건성 평가 실행 (Eval)
        echo "Step 2. Evaluating Robustness on $DATA_ROOT_C..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python -c "
from src.eval_robustness import main
from hydra import initialize_config_dir, compose
import os

config_dir = os.path.abspath('./configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='config', overrides=['model=$MODEL', 'robustness.data_root=$DATA_ROOT_C'])
    main(cfg, weights_path='$SAVE_DIR/best_model.pth')
"
        
        # (5) 평가 결과 이동
        if [ -f "./logs/robustness_results_${MODEL}.csv" ]; then
            mv "./logs/robustness_results_${MODEL}.csv" "$SAVE_DIR/robustness_results.csv"
        fi

        echo "✅ Experiment Finished for $MODEL on $DATA"
        echo "📂 Results saved to: $SAVE_DIR"
        echo ""
    done
done

echo "🎉 All Experiments Completed!"