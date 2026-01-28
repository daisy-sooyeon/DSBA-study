#!/bin/bash

# 에러가 나면 즉시 스크립트 중단
set -e

# ==========================================
# 1. 실험할 모델과 데이터셋 설정
# ==========================================
MODELS=("resnet50" "resnet50_pretrained" "vit_small" "vit_small_pretrained") 
DATASETS=("cifar10") 
GPU_ID=0  

# ==========================================
# 2. 이중 반복문으로 실험 자동 수행
# ==========================================
for MODEL in "${MODELS[@]}"
do
    for DATA in "${DATASETS[@]}"
    do
        echo "========================================================"
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
        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
            --model "$MODEL" \
            --data "$DATA" 
        
        IS_PRETRAINED=$(python -c "import yaml; print(yaml.safe_load(open('./configs/models/${MODEL}.yaml')).get('pretrained', False))")
        
        echo "   -> Detected Pretrained Status: $IS_PRETRAINED"

        # (3) 학습 결과 파일 이동 및 정리
        # 파일명 형식: {DATA}_{MODEL}_{IS_PRETRAINED}_best.pth
        SOURCE_FILE="./logs/${DATA}_${MODEL}_${IS_PRETRAINED}_best.pth"
        
        if [ -f "$SOURCE_FILE" ]; then
            mv "$SOURCE_FILE" "$SAVE_DIR/best_model.pth"
            echo "   -> Moved best model to: $SAVE_DIR/best_model.pth"
        else
            echo "⚠️ Warning: Best model file not found: $SOURCE_FILE"
        fi
        
        # 설정 파일 백업
        cp "./configs/models/${MODEL}.yaml" "$SAVE_DIR/model_config.yaml"
        cp "./configs/data/${DATA}.yaml" "$SAVE_DIR/data_config.yaml"

        # (4) 강건성 평가 실행 (Eval)
        echo "Step 2. Evaluating Robustness on $DATA_ROOT_C..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python eval_robustness.py \
            --config "./configs/models/${MODEL}.yaml" \
            --weights "$SAVE_DIR/best_model.pth" \
            --data_root "$DATA_ROOT_C"

        REAL_MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model_name'])")

        # (5) 평가 결과 이동
        if [ -f "./logs/robustness_results_${REAL_MODEL_NAME}_${IS_PRETRAINED}.csv" ]; then
            mv "./logs/robustness_results_${REAL_MODEL_NAME}_${IS_PRETRAINED}.csv" "$SAVE_DIR/robustness_results.csv"
        fi

        echo "✅ Experiment Finished for $MODEL on $DATA"
        echo "📂 Results saved to: $SAVE_DIR"
        echo ""
    done
done

echo "🎉 All Experiments Completed!"