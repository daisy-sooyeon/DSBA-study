## 📋 현재 실행 가능 상태

### ✅ 실행 가능한 스크립트

#### 1. **run_all.sh** - 모델 학습 및 강건성 평가
```bash
bash run_all.sh
```
- 4개 모델(resnet50, resnet50_pretrained, vit_small, vit_small_pretrained)을 cifar10에서 학습
- 각 모델마다 CIFAR-10-C 강건성 평가 수행
- 결과를 `checkpoints/{MODEL}_cifar10/` 에 저장

#### 2. **run_background_all.sh** - 배경 편향 강건성 평가
```bash
bash run_background_all.sh
```
- ImageNet-9 데이터셋으로 ResNet50과 ViT-Small 평가
- 배경 편향 강건성 테스트

#### 3. **run_finetune_all.sh** - 미세조정 및 강건성 평가
```bash
bash run_finetune_all.sh
```
- 사전학습 모델(resnet50_pretrained, vit_small_pretrained)을 CIFAR-10-C로 미세조정
- 각 모델별 강건성 평가 수행
- 결과를 `./logs/finetune_results_{model}_{pretrained}.csv` 에 저장

---

## 🚀 빠른 테스트

### 단일 모델 학습 (Hydra 명령)
```bash
# 기본 설정으로 resnet50 학습
python main.py model=resnet50 dataset=cifar10

# 파라미터 변경
python main.py model=vit_small dataset=cifar10 train.epochs=30

# 강건성 평가 (별도 스크립트 필요)
python main.py model=resnet50 dataset=cifar10 robustness.data_root=./data/CIFAR-10-C
```

---

## 📝 구현 완료

모든 3개의 스크립트가 **Hydra 기반**으로 완전 구현되었습니다:
- ✅ `src/train.py` - Hydra DictConfig 지원
- ✅ `src/finetune.py` - Hydra DictConfig + weights_path 파라미터 지원
- ✅ `src/eval_robustness.py` - Hydra DictConfig + weights_path 파라미터 지원  
- ✅ `src/eval_background_robustness.py` - Hydra DictConfig 지원

필요하면 알려주세요!
