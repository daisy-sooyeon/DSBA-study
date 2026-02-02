# Image Classification Robustness Comparison

---

# Table of Contents

1. [Overview](#overview)
2. [Datasets & Models](#datasets--models)  
3. [Project Structure](#project-structure)  
4. [Setup & Usage](#setup--usage)  
5. [Training Configuration](#training-configuration)  
6. [Results & Evaluation](#results--evaluation)

---

# Overview

본 실험은 CNN 기반 모델과 Transformer 기반 모델이 이미지 분류 과제에 있어 robustness(강건성)에서 어떤 차이를 보이는지 확인하기 위한 실험으로, image classification pipeline 구축 과정의 경험을 목적으로 한다.

우선적으로 이미지 전반에 노이즈를 추가한 상황에서의 robustness를 평가한다. Transformer 기반 모델들과 달리 CNN 기반 모델들은 예측 시 이미지 질감(texture)에 많은 영향을 받는다는 연구가 있어 이러한 노이즈에 민감할 것이라 예상되는데, 실험을 통해 이를 검증할 것이다. 더불어 fine-tuning을 통해 기존 모델들이 갖는 이러한 한계를 극복할 수 있을지 확인해보고자 한다. 마지막으로 이미지에 노이즈가 끼어 있는 경우 성능 하락이 발생하는 원인을 규명하기 위해 배경 의존도를 측정하여, 모델이 객체의 전체적인 형태가 아닌 주변 맥락에 과도하게 의존하는지 파악하고자 한다.

### Key Hypothesis
1. 질감보다 모양에 초점을 맞춘 모델인 Transformer 기반의 모델이 CNN 기반의 모델보다 robustness가 높을 것이며, pretrained되어 있을수록 더 강건성이 높을 것이다.
2. Fine-tuning을 진행할 시 모델 성능이 전반적으로 향상될 것이다. 기존에 낮은 성능을 보였던 모델의 성능 향상 정도가 가장 크게 나타날 것이다.
3. 물체가 아닌 배경에 대한 의존도는 CNN 기반의 모델이 Transformer 기반의 모델보다 높을 것이다. 즉, CNN 기반 모델의 배경 의존도가 더 높을 것이다.

---

# Datasets & Models

Datasets:

- **CIFAR-10**: 모델 훈련에 사용할 기본 데이터셋
- **CIFAR-10-C**: CIFAR 데이터셋에 총 19가지의 corruption이 적용된 데이터셋

> brightness, contrast, defocus blur, elastic, fog, frost, frosted glass blur, gaussian blur, gaussian noise, impluse noise, jpeg compression, motion blur, pixelate, saturate, shot_noise, snow, spatter, speckle noise, zoom blur

- **ImageNet-9**: 배경 의존성과 강건성 파악을 위한 데이터셋으로, ImageNet에서 가장 대표적인 9개 상위 클래스만 뽑고 물체가 아닌 배경을 바꾼 데이터셋

> 데이터 경로: `./data/ImageNet9/bg_challenge/`  
> `original`(Training 용), `mixed_rand`(배경을 다른 클래스의 배경으로 랜덤하게 바꾼 것, Test 용), `only_fg`(배경을 검은색으로 지우고 물체만 남긴 것, 보조)


Models: 
#### CNN-based Models:
- ResNet50 w/o pre-trained weights
- ResNet50 w/ pre-trained on ImageNet 1k

#### Transformer-based Models:
- ViT-S/16 w/o pre-trained weights
- ViT-S/16 w/ pre-trained on ImageNet 1k

---

# Project Structure

```
CV
├── README.md
├── main.py
├── configs
│   ├── config.yaml
│   ├── defaults.yaml
│   ├── dataset/
│   │   └── cifar10.yaml
│   └── model/
│       ├── resnet50.yaml
│       ├── resnet50_pretrained.yaml
│       ├── resnet50_pretrained_in9.yaml
│       ├── vit_small.yaml
│       ├── vit_small_pretrained.yaml
│       └── vit_small_pretrained_in9.yaml
├── data/                         
│   ├── cifar10/
│   ├── CIFAR-10-C/
│   └── ImageNet9/
│       └── bg_challenge/
│           ├── original/
│           ├── mixed_rand/
│           └── only_fg/
├── models/
│   ├── __init__.py
│   └── factory.py 
├── src/
│   ├── __init__.py
│   ├── eval_background_robustness.py
│   ├── eval_robustness.py
│   ├── finetune.py
│   └── train.py
├── run_background_all.sh
├── run_finetune_all.sh
├── run_all.sh
└── utils
    ├── __init__.py
    ├── common.py
    ├── constants.py
    ├── data_loader.py
    ├── imagenet_utils.py
    ├── logger.py
    └── metrics.py
```

# Setup & Usage


### Quick Start

**Step 1. Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2. Prepare datasets**

Place your dataset files in the following path:
```bash
./data/
```

데이터셋 구조:
- `./data/cifar10/` - CIFAR-10 데이터셋
- `./data/CIFAR-10-C/` - CIFAR-10-C 데이터셋
- `./data/ImageNet9/bg_challenge/` - ImageNet-9 데이터셋 (original, mixed_rand, only_fg 등 포함)

**Step 3. Set WandB API key**
```bash
wandb login [API_KEY]
```


# Training Configuration

각 모델에 사용된 학습 설정은 관련 모델별 YAML 파일에서 확인 가능

#### Data
- Datasets: CIFAR-10, CIFAR-10-C, ImageNet-9
- Input Size: 32x32 for CNN-based models, 224×224 for Transformer based models
- Epochs: 50 for models w/o pretrained weights, 20 for models w/ pretrained weights
- Batch Size: 64  
- Loss Function: Cross Entropy Loss  

#### Models

- CNN-based: `resnet50`  
    - w/o pretrained weights
        - learning_rate: 0.1
        - weight_decay: 5e-4
        - optimizer: "sgd"
        - momentum: 0.9
    - w/ pretrained weights
        - learning_rate: 0.01
        - weight_decay: 5e-4
        - optimizer: "sgd"
        - momentum: 0.9
- ViT-based: `vit_small_patch16_224`
    - w/o pretrained weights
        - learning_rate: 1e-4
        - weight_decay: 0.05
        - optimizer: "adamw"
    - w/ pretrained weights
        - learning_rate: 1e-5
        - weight_decay: 1e-4
        - optimizer: "adamw"

> `pretrained` value: `True` or `False`

선택된 최적화 파라미터들은 해당 모델이 가장 좋은 성능을 낼 수 있는 파라미터로 최대한 설정함

#### Evaluation Metrics

- Accuracy (Top-1)  
- Accuracy (Top-5)

# Results & Evaluation

### Experiment 1. Comparison of Overfitting Patterns

- Dataset: CIFAR-10
    - Train set: train(90%)/validation(10%) set으로 분리
    - Test
- Models: ResNet50 w/o pre-trained weights(`resnet50`), ResNet50 w/ pre-trained on ImageNet 1k(`resnet50_pretrained`), ViT-S/16 w/o pre-trained weights(`vit_small`), ViT-S/16 w/ pre-trained on ImageNet 1k(`vit_small_pretrained`)


#### Run all training jobs
```bash
cd /workspace/CV
bash run_all.sh
```

또는 개별 모델 학습:
```bash
python main.py model=resnet50_pretrained dataset=cifar10
python main.py model=vit_small_pretrained dataset=cifar10
```

Experiment 2는 Experiment 1에서 훈련된 최적 모델에 기반해 robustness를 평가하기 때문에 이 작업은 Experiment 1 & 2를 동시에 실행

#### Output

The results will be saved to:

```
/CV/checkpoints/{model_name}_{data_name}
```

Saved files include:

| File                        | Description                         |
|-----------------------------|-------------------------------------|
| `best_model.pth`            | Best model (based on validation)   |
| `data_config.yaml`             | Snapshot of data configs used            |
| `model_config.yaml`             | Snapshot of model configs        |


### Result

| model | val_acc(top1)* | val_acc(top5)* | val_acc*@epoch | test_loss | test_acc(top1) | test_acc(top5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| resnet50 | 72.14% | 97.40% | 45 | 0.8155 | 71.51% | 97.96% |
| resnet50_pretrained | 82.04% | 94.36% | 18 | 0.6358 | 82.10% | 99.01% |
| vit_small | 71.54% | 97.14% | 46 | 1.2877 | 72.15% | 97.35% |
| **vit_small_pretrained** | 98.76% | 100.00% | 18 | 0.0914 | **98.23%** | **99.94%** |

<div align="center">
  <table width="100%">
    <tr>
      <td width="50%" align="center">
        <img src="https://github.com/user-attachments/assets/57a768e5-a5cf-4b53-9d36-8e6f726dee11" width="100%" />
        <br>
        Fig 1. resnet50
      </td>
      <td width="50%" align="center">
        <img src="https://github.com/user-attachments/assets/f4b7e6e6-e529-4c55-91bc-690a3b5779bf" width="100%" />
        <br>
        Fig 2. resnet50_pretrained
      </td>
    </tr>
  </table>
</div>


<div align="center">
  <table width="100%">
    <tr>
      <td width="50%" align="center">
        <img src="https://github.com/user-attachments/assets/e05cc4e9-4ace-40b4-8ec8-837322ba2f14" width="100%" />
        <br>
        Fig 3. vit_small
      </td>
      <td width="50%" align="center">
        <img src="https://github.com/user-attachments/assets/f4b7e6e6-e529-4c55-91bc-690a3b5779bf" width="100%" />
        <br>
        Fig 4. vit_small_pretrained
      </td>
    </tr>
  </table>
</div>

> Best epoch은 top-1 validation accuracy를 기준으로 선정되며, 해당 시점의 모델이 best model로 저장됨

---

### Experiment 2. Comparison of Robustness

- Dataset: CIFAR-10-C
    - 19개의 corruption 종류, 5개의 severity 정도로 구성되어 있어 총 90 종류의 조합에 대해 각각 평가를 수행한다. 본 문서에서는 90 종류에 대한 평가 결과의 평균 정확도를 첨부한다.
- Models: ResNet50 w/o pre-trained weights(`resnet50`), ResNet50 w/ pre-trained on ImageNet 1k(`resnet50_pretrained`), ViT-S/16 w/o pre-trained weights(`vit_small`), ViT-S/16 w/ pre-trained on ImageNet 1k(`vit_small_pretrained`)


#### Output

The results will be saved to:

```
./checkpoints/{model_name}_{data_name}/
```

| File                        | Description                         |
|-----------------------------|-------------------------------------|
| `robustness_results.csv`            | top1_acc, top5_acc for all corruption, severity pairs   |


### Result 

| model | mean_acc(top1) | mean_acc(top5) |
| :--- | :---: | :---: |
| resnet50 | 62.45% | 94.82% |
| resnet50_pretrained | 69.67% | 96.44% |
| vit_small | 60.10% | 93.80% |
| **vit_small_pretrained** | **91.37%** | **98.94%** |

*Hypothesis 1*: 질감보다 모양에 초점을 맞춘 모델인 Transformer 기반의 모델이 CNN 기반의 모델보다 robustness가 높을 것이며, pretrained되어 있을수록 더 강건성이 높을 것이다. ☑️


---

### Experiment 3. Comparison of Improvement due to Fine-tuning

- Dataset: CIFAR-10-C
    - 모든 Corruption의 앞쪽 90% 데이터를 train 및 validation set, 뒤쪽 10% 데이터를 test set으로 사용하며, 앞쪽 90% 데이터 안에서 train(90%)/validation(10%) set으로 분리한다.
- Models: ResNet50 w/o pre-trained weights(`resnet50`), ResNet50 w/ pre-trained on ImageNet 1k(`resnet50_pretrained`), ViT-S/16 w/o pre-trained weights(`vit_small`), ViT-S/16 w/ pre-trained on ImageNet 1k(`vit_small_pretrained`)
- Epoch: 5

#### Run all training jobs
```bash
cd /workspace/CV
bash run_finetune_all.sh
```

#### Output

The results will be saved to:

```
./checkpoints/{model_name}_{data_name}/
```

| File                        | Description                         |
|-----------------------------|-------------------------------------|
| `finetune_results.csv`            | test top1_acc, top5_acc for all corruption, severity pairs calculated w/ the fine-tuned model   |


### Result 

| model | mean_acc(top1) | mean_acc(top5) | improvement(top1) compared to Exp.2 |
| :--- | :---: | :---: | :--: |
| resnet50 | 93.01% | 99.86% | +30.56%p
| resnet50_pretrained | 99.78% | 100.00% | +30.11%p
| vit_small | 99.62% | 99.99% | **+35.76%p**
| vit_small_pretrained | 99.92% | 100.00% | +8.55%p

*Hypothesis 2*: Fine-tuning을 진행할 시 모델 성능이 전반적으로 향상될 것이다. 기존에 낮은 성능을 보였던 모델의 성능 향상 정도가 가장 크게 나타날 것이다. 🔺

실제로 4가지 모델이 전부 유사한 성능을 보임을 확인할 수 있어, fine-tuning 시 CNN 기반, Transformer 기반 모델의 robustness 차이가 거의 사라짐을 알 수 있다. 


---

### Experiment 4. Background Robustness

- Dataset: ImageNet-9
    - original: 원본 이미지 데이터셋 (Training 용)
    - mixed_rand: 배경을 다른 클래스의 배경으로 랜덤하게 바꾼 데이터셋 (Test 용)
    - only_fg: 배경을 검은색으로 지우고 물체만 남긴 데이터셋 (보조)
    - `background_gap` = `original(test)_acc` - `mixed_rand_acc` 를 통해 배경 의존도 측정
- Models: ResNet50 w/ pre-trained on ImageNet 1k(`resnet50_pretrained`), ViT-S/16 w/ pre-trained on ImageNet 1k(`vit_small_pretrained`)
    - ImageNet 데이터셋에 적용하기에 적합한 pretrained 모델만 사용
- Epoch: 5


#### Run all training jobs
```bash
cd /workspace/CV
bash run_background_all.sh
```

또는 개별 모델 실행:
```bash
python -c "
from src.eval_background_robustness import main
from hydra import initialize_config_dir, compose
import os

config_dir = os.path.abspath('./configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='config', overrides=['model=resnet50_pretrained_in9'])
    main(cfg)
"
```

### Result 

| model | original_acc(test) | mixed_rand_acc | only_fg_acc | background gap
| :--- | :---: | :---: | :--: | :--: |
| resnet50_pretrained | Top-1: 94.81% / Top-5: 100.00% | Top-1: 78.27% / Top-5: 97.88% | Top-1: 88.84% / Top-5: 99.51% | 16.54%p |
| **vit_small_pretrained** | Top-1: 94.32% / Top-5: 99.75% | Top-1: 79.95% / Top-5: 98.07% | Top-1: 87.09% / Top-5: 98.72% | **14.37%p**

*Hypothesis 3*: 물체가 아닌 배경에 대한 의존도는 CNN 기반의 모델이 Transformer 기반의 모델보다 높을 것이다. 즉, CNN 기반 모델의 배경 의존도가 더 높을 것이다. ☑️

---

> ### 강건성은 Transformer 기반 모델이 CNN 기반 모델보다 더 좋으며, Transformer 기반 모델은 배경이 아닌 사물 자체에 더 집중하기 때문에 이러한 결과가 나타난다.

