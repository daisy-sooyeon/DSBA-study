# IMDB Sentiment Classification: Effect of Target Batch Sizes via Gradient Accumulation

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

본 실험은 IMDB 영화 리뷰 데이터셋을 활용한 감정 분류(sentiment classification) 작업으로, Gradient Accumulation을 통해 각기 다른 batch size로 훈련을 진행했을 때 Transformer 기반 모델의 성능과 학습 양상이 어떻게 변화하는지 비교 및 분석하기 위한 실험이다.

일반적으로 배치 사이즈가 커질수록 GPU 병렬 처리 효율은 올라가지만, 모델이 local minima에 빠지기 쉬워져 일반화 성능이 하락하는 경향이 있다. 본 실험을 통해 BERT와 ModernBERT 두 가지 모델이 배치 사이즈 크기 변화에 따라 어떤 성능 차이를 보이는지 확인하고, Attention Map을 통해 모델의 내부 동작 방식을 시각적으로 분석하고자 한다.

### Key Objectives
1. Gradient Accumulation 기법을 적용하여 메모리 한계를 극복하고 대규모 Batch Size로 학습하는 방법론 이해
2. Target Batch Size(64, 256, 1024) 변화에 따른 모델의 수렴 속도 및 최종 Test Accuracy 비교 분석
3. 순수 PyTorch와 Huggingface Accelerate를 각각 활용한 Gradient Accumulation 코드 구현

---

# Datasets & Models

## Datasets

- **IMDB**: 영화 리뷰 감정 분류 데이터셋
  - 출처: Hugging Face `stanfordnlp/imdb`
  - 클래스: Positive (1), Negative (0)
  - 데이터 분할: Train (80%), Validation (10%), Test (10%)
  - 최대 시퀀스 길이: 128 tokens

## Models

#### Transformer-based Models:
- **BERT-base-uncased**: Google에서 개발한 양방향 Transformer 기반 언어 모델
  - 모델명: `bert-base-uncased`
  - Hidden size: 768
  - Parameters: ~110M
  
- **ModernBERT-base**: BERT의 개선된 버전으로 더 효율적인 아키텍처 채택
  - 모델명: `answerdotai/ModernBERT-base`
  - Hidden size: 768
  - Parameters: ~110M

---

# Project Structure

```
exp_1/
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   ├── config.yaml          
│   └── model/
│       ├── bert.yaml                # BERT 모델 설정
│       └── modernbert.yaml          # ModernBERT 모델 설정
├── src/
│   ├── main.py                      # 메인 훈련 스크립트
│   ├── analyze_errors.py            # t-SNE 경계 오분류 / attention 분석 스크립트
│   ├── model.py             
│   ├── data.py                      # 데이터 로딩 및 전처리
│   ├── huggingface_accumulation.py  # accelerator 사용한 버전의 메인 훈련 스크립트
│   └── utils.py             
├── scripts/
│   ├── run_all_models.sh            # 모든 모델 자동 실행 스크립트
│   └── analyze_models.sh            # attention 비교 실행 스크립트
├── analysis_outputs/                # 분석 결과 저장
├── logs/                    
├── checkpoints/                     # 모델 체크포인트 저장 디렉터리
├── outputs/                         # Hydra 출력 파일 저장 디렉터리
└── wandb/                           # WandB 실험 추적 파일 저장 디렉터리
```

---

# Setup & Usage

### Quick Start

**Step 1. Install dependencies**
```bash
cd /workspace/NLP/exp_2
pip install -r requirements.txt
```

**Step 2. Set WandB API key**
```bash
wandb login [API_KEY]
```

**Step 3. Run experiments**

단일 모델 훈련:
```bash
# BERT 모델 훈련
python src/main.py model=bert

# ModernBERT 모델 훈련
python src/main.py model=modernbert
```

또는 스크립트 사용:
```bash
# 모든 모델 자동 실행
bash scripts/run_all_models.sh
```

huggingface의 accelerate를 사용하고 싶은 경우:

`scripts/run_all_models.sh`의 `main.py` 부분을 `huggingface_accumulation.py`로 수정 후
```bash
bash scripts/run_all_models.sh
```

batch size에 따라 learning rate를 바꿔가며 학습하고 싶은 경우: 

`scripts/run_all_models.sh`의 `main.py` 부분을 `differentiate_lr.py`로 수정 후
```bash
bash scripts/run_all_models.sh
```

---

# Training Configuration

각 모델에 사용된 학습 설정은 관련 모델별 YAML 파일에서 확인 가능

#### Data
- Dataset: IMDB (stanfordnlp/imdb)
- Max Sequence Length: 128 tokens
- Train/Validation/Test Split: 80% / 10% / 10%
- **Target Batch Size: 64, 256, 1024**
- Physical Batch Size: 16
- Accumulation Steps: Target BS // Physical BS
- Tokenizer: 각 모델의 기본 tokenizer 사용

#### Models

- **BERT-base-uncased** (`bert`)
  - learning_rate: 5e-5
  - optimizer: "adam"
  - epochs: 5
  - scheduler: "constant"
  
- **ModernBERT-base** (`modernbert`)
  - learning_rate: 5e-5
  - optimizer: "adam"
  - epochs: 5
  - scheduler: "constant"

#### Training Details
- Loss Function: Cross Entropy Loss
- Seed: 42

#### Evaluation Metrics
- Accuracy
- Loss (Cross Entropy)

---

# Results & Evaluation

### Experiment: Batch Size Impact on Performance

동일한 Learning Rate 환경에서 Batch Size만 증가시켰을 때 발생하는 Generalization Gap 관찰

#### Run all training jobs
```bash
cd /workspace/NLP/exp_2
bash scripts/run_all_models.sh
```

#### Output

The results will be saved to:

```
exp_2/
├── checkpoints
│   ├── answerdotai
│   │   └── modernbert_base
│   │       ├── TargetBS_1024_answerdotai
│   │       │   └── ModernBERT-base
│   │       │       └── best_epoch_*_acc_*.pt
│   │       ├── TargetBS_256_answerdotai
│   │       │   └── ModernBERT-base
│   │       │       └── best_epoch_*_acc_*.pt
│   │       └── TargetBS_64_answerdotai
│   │           └── ModernBERT-base
│   │               └── best_epoch_*_acc_*.pt
│   └── bert_base_uncased
│       ├── TargetBS_1024_bert-base-uncased
│       │   └── best_epoch_*_acc_*.pt
│       ├── TargetBS_256_bert-base-uncased
│       │   └── best_epoch_*_acc_*.pt
│       └── TargetBS_64_bert-base-uncased
│           └── best_epoch_*_acc_*.pt
├── outputs/
│   └── main.log
└── wandb/
    └── run-*/
```

Saved files include:

| File                        | Description                         |
|-----------------------------|-------------------------------------|
| `checkpoints/{model_name}/{target_batch)size}/best_epoch_*_acc_*.pt` | Checkpoint saved every epoch (filename includes val acc) |
| `outputs/main.log` | Hydra output log file        |
| `wandb/run-*/`              | WandB experiment tracking files     |

### Result

| model | Target BS | val_acc@epoch | test_acc |
| :--- | :---: | :---: | :---: |
| bert-base-uncased | 64 | 0.9007@1 | **0.9013** |
| bert-base-uncased | 256 | 0.9041@2 | 0.9005 |
| bert-base-uncased | 1024 | 0.8971@3 | 0.8963 |
| modernbert-base | 64 | 0.9170@1 | **0.9204** |
| modernbert-base | 256 | 0.9138@4 | 0.9092 |
| modernbert-base | 1024 | 0.9120@2 | 0.9146 |

> Test는 validation accuracy 기준 best epoch의 checkpoint를 로드하여 진행함 (checkpoint는 매 epoch 저장)

<div align="center">
  <img width="900" alt="BERT Train Accuracy (Step)" src="https://github.com/user-attachments/assets/25ef28f6-af7e-462f-bd61-a48863d2ad46" />
</div>

<div align="center">
  <img width="900" alt="ModernBERT Train Accuracy (Step)" src="https://github.com/user-attachments/assets/a46fc8d8-7b61-4a4a-a253-5be58abff461" />
</div>

### Evaluation

배치 사이즈가 64에서 1024로 커질수록 두 모델 모두 전반적으로 test accuracy가 하락하는 경향을 보였다. 이는 배치가 클수록 gradient의 노이즈가 줄어들어, 학습 초기에 발견한 local minima에 빠져 test set에 대한 대처 능력이 떨어지기 때문으로 분석된다. 또한 validation accuracy는 반대로 batch size가 커질수록 더 높아지는 모습을 보이는데, 이를 통해 overfitting이 발생할 가능성이 배치가 클수록 증가함을 알 수 있다.

#### Plot Accuracy Map
```bash
cd /workspace/NLP/exp_2
bash scripts/analyze_models.sh
```

#### Output

Saved files include:

| File                        | Description                         |
|-----------------------------|-------------------------------------|
| `analysis_outputs/*`        | Attention map outputs |


- **Attention Map 확인**

<div align="center">
<table>
<tr>
<td align="center" width="50%">
<strong>bert-base-uncased</strong><br><br>

<img width="600" alt="BS 64" src="https://github.com/user-attachments/assets/cb6357cd-37af-4e68-abbe-fd06ae708159" /><br>
<em>Target Batch Size: 64</em><br><br>

<img width="600" alt="BS 256" src="https://github.com/user-attachments/assets/6e7c3d5b-ea21-4b42-8219-6a2a9e5d4aad" /><br>
<em>Target Batch Size: 256</em><br><br>

<img width="600" alt="BS 1024" src="https://github.com/user-attachments/assets/6f2adb42-2596-4926-bf0c-3f9f9ca8236e" /><br>
<em>Target Batch Size: 1024</em><br><br>

</td>
<td align="center" width="50%">
<strong>modernbert-base</strong><br><br>

<img width="600" alt="BS 64" src="https://github.com/user-attachments/assets/916e7bc2-dde1-4281-ae2c-9a9f2a4b91ed" /><br>
<em>Target Batch Size: 64</em><br><br>

<img width="600" alt="BS 256" src="https://github.com/user-attachments/assets/b4326351-9e6a-4194-8036-d8e0905d71bb" /><br>
<em>Target Batch Size: 256</em><br><br>

<img width="600" alt="BS 1024" src="https://github.com/user-attachments/assets/50d90c40-8ae6-45b8-8b5e-01a3e8674a8e" /><br>
<em>Target Batch Size: 1024</em><br><br>

</td>
</tr>
</table>
</div>

exp_1에서와 마찬가지로 BERT는 특정 토큰에 집중하여 보고 있고, ModernBERT는 특정 단어에 집중하면서도 자기 자신 및 주변 단어과의 강한 관계를 유지하는 모습을 확인할 수 있다. batch size(64, 256, 1024)에 따라 생성된 Attention Map 이미지를 비교한 결과, 거시적인 어텐션 패턴에는 큰 차이가 발견되지 않았다.

Colorbar 수치를 확인하면, BERT의 최대 어텐션 가중치는 약 0.16인 반면, ModernBERT는 약 0.3 정도다. ModernBERT는 불필요한 노이즈를 줄이고 핵심 문맥에 2배 이상 강하게 집중하는 방식을 보여준다. 그리고 두 모델 모두 `to`, `the` 등 의미가 적은 토큰에 짙은 세로줄이 그어지는 현상이 관찰되었다. 이는 Softmax의 총합을 1로 맞춰야 하는 구조적 한계 때문에 남는 집중도를 안전하게 버리는 Attention Sink 현상이다. ModernBERT는 BERT에 비해 이 쓰레기통 역할을 하는 토큰의 수를 제한하여 어텐션 자원을 훨씬 효율적으로 관리함을 시각적으로 확인할 수 있다.

---

### huggingface accelerator 사용 결과

PyTorch 사용한 버전과 동일 (seed 동일하기 때문)

- PyTorch: 배치 인덱스(i)를 추적하여 (i + 1) % accumulation_steps == 0 조건을 수동으로 계산해야 한다. 또, 지정된 Target Batch Size를 맞추기 위해 loss를 accumulation_steps로 직접 나누어 주어야 하며, 에폭의 마지막 자투리 배치(Remainder Batch)를 처리하기 위해 or (i + 1) == len(dataloader) 같은 예외 처리 로직이 강제된다.

- Huggingface Accelerate: with accelerator.accumulate(model):라는 Context Manager 블록 하나로 복잡한 수학적 계산과 예외 처리가 모두 대체된다. loss scaling과 더불어 epoch의 마지막에 도달했을 때 남은 자투리 데이터에 대한 가중치 업데이트까지 프레임워크가 자동으로 안전하게 처리한다.

---

### 공정한 비교인가?

epoch 수를 고정하고 있는 현재 상황에서는 batch size에 따라 가중치 업데이트 횟수가 차이나기 때문에 불공정한 비교일 수 있다. 하지만 step 수를 맞춰서 학습을 하게 되면 batch size가 클 때 컴퓨터 자원을 많이 사용하기 때문에 이 또한 공정한 비교라 보기 어렵다. 따라서 epoch 수는 고정하되, learning rate를 크게 하여 큰 배치에서의 한계를 보완한 경우에 대해서 실험을 진행해보았다.

- BS 64 learning rate: 5e-5
- BS 256 learning rate: 1e-4
- BS 1024 learning rate: 2e-4

#### Result

| model | Target BS | val_acc@epoch | test_acc |
| :--- | :---: | :---: | :---: |
| bert-base-uncased | 64 | 0.9007@1 | **0.9013** |
| bert-base-uncased | 256 | 0.8935@2 | 0.8951 |
| bert-base-uncased | 1024 | 0.8903@4 | 0.8901 |
| modernbert-base | 64 | 0.9170@1 | **0.9204** |
| modernbert-base | 256 | 0.9031@3 | 0.917 |
| modernbert-base | 1024 | 0.9152@2 | 0.9124 |

여전히 batch size가 작을수록 좋은 일반화 성능을 보임을 확인할 수 있다.

<div align="center">
  <img width="900" alt="BERT Train Accuracy (Step)" src="https://github.com/user-attachments/assets/5a945506-22c5-45d5-bc52-91a448356976" />
</div>

<div align="center">
  <img width="900" alt="ModernBERT Train Accuracy (Step)" src="https://github.com/user-attachments/assets/0b56e506-d876-45ef-8202-2926f91cbed2" />
</div>