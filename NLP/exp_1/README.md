# IMDB Sentiment Classification with BERT and ModernBERT

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

본 실험은 IMDB 영화 리뷰 데이터셋을 활용한 감정 분류(sentiment classification) 작업으로, BERT와 ModernBERT 두 가지 Transformer 기반 모델의 성능을 비교하고 평가하기 위한 실험이다.

BERT(Bidirectional Encoder Representations from Transformers)는 양방향 문맥 이해가 가능한 사전 훈련된 언어 모델이며, ModernBERT는 최근에 제안된 BERT의 개선된 버전으로 더 효율적인 아키텍처와 훈련 방식을 채택하고 있다. 본 실험을 통해 두 모델이 텍스트 분류 작업에서 어떤 차이를 보이는지 확인하고, 각 모델의 특성을 분석하고자 한다.

### Key Objectives
1. BERT와 ModernBERT 모델의 IMDB 감정 분류 성능 비교
2. 성능 차이의 원인 규명
3. 코드 작성 시 주의해야 할 점 파악

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
│       ├── bert.yaml        # BERT 모델 설정
│       └── modernbert.yaml  # ModernBERT 모델 설정
├── src/
│   ├── main.py              # 메인 훈련 스크립트
│   ├── analyze_errors.py    # t-SNE 경계 오분류 / attention 분석 스크립트
│   ├── model.py             
│   ├── data.py              # 데이터 로딩 및 전처리
│   └── utils.py             
├── scripts/
│   ├── run_all_models.sh    # 모든 모델 자동 실행 스크립트
│   └── analyze_errors.sh    # 오분류 역추적/attention 비교 실행 스크립트
├── analysis_outputs/        # 분석 결과 저장
├── logs/                    
├── checkpoints/             # 모델 체크포인트 저장 디렉터리
├── outputs/                 # Hydra 출력 파일 저장 디렉터리
└── wandb/                   # WandB 실험 추적 파일 저장 디렉터리
```

---

# Setup & Usage

### Quick Start

**Step 1. Install dependencies**
```bash
cd /workspace/NLP/exp_1
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

---

# Training Configuration

각 모델에 사용된 학습 설정은 관련 모델별 YAML 파일에서 확인 가능

#### Data
- Dataset: IMDB (stanfordnlp/imdb)
- Max Sequence Length: 128 tokens
- Train/Validation/Test Split: 80% / 10% / 10%
- Batch Size: 8
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

### Experiment: BERT vs ModernBERT Comparison

- Dataset: IMDB
  - Train set: 전체 데이터의 80%
  - Validation set: 전체 데이터의 10%
  - Test set: 전체 데이터의 10%
- Models: BERT-base-uncased (`bert`), ModernBERT-base (`modernbert`)
   - `EncoderForClassification`: 사전 훈련된 encoder 위에 classification head를 추가한 구조
   - Mean pooling을 통해 모든 token representation을 집계
   - Linear layer를 통한 이진 분류

#### Run all training jobs
```bash
cd /workspace/NLP/exp_1
bash scripts/run_all_models.sh
```

#### Output

The results will be saved to:

```
exp_1/
├── checkpoints/
│   ├── bert_base_uncased/
│   │   └── epoch_*_acc_*.pt
│   └── modernbert_base/
│       └── epoch_*_acc_*.pt
├── outputs/
│   └── main.log
└── wandb/
    └── run-*/
```

Saved files include:

| File                        | Description                         |
|-----------------------------|-------------------------------------|
| `checkpoints/{model_name}/epoch_*_acc_*.pt` | Checkpoint saved every epoch (filename includes val acc) |
| `outputs/main.log` | Hydra output log file        |
| `wandb/run-*/`              | WandB experiment tracking files     |
| `analysis_outputs/*`        | Error analysis outputs (t-SNE boundary samples, attention logs/images) |

### Result

| model | val_acc@epoch | test_loss | test_acc |
| :--- | :---: | :---: | :---: |
| bert-base-uncased | 0.8832@3 | 0.3428 | 0.8842 |
| modernbert-base | 0.9104@1 | 0.2096 | **0.9154** |

> Test는 validation accuracy 기준 best epoch의 checkpoint를 로드하여 진행함 (checkpoint는 매 epoch 저장)

### Evaluation

- **t-SNE 비교**

#### Run all training jobs
```bash
cd /workspace/NLP/exp_1
bash scripts/analyze_errors.sh
```

#### Output

Saved files include:

| File                        | Description                         |
|-----------------------------|-------------------------------------|
| `analysis_outputs/*`        | Error analysis outputs (t-SNE boundary samples, attention logs/images) |

<div align="center">
<table>
<tr>
<td align="center" width="50%">
<strong>Column 1</strong><br><br>
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/51360741-9bbb-491b-881b-189e0b9f9a46" /><br><br>
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/f3ec0fdb-4579-461f-af6a-c4e9f0c80afc" /><br><br>
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/459280b3-eb2a-45cb-9cd7-bac072a365bf" /><br><br>
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/1597e5e7-1be0-4f23-81ed-0e489cdd260d" /><br><br>
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/4dfa61a9-ab80-49e8-b4a4-8c56914466dd" />
</td>
<td align="center" width="50%">
<strong>Column 2</strong><br><br>
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/ca20f72a-339f-4179-9ce2-1bb316c18e06" /><br><br>
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/f1868382-31f6-4e2b-8480-5a9c91db9825" /><br><br>
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/c5ce47ec-f382-45cf-b039-cba9dda8c5d0" /><br><br>
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/c43f443b-a3a5-4b27-ac34-6decdddd57bd" /><br><br>
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/79fb4a37-4e7c-4ed8-8409-7e0205bb83a1" />
</td>
</tr>
</table>

</div>

이렇게만 보면 BERT가 더 잘하는 건가 싶지만, BERT는 데이터를 단순하게 파악하는 반면 ModernBERT는 복잡한 의미론적 정보를 파악하기 때문에 2차원으로 나타내기 어려웠던 것일 수도 있다.

- **Attention Map 확인**

<div align="center">
  <img width="900" alt="Attention Map" src="Image" src="https://github.com/daisy-sooyeon/DSBA-study/issues/10#issue-3889241781" />
</div>

BERT는 특정 토큰에 집중하여 보고 있는 상황이라 문맥 전체의 의미를 특정 지점으로 응축하려는 경향이 있고, ModernBERT는 특정 단어에 집중하면서도 자기 자신 및 주변 단어과의 강한 관계를 유지함으로써 문장 전체에 대해 다각적인 정보에 집중하는 상황이라고 이해할 수 있다. 

> 실제로 ModernBERT는 Rotary Positional Embeddings (RoPE)를 통해 상대적인 위치 정보를 반영하고, Local-Global Alternating Attention이 가능하며, BERT에 비해 많은 token을 한 번에 볼 수 있기 때문에 유리한 위치에 있다.

---

### 코드 작성 시 유의할 점

1. 모델 간 input spec의 차이 고려

`token_type_ids`는 BERT 모델만 받는 입력으로, 몇 번째 문장인지 구분하기 위해 필수적으로 받는(혹은 생성하는) input이다. ModernBERT의 경우 이 input을 사용하지 않아 인자를 받을 때 유의해서 코드를 작성해야 한다. 

2. Pooling 시 padded token 포함 여부

현재 분류 과제를 수행하기 위해 mean pooling을 하고 있는데, 이때 padding 값들을 평균 계산에 포함시키면 문장이 짧을 경우 padding 값들이 결과에 영향을 줄 수 있다. 따라서 masked mean pooling을 사용하여 padding을 제외한 pooling을 진행하여야 한다. 
