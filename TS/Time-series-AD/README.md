# Time-series Anomaly Detection - TranAD

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

본 실험은 TranAD 기반의 시계열 이상 감지 실험으로, 다변량(multivariate) 설정에서 PSM 데이터셋에 대한 성능 평가를 진행하고자 한다.

### Key Objectives
1. TranAD 모델을 사용해 시계열 이상 감지 성능 평가
2. 시계열 데이터의 특성과 모델 입력 형태를 고려한 최적의 파이프라인 구축

---

# Datasets & Models

## Datasets

- **PSM (Pooled Server Metrics)**: eBay 서버 모니터링 지표
  - 구성: 25개의 변량 존재
  - 데이터 분할: Train set을 Train (80%) / Validation (20%) 로 분할

## Models

- **TranAD**: Transformer 기반의 이상 감지 모델로, 시계열 데이터를 재구성하여 이상을 감지
  - 아키텍처 설정: Transformer layers에 Adversarial Training & Meta-Learning 구조 적용

---

# Project Structure

```
TS/Time-series-AD/
├── README.md
├── requirements.txt
├── data/                      # 원시/전처리 데이터 위치
├── src/
│   ├── main.py                # Entrypoint (train / eval)
│   ├── exp_builder_dl.py    
│   ├── arguments.py      
│   ├── configs/
│   │   ├── default_setting.yaml
│   │   └── model_setting.yaml
│   ├── data_provider/
│   │   ├── build_dataset.py   
│   │   ├── factory.py   
│   │   └── load_dataset.py   
│   ├── dataset/
│   │   └── PSM/
│   │       ├── train.csv
│   │       ├── test.csv
│   │       └── test_label.csv
│   ├── layers/
│   ├── losses/
│   ├── models/
│   │   └── TranAD.py   
│   ├── optimizers/
│   ├── scripts/
│   │   └── TranAD/         
│   ├── utils/
└── └── saved_model/
```

---

# Setup & Usage

### Quick Start

**Step 1. Install dependencies**
```bash
cd TS/Time-series-AD
pip install -r requirements.txt
```

**Step 2. Set WandB API key**
```bash
wandb login [API_KEY]
```

**Step 3. Run experiments**

단일 세팅 실행:
```bash
accelerate launch --num_processes 1 src/main.py --model_name TranAD --default_cfg src/configs/default_setting.yaml --model_cfg src/configs/model_setting.yaml --opts DEFAULT.exp_name anomaly_detection_${data_name}_TranAD
```

또는 스크립트 사용:
```bash
bash src/scripts/run_all_TranAD.sh
```

---

# Training Configuration

#### Data
- Dataset: PSM
- Scaling: minmax scaler

#### Models

- stride_len: 1
- learning_rate: 0.01 / meta learning rate: 0.02
- optimizer: "adamw"
- epochs: 20
- early_stopping: true
- scheduler: "step decay"

#### Training Details
- Loss Function: MSE (Mean Squared Error) for reconstruction
- Seed: 42
- POT 방식으로 threshold 설정
  - Coefficient (q): 1e-4
  - Low quantile (level): 0.001

#### Evaluation Metrics
- F1-Score
- Precision
- Recall
- AUC
- HitRate@P%
- NDCG@P%

---

# Results & Evaluation

### Experiment: TranAD Results

PSM 데이터셋에 대한 TranAD 모델의 성능 확인

#### Output

The results will be saved to:

```
TS/Time-series-AD/src/
├── saved_model/
│   └── TranAD/
│       └── AnomalyDetection/version{n}/
│           ├── best_model.pt
│           ├── configs.yaml
│           ├── TEST_anomaly_plot.png
│           └── TranAD_PSM_test_results.json
└── outputs/

```

### Result

| Dataset | F1-Score | Precision | Recall | AUC | HitRate@100% | HitRate@150% | NDCG@100% | NDCG@150% | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PSM | 0.7642 | 0.6776 | 0.8761 | 0.7779 | 0.3967 | 0.7259 | 0.4208 | 0.7059 | 0.009 |

> Test는 validation loss 기준 best epoch의 `best_model.pt` 체크포인트를 로드하여 진행함.

### Evaluation

Recall이 Precision 대비 높게 측정된 것을 볼 때, 모델이 실제 이상치를 놓치지 않고 민감하게 잘 잡아내고 있음을 알 수 있다. 특히 AUC가 0.7779로 측정되어 전반적인 이상 탐지 변별력이 준수하게 확보되었다.

#### Plot Time-Series Anomaly Detection Example

PSM 데이터셋의 특정 구간에 대한 Anomaly Score 및 POT Threshold를 시각화한 결과다.

<div align="center">

<img src="https://github.com/user-attachments/assets/0d725845-129c-4669-8f4e-829e5bb21fb7" width="400" /><br>

</div>
