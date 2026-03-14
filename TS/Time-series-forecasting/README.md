# Time-series Forecasting — Informer

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

본 실험은 Informer 기반의 시계열 예측 실험으로, 단변량(univariate) 및 다변량(multivariate) 설정에서 다양한 prediction length에 대한 성능 비교를 진행하고자 한다.

### Key Objectives
1. Informer 모델을 사용해 긴 시계열 예측 성능 평가
2. Prediction lengths: 24, 48, 168, 336, 720, 960에 대한 단/다변량 비교
3. 시계열 데이터의 특성과 모델 입력 형태를 고려한 최적의 파이프라인 구축

---

# Datasets & Models

## Datasets

- **ETT (Electricity Transformer Temperature)**: 전력 변압기 온도 데이터셋
  - 구성: ETTh1, ETTh2 (Hourly, 1시간 단위) / ETTm1, ETTm2 (Minutely, 15분 단위)
  - 타겟 변수: `OT` (Oil Temperature)
  - 데이터 분할: 총 20개월의 데이터를 Train (12개월) / Validation (4개월) / Test (4개월) 로 분할

## Models

- **InformerStack**: 다중 스케일의 시간적 흐름을 동시에 학습하기 위해 여러 인코더 스택을 쌓아 올린 Informer의 확장 구조
  - 아키텍처 설정: `e_layers=[3, 1]`
  - 단변량 (Univariate): `enc_in = 1`, `dec_in = 1`, `c_out = 1`
  - 다변량 (Multivariate): `enc_in = 7`, `dec_in = 7`, `c_out = 7`

---

# Project Structure

```
TS/Time-series-forecasting/
├── README.md
├── requirements.txt
├── data/                      # 원시/전처리 데이터 위치
├── src/
│   ├── main.py                # Entrypoint (train / eval)
│   ├── exp_builder.py    
│   ├── arguments.py      
│   ├── configs/
│   │   ├── default_setting.yaml
│   │   └── model_setting.yaml
│   ├── data_provider/
│   │   ├── build_dataset.py   
│   │   ├── factory.py   
│   │   └── load_dataset.py   
│   ├── dataset/
│   │   ├── ETTh1.csv
│   │   ├── ETTh2.csv
│   │   ├── ETTm1.csv
│   │   └── ETTm2.csv
│   ├── layers/
│   ├── losses/
│   ├── models/
│   │   └── Informer.py   
│   ├── optimizers/
│   ├── scripts/
│   │   └── Informer/          # pred_len별 실행 스크립트
│   ├── utils/
└── └── saved_model/
```

---

# Setup & Usage

### Quick Start

**Step 1. Install dependencies**
```bash
cd TS/Time-series-forecasting
pip install -r requirements.txt
```

**Step 2. Set WandB API key**
```bash
wandb login [API_KEY]
```

**Step 3. Run experiments**

단일 세팅 실행:
```bash
accelerate launch --num_processes 1 src/main.py --model_name InformerStack --default_cfg src/configs/default_setting.yaml --model_cfg src/configs/model_setting.yaml
```

또는 스크립트 사용:
```bash
bash src/scripts/run_all_Informer.sh
```

---

# Training Configuration

#### Data
- Dataset: ETT (ETTh1, ETTh2, ETTm1, ETTm2)
- Target Length (pred_len): 24, 48, 168, 336, 720, 960
- Context Length (seq_len): 96 (과거 관측 길이)
- Label Length (label_len): 48 (디코더 도움닫기 길이)
- Features: S (단변량) 또는 M (다변량)

#### Models

- d_model: 512(모델 내부의 은닉층(Hidden state) 차원 크기)
- n_heads: 8(Multi-head Attention의 헤드 개수)
- d_layers: 2(Decoder의 층 개수)
- learning_rate: 0.0001
- optimizer: "adam"
- epochs: 8
- early_stopping: true
- scheduler: "exponential"

#### Training Details
- Loss Function: MSE (Mean Squared Error)
- Seed: 42

#### Evaluation Metrics
- MSE
- MAE

---

# Results & Evaluation

### Experiment: Prediction Length & Feature Target Impact

동일한 시퀀스 길이(`seq_len=96`) 조건에서 예측해야 할 미래 구간(`pred_len`)이 늘어남에 따라 단변량/다변량 모델의 오차가 어떻게 변화하는지 관찰

#### Output

The results will be saved to:

```
TS/Time-series-forecasting/src/
├── saved_model/
│   └── InformerStack/
│       └── forecasting_ETTh1_InformerStack_96_24/  # DEFAULT.exp_name 기준 분리
│           ├── best_model.pt
│           ├── configs.yaml
│           └── InformerStack_ETTh1_test_results.json
└── outputs/

```

### Result

| Dataset | Features | Pred_len | MSE | MAE |
| --- | --- | --- | --- | --- |
| ETTh1 | S | 24 | 0.0883 | 0.2352 |
| ETTh1 | S | 48 | 0.1665 | 0.3285 |
| ETTh1 | S | 168 | 0.238 | 0.4158 |
| ETTh1 | S | 336 | 0.2364 | 0.4118 |
| ETTh1 | S | 720 | 0.2333 | 0.4045 |
| ETTh1 | S | 960 | 0.2042 | 0.3759 |
| ETTh1 | M | 24 | 0.8472 | 0.6695 |
| ETTh1 | M | 48 | 0.9134 | 0.706 |
| ETTh1 | M | 168 | 0.9506 | 0.7422 |
| ETTh1 | M | 336 | 1.0362 | 0.7889 |
| ETTh1 | M | 720 | 1.2765 | 0.8913 |
| ETTh1 | M | 960 | 1.2518 | 0.8944 |
| ETTh2 | S | 24 | 0.0893 | 0.2369 |
| ETTh2 | S | 48 | 0.1626 | 0.3245 |
| ETTh2 | S | 168 | 0.232 | 0.4102 |
| ETTh2 | S | 336 | 0.2346 | 0.4098 |
| ETTh2 | S | 720 | 0.2329 | 0.4041 |
| ETTh2 | S | 960 | 0.2054 | 0.3773 |
| ETTh2 | M | 24 | 0.8331 | 0.6606 |
| ETTh2 | M | 48 | 0.9049 | 0.7006 |
| ETTh2 | M | 168 | 0.9485 | 0.7406 |
| ETTh2 | M | 336 | 1.036 | 0.7887 |
| ETTh2 | M | 720 | 1.2758 | 0.8914 |
| ETTh2 | M | 960 | 1.2521 | 0.8942 |
| ETTm1 | S | 24 | 0.1109 | 0.2679 |
| ETTm1 | S | 48 | 0.2458 | 0.41 |
| ETTm1 | S | 168 | 0.2382 | 0.4116 |
| ETTm1 | S | 336 | 0.2383 | 0.4125 |
| ETTm1 | S | 720 | 0.3164 | 0.4782 |
| ETTm1 | S | 960 | 0.2768 | 0.4512 |
| ETTm1 | M | 24 | 0.9203 | 0.6819 |
| ETTm1 | M | 48 | 0.8803 | 0.6752 |
| ETTm1 | M | 168 | 1.0066 | 0.7846 |
| ETTm1 | M | 336 | 1.1159 | 0.8395 |
| ETTm1 | M | 720 | 1.2655 | 0.8946 |
| ETTm1 | M | 960 | 1.2658 | 0.9103 |
| ETTm2 | S | 24 | 0.1146 | 0.2741 |
| ETTm2 | S | 48 | 0.2413 | 0.4066 |
| ETTm2 | S | 168 | 0.2352 | 0.4088 |
| ETTm2 | S | 336 | 0.2491 | 0.413 |
| ETTm2 | S | 720 | 0.314 | 0.4761 |
| ETTm2 | S | 960 | 0.2777 | 0.4521 |
| ETTm2 | M | 24 | 0.9141 | 0.6848 |
| ETTm2 | M | 48 | 0.8753 | 0.6756 |
| ETTm2 | M | 168 | 1.0042 | 0.7832 |
| ETTm2 | M | 336 | 1.1148 | 0.8389 |
| ETTm2 | M | 720 | 1.2674 | 0.8935 |
| ETTm2 | M | 960 | 1.2657 | 0.9103 |

> Test는 validation loss 기준 best epoch의 `best_model.pt` 체크포인트를 로드하여 진행함.

### Evaluation

일반적으로 `pred_len`이 길어질수록 불확실성이 커져 오차(MSE, MAE)가 증가하는 경향을 보인다. 다만 단변량(S) 설정에서 `pred_len`이 720에서 960으로 늘어날 때, 오히려 오차가 소폭 감소하거나 정체되는 현상이 관찰되는데, 이를 보아 Informer 모델이 데이터의 장기 주기성을 성공적으로 포착하고 있음을 알 수 있다. 더불어 1시간 단위인 `ETTh` 데이터가 15분 단위인 `ETTm` 데이터보다 근소하게 더 낮은 오차를 보임도 확인하였다. 이는 15분 단위 데이터(`ETTm`)는 더 짧은 시간 내의 미세한 노이즈와 변동성을 포함하고 있어, 모델이 트렌드를 잡는 데 더 어려움을 겪는 것으로 분석된다.

#### Plot Time-Series Forecasting Example

다음은 ETTh1 데이터셋의 첫 번째 샘플에 대해 예측 결과를 시각화한 결과다.

<div align="center">

| **Univariate (단변량)** | **Multivariate (다변량)** |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/b1abac5d-f740-4f86-a1cf-cccc27e273ab" width="400" /><br><em>Prediction Length: 24</em> | <img src="https://github.com/user-attachments/assets/d444158e-adab-4144-8221-b93d320b05f0" width="400" /><br><em>Prediction Length: 24</em> |
| <img src="https://github.com/user-attachments/assets/7d6b64c9-02ac-46cb-9fe3-da4bdb2c63a2" width="400" /><br><em>Prediction Length: 48</em> | <img src="https://github.com/user-attachments/assets/a891a630-cca1-49a3-a613-779e3d95b286" width="400" /><br><em>Prediction Length: 48</em> |
| <img src="https://github.com/user-attachments/assets/f001dcb0-d663-4467-b6cc-67d489c9e936" width="400" /><br><em>Prediction Length: 168</em> | <img src="https://github.com/user-attachments/assets/da27e150-1a8e-44bb-a47b-465736f016c4" width="400" /><br><em>Prediction Length: 168</em> |
| <img src="https://github.com/user-attachments/assets/ecc9ed2c-3d9a-4344-a1cd-d9b6e3c5bcf0" width="400" /><br><em>Prediction Length: 336</em> | <img src="https://github.com/user-attachments/assets/7eb62aae-5947-4e5f-bbdd-2c6ef3de33b4" width="400" /><br><em>Prediction Length: 336</em> |
| <img src="https://github.com/user-attachments/assets/81a21b26-113b-45d8-8597-9793722af4ea" width="400" /><br><em>Prediction Length: 720</em> | <img src="https://github.com/user-attachments/assets/68a6bb2f-032d-44ac-95f6-b35cca8a5811" width="400" /><br><em>Prediction Length: 720</em> |
| <img src="https://github.com/user-attachments/assets/876f7d8a-7c84-4029-a402-f3deb4627b53" width="400" /><br><em>Prediction Length: 960</em> | <img src="https://github.com/user-attachments/assets/b684109d-841f-4c22-83b0-82fd333faf3b" width="400" /><br><em>Prediction Length: 960</em> |

</div>

### Extra Experiment: RevIN Utilization

RevIN 방식을 적용한 후 성능 차이가 어떻게 나는지 확인

| Dataset | Features | Pred_len | MSE | MAE |
| --- | --- | --- | --- | --- |
| ETTh1 | S | 24 | 0.0359 | 0.1443 |
| ETTh1 | S | 48 | 0.0521 | 0.1767 |
| ETTh1 | S | 168 | 0.1041 | 0.2472 |
| ETTh1 | S | 336 | 0.1397 | 0.2966 |
| ETTh1 | S | 720 | 0.2058 | 0.3653 |
| ETTh1 | S | 960 | 0.2086 | 0.3694 |
| ETTh1 | M | 24 | 0.7572 | 0.5862 |
| ETTh1 | M | 48 | 0.0162 | 0.6372 |
| ETTh1 | M | 168 | 1.0723 | 0.7079 |
| ETTh1 | M | 336 | 1.6371 | 0.9616 |
| ETTh1 | M | 720 | 1.3854 | 0.9126 |
| ETTh1 | M | 960 | 1.5506 | 0.9789 |
| ETTh2 | S | 24 | 0.0356 | 0.1433 |
| ETTh2 | S | 48 | 0.0521 | 0.1769 |
| ETTh2 | S | 168 | 0.108 | 0.2519 |
| ETTh2 | S | 336 | 0.1405 | 0.2975 |
| ETTh2 | S | 720 | 0.1665 | 0.3261 |
| ETTh2 | S | 960 | 0.1966 | 0.3567 |
| ETTh2 | M | 24 | 0.774 | 0.5915 |
| ETTh2 | M | 48 | 0.9111 | 0.6375 |
| ETTh2 | M | 168 | 1.0413 | 0.6918 |
| ETTh2 | M | 336 | 1.6648 | 0.9691 |
| ETTh2 | M | 720 | 1.403 | 0.9196 |
| ETTh2 | M | 960 | 1.5566 | 0.9828 |
| ETTm1 | S | 24 | 0.0352 | 0.1438 |
| ETTm1 | S | 48 | 0.0558 | 0.1854 |
| ETTm1 | S | 168 | 0.1031 | 0.2453 |
| ETTm1 | S | 336 | 0.1335 | 0.2878 |
| ETTm1 | S | 720 | 0.1935 | 0.3543 |
| ETTm1 | S | 960 | 0.2099 | 0.3679 |
| ETTm1 | M | 24 | 0.8899 | 0.6282 |
| ETTm1 | M | 48 | 0.8927 | 0.6345 |
| ETTm1 | M | 168 | 1.1077 | 0.7238 |
| ETTm1 | M | 336 | 1.5687 | 0.9388 |
| ETTm1 | M | 720 | 1.5221 | 0.9556 |
| ETTm1 | M | 960 | 1.6245 | 0.9797 |
| ETTm2 | S | 24 | 0.0349 | 0.1432 |
| ETTm2 | S | 48 | 0.0555 | 0.1805 |
| ETTm2 | S | 168 | 0.1116 | 0.2524 |
| ETTm2 | S | 336 | 0.1309 | 0.2849 |
| ETTm2 | S | 720 | 0.1719 | 0.3309 |
| ETTm2 | S | 960 | 0.2062 | 0.3665 |
| ETTm2 | M | 24 | 0.799 | 0.5884 |
| ETTm2 | M | 48 | 0.8824 | 0.6338 |
| ETTm2 | M | 168 | 1.0643 | 0.7169 |
| ETTm2 | M | 336 | 1.5695 | 0.9388 |
| ETTm2 | M | 720 | 1.4963 | 0.9484 |
| ETTm2 | M | 960 | 1.6129 | 0.9762 |

단변량의 경우 RevIN을 적용한 방식이 distribution shift에 잘 적응할 수 있어 성능이 더 좋아지는 모습을 확인할 수 있다. 다만 다변량의 경우에는 오히려 성능이 떨어지는 모습을 볼 수 있다.

한편 scaling 진행 없이 RevIN만 사용 시 성능 저하가 크게 확인되어, 전체 series에 대한 scaling을 진행함과 동시에 RevIN을 적용한 성능을 첨부했다.