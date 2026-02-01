## 🎯 Hydra 기반 설정 사용 방법

### 기본 실행
```bash
# 기본 설정으로 resnet50 학습
python main.py

# 특정 모델과 데이터셋 선택
python main.py model=vit_small_pretrained dataset=cifar10
```

### 설정 오버라이드
```bash
# 에포크 수 변경
python main.py train.epochs=100

# 학습률 변경
python main.py train.learning_rate=0.01

# 배치 크기 변경
python main.py train.batch_size=32

# 옵티마이저 변경
python main.py train.optimizer=adamw
```

### 여러 설정 조합
```bash
# 여러 설정을 한 번에 변경
python main.py model=vit_small train.epochs=30 train.learning_rate=1e-4

# ImageNet-9 모델 사용
python main.py model=resnet50_pretrained_in9 train.epochs=5

# 설정 파일 이름으로 모든 것을 한 번에 변경
python main.py model=resnet50_pretrained_in9
```

### 현재 설정 확인
```bash
# 기본 설정 확인
python main.py --info

# 특정 설정의 최종 결과 보기
python main.py model=vit_small --cfg job
```

---

## 📁 설정 파일 구조

```
configs/
├── config.yaml           # 메인 설정 파일 (기본값)
├── model/               # 모델 설정들
│   ├── resnet50.yaml
│   ├── resnet50_pretrained.yaml
│   ├── resnet50_pretrained_in9.yaml
│   ├── vit_small.yaml
│   ├── vit_small_pretrained.yaml
│   └── vit_small_pretrained_in9.yaml
├── dataset/             # 데이터셋 설정
│   └── cifar10.yaml
└── data/               # 기타 데이터 설정 (선택사항)
```

---

## 💡 Hydra 장점

✅ **명령줄에서 직관적으로 설정 변경** - config 파일 수정 불필요  
✅ **설정값이 계층적으로 구조화** - 모델, 데이터, 학습 설정을 독립적으로 관리  
✅ **실험 자동 추적** - 각 실행마다 `.hydra` 폴더에 사용한 설정이 자동 저장  
✅ **다양한 조합 실험 가능** - sweep으로 여러 설정을 자동 조합 실행  
✅ **기본값 오버라이드 용이** - 명령행에서 설정 재정의 가능

---

## 🚀 고급 사용법

### 설정 확인 후 실행
```bash
python main.py model=vit_small --cfg job 2>&1 | head -50  # 최종 설정 미리보기
```

### 일관된 실험 기록
모든 실행 시 `outputs/` 디렉토리에 다음이 자동으로 저장됨:
- `.hydra/config.yaml` - 실행한 설정
- `.hydra/hydra.yaml` - Hydra 자체 설정  
- `.hydra/overrides.txt` - 변경한 파라미터들

이를 통해 어떤 설정으로 실험했는지 정확히 추적 가능!
