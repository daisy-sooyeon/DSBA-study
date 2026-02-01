#!/usr/bin/env python
"""
CV 모델 학습 및 평가 메인 진입점 (Hydra 기반)

사용 방법:
    python main.py train model=resnet50 dataset=cifar10
    python main.py train model=vit_small_pretrained dataset=cifar10 train.epochs=30
    python main.py eval_robustness model=resnet50 dataset=cifar10
    python main.py eval_background model=resnet50_pretrained_in9
"""

import hydra
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig, OmegaConf
import os
import sys

# src 모듈 임포트
from src.train import main as train_main


config_dir = os.path.abspath("./configs")


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 함수: Hydra 설정으로 실행"""
    
    # logs 폴더 생성
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    
    # 설정 출력
    print(OmegaConf.to_yaml(cfg))
    
    # 학습 실행
    train_main(cfg)


if __name__ == "__main__":
    main()
