"""
Hydra 기반 강건성 평가 모듈 (CIFAR-10-C)
"""
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.factory import create_model_factory
from utils.data_loader import get_robustness_loader
from utils.logger import setup_logger
from utils.common import evaluate_accuracy_only
from utils.constants import CORRUPTIONS


def main(cfg: DictConfig, weights_path: str):
    """Hydra 설정을 사용한 강건성 평가
    
    Args:
        cfg: Hydra 설정
        weights_path: 모델 가중치 경로
    """
    
    is_pretrained = cfg.model.pretrained
    
    # 1. WandB 초기화
    wandb.init(
        project="cv",  
        name=f"eval_robustness_{cfg.model.cfg_name}",
        config=OmegaConf.to_container(cfg),
        tags=["evaluation", "robustness", cfg.model.cfg_name, str(is_pretrained)]
    )

    # 2. Logger 및 Device
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger("./logs", name=f"eval_robustness_{cfg.model.cfg_name}")
    logger.info(f"--- Robustness Evaluation: {cfg.model.cfg_name} (pretrained={is_pretrained}) ---")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # 3. 모델 로드
    logger.info(f"Loading model: {cfg.model.name} from {weights_path}")
    model = create_model_factory(cfg.model.name, cfg.model.num_classes, pretrained=False)
    
    checkpoint = torch.load(weights_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    
    results = []
    
    # 4. WandB Table 생성
    wandb_table = wandb.Table(columns=["model", "corruption", "severity", "top1_acc", "top5_acc"])

    logger.info("Starting Robustness Evaluation on CIFAR-10-C...")

    # 5. 모든 Corruption 평가
    data_root = cfg.robustness.data_root
    total_steps = len(CORRUPTIONS) * 5
    current_step = 0

    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            try:
                loader = get_robustness_loader(
                    data_root, 
                    corruption, 
                    severity, 
                    batch_size=cfg.robustness.batch_size,
                    image_size=cfg.model.image_size
                )
                
                acc1, acc5 = evaluate_accuracy_only(model, loader, device)
                
                result_row = {
                    'model': cfg.model.cfg_name,
                    'corruption': corruption,
                    'severity': severity,
                    'top1_acc': acc1,
                    'top5_acc': acc5 
                }
                results.append(result_row)
                
                wandb.log({
                    "eval/corruption": corruption,
                    "eval/severity": severity,
                    "eval/top1_acc": acc1,
                    "eval/top5_acc": acc5,    
                    f"detail/{corruption}_top1": acc1
                })
                
                wandb_table.add_data(cfg.model.cfg_name, corruption, severity, acc1, acc5)
                
                logger.info(f"[{corruption} Lv.{severity}] Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}%")
            
            except Exception as e:
                logger.warning(f"Could not evaluate {corruption} s={severity}: {e}")
            
            current_step += 1

    # 6. 결과 저장
    df = pd.DataFrame(results)
    save_path = f"./logs/robustness_results_{cfg.model.cfg_name}.csv"
    
    if len(results) > 0:
        df.to_csv(save_path, index=False)
        avg_rob_top1 = df['top1_acc'].mean()
        avg_rob_top5 = df['top5_acc'].mean()
        
        wandb.summary["avg_robustness_top1"] = avg_rob_top1
        wandb.summary["avg_robustness_top5"] = avg_rob_top5
        
        wandb.log({"robustness_results_table": wandb_table})
        
        logger.info(f"Eval Complete. Mean Top-1: {avg_rob_top1:.2f}% | Mean Top-5: {avg_rob_top5:.2f}%")
        logger.info(f"Results saved to {save_path}")
    else:
        logger.warning("❌ No evaluation results - all corruptions failed!")
        logger.warning("This may be due to CUDA compatibility issues or device errors.")
        df.to_csv(save_path, index=False)  # Save empty CSV
        wandb.summary["eval_status"] = "failed"
    
    wandb.finish()
