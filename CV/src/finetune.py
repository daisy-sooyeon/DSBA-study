"""
Hydra 기반 모델 미세조정 모듈
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.factory import create_model_factory
from utils.data_loader import get_combined_cifar10c_loader, get_test_only_robustness_loader
from utils.logger import setup_logger
from utils.common import train_one_epoch, evaluate
from utils.constants import CORRUPTIONS


def main(cfg: DictConfig, weights_path: str = None):
    """Hydra 설정을 사용한 모델 미세조정
    
    Args:
        cfg: Hydra 설정
        weights_path: 사전학습 가중치 경로
    """
    
    # 1. 설정 로드
    wandb.init(
        project="cv",
        name=f"finetune_{cfg.model.cfg_name}",
        config=OmegaConf.to_container(cfg),
        tags=["finetune", "robustness"]
    )
    
    img_size = cfg.model.image_size
    is_pretrained = cfg.model.pretrained
    
    # 2. Logger 설정
    log_name = f"finetune_{cfg.model.name}_{is_pretrained}"
    logger = setup_logger("./logs", name=log_name)
    logger.info(f"--- Fine-tuning: {cfg.model.name} (pretrained={is_pretrained}) ---")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # 3. Device 설정
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    
    # 4. Data Loader
    data_root = cfg.train.get('data_root', './data/CIFAR-10-C')
    train_loader, val_loader = get_combined_cifar10c_loader(
        data_root, CORRUPTIONS, batch_size=cfg.train.batch_size, image_size=img_size
    )
    
    # 5. Model
    model = create_model_factory(cfg.model.name, num_classes=cfg.model.num_classes, pretrained=False)
    
    # 6. 가중치 로드
    if weights_path:
        logger.info(f"Loading weights from {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        if 'state_dict' in ckpt: 
            ckpt = ckpt['state_dict']
        
        model_state = model.state_dict()
        filtered_ckpt = {}
        
        for k, v in ckpt.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_ckpt[k] = v
                else:
                    logger.warning(f"⚠️ Skipping layer due to shape mismatch: {k}")
        
        model.load_state_dict(filtered_ckpt, strict=False)
        logger.info("✅ Weights loaded successfully (excluding mismatched layers)")
    
    model.to(device)
    
    # 7. Optimizer & Criterion
    opt_name = cfg.train.optimizer.lower()
    lr = float(cfg.train.learning_rate)
    wd = float(cfg.train.weight_decay)
    
    if opt_name == 'sgd':
        momentum = float(cfg.train.get('momentum', 0.9))
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        logger.info(f"Optimizer: SGD (lr={lr}, momentum={momentum})")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        logger.info(f"Optimizer: AdamW (lr={lr})")
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    
    # 8. Training Loop
    best_acc1 = 0.0
    save_path = f"./logs/{log_name}_best.pth"
    
    logger.info("🚀 Starting Fine-tuning...")
    for epoch in range(cfg.train.epochs):
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, desc="Train"
        )
        val_loss, val_acc1, val_acc5 = evaluate(
            model, val_loader, criterion, device, desc="Val"
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/top1": train_acc1,
            "val/loss": val_loss,
            "val/top1": val_acc1,
            "learning_rate": current_lr
        })
        logger.info(f"Epoch {epoch+1}/{cfg.train.epochs} | Train: {train_acc1:.2f}% | Val: {val_acc1:.2f}%")
        
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"⭐️ Best Model Saved! (Top-1: {best_acc1:.2f}%)")

    # 9. Final Evaluation on Test Data
    logger.info("\n📊 Starting Final Robustness Evaluation...")
    
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    wandb_table = wandb.Table(columns=["corruption", "severity", "top1_acc", "top5_acc"])
    
    total_top1_sum = 0.0
    total_top5_sum = 0.0
    count = 0
    results_list = []

    # 모든 Corruption 순회
    for corruption in CORRUPTIONS:
        logger.info(f"\n👉 Evaluating {corruption}...")
        
        # 1~5 Severity 순회
        for sev in range(1, 6):
            try:
                loader = get_test_only_robustness_loader(
                    data_root, corruption, sev, batch_size=256, image_size=img_size
                )
                
                _, acc1, acc5 = evaluate(model, loader, criterion, device, desc=f"{corruption}_s{sev}")
                
                logger.info(f"   Lv.{sev}: Top-1 {acc1:.2f}% | Top-5 {acc5:.2f}%")
                
                wandb.log({
                    f"test_detail/{corruption}_s{sev}_top1": acc1,
                    f"test_detail/{corruption}_s{sev}_top5": acc5,
                })
                wandb_table.add_data(corruption, sev, acc1, acc5)
                
                results_list.append({
                    "corruption": corruption,
                    "severity": sev,
                    "top1_acc": acc1,
                    "top5_acc": acc5
                })
                
                total_top1_sum += acc1
                total_top5_sum += acc5
                count += 1
            except Exception as e:
                logger.warning(f"Could not evaluate {corruption} s={sev}: {e}")
            
    final_avg_top1 = total_top1_sum / count if count > 0 else 0.0
    final_avg_top5 = total_top5_sum / count if count > 0 else 0.0

    logger.info(f"Eval Complete. Mean Top-1: {final_avg_top1:.2f}% | Mean Top-5: {final_avg_top5:.2f}%")

    wandb.summary["avg_robustness_top1"] = final_avg_top1
    wandb.summary["avg_robustness_top5"] = final_avg_top5
    wandb.summary["best_finetune_top1"] = best_acc1

    wandb.log({"finetune_results_table": wandb_table})

    # 10. Save Results
    df = pd.DataFrame(results_list)
    csv_save_path = f"./logs/finetune_results_{cfg.model.name}_{is_pretrained}.csv"
    
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    df.to_csv(csv_save_path, index=False)
    logger.info(f"💾 Results saved to {csv_save_path}")

    wandb.finish()
