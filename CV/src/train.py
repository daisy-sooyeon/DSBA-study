"""
Hydra 기반 모델 학습 모듈
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import wandb 
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.factory import create_model_factory
from utils.data_loader import get_loaders
from utils.logger import setup_logger
from utils.common import train_one_epoch, evaluate


def main(cfg: DictConfig):
    """Hydra 설정을 사용한 모델 학습"""
    
    # 1. 설정 로드 및 로깅
    wandb.init(
        project="cv",
        name=f"{cfg.model.cfg_name}_{cfg.dataset.name}",
        config=OmegaConf.to_container(cfg),
        tags=[cfg.model.cfg_name, cfg.dataset.name, "train"]
    )

    config = wandb.config

    # logger 설정
    log_name = f"{cfg.dataset.name}_{cfg.model.cfg_name}"
    logger = setup_logger("./logs", name=log_name)
    logger.info(f"--- Experiment: {cfg.dataset.name} + {cfg.model.name} ---")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Setup
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    
    # 3. Data & Model
    data_config = {
        'data_root': './data',
        'dataset_name': cfg.dataset.name,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.data.num_workers,
        'image_size': cfg.model.image_size
    }
    train_loader, val_loader, test_loader = get_loaders(data_config)
    
    model = create_model_factory(
        cfg.model.name, 
        cfg.model.num_classes, 
        pretrained=cfg.model.pretrained
    )
    model = model.to(device)
    
    # 4. Train Prep
    criterion = nn.CrossEntropyLoss()
    
    opt_name = cfg.train.optimizer.lower()
    lr = float(cfg.train.learning_rate)
    wd = float(cfg.train.weight_decay)
    
    if opt_name == 'sgd':
        momentum = float(cfg.train.get('momentum', 0.9))
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        logger.info(f"Optimizer: SGD (lr={lr}, momentum={momentum})")
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        logger.info(f"Optimizer: AdamW (lr={lr})")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        logger.info(f"Optimizer: Defaulting to AdamW (lr={lr})")

    # 5. Loop
    best_acc = 0.0
    epochs = int(cfg.train.epochs)
    
    for epoch in range(epochs):
        train_loss, train_top1, train_top5 = train_one_epoch(model, train_loader, criterion, optimizer, device, desc="Train")
        val_loss, top1, top5 = evaluate(model, val_loader, criterion, device, desc="Val")        
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/top1": train_top1,
            "train/top5": train_top5,
            "val/loss": val_loss,
            "val/top1_acc": top1,
            "val/top5_acc": top5,
            "learning_rate": current_lr
        })
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Top1: {train_top1:.2f}% | Val Top1: {top1:.2f}% | Val Top5: {top5:.2f}%")
        
        if top1 > best_acc:
            best_acc = top1
            save_path = f"./logs/{log_name}_best.pth"
            torch.save(model.state_dict(), save_path)
            
            wandb.summary["best_top1_acc"] = best_acc  
            wandb.summary["best_epoch"] = epoch + 1  
            
            logger.info(f"   >>> Best model saved (Top1: {best_acc:.2f}%)")
    
    logger.info("--- Training Finished. Running Final Evaluation with Best Model ---")
    
    # 저장된 최고 성능 모델 로드
    best_model_path = f"./logs/{log_name}_best.pth"
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    
    final_loss, final_top1, final_top5 = evaluate(model, test_loader, criterion, device, desc="Test")
    
    logger.info(f"Final Result -> Loss: {final_loss:.4f} | Top-1: {final_top1:.2f}% | Top-5: {final_top5:.2f}%")
    
    wandb.log({
        "final/test_loss": final_loss,
        "final/top1_acc": final_top1,
        "final/top5_acc": final_top5
    })
    
    wandb.finish()
