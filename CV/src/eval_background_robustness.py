"""
Hydra 기반 배경 편향 강건성 평가 모듈 (ImageNet-9)
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
from utils.logger import setup_logger
from utils.common import train_one_epoch, evaluate_accuracy_only
from utils.imagenet_utils import get_train_val_test_loaders, get_eval_loader


def main(cfg: DictConfig):
    """Hydra 설정을 사용한 배경 편향 강건성 평가
    
    Args:
        cfg: Hydra 설정
    """
    
    is_pretrained = cfg.model.pretrained

    # 1. WandB 초기화
    wandb.init(
        project="cv", 
        name=f"background_{cfg.model.cfg_name}", 
        config=OmegaConf.to_container(cfg),
        tags=["bias", "background_robustness"]
    )
    
    # 2. Device 및 Logger
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger("./logs", name="background_robustness")
    logger.info(f"--- Background Robustness: {cfg.model.name} (pretrained={is_pretrained}) ---")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # 3. 데이터 로더
    logger.info("📦 Preparing Data Splits...")
    data_root = cfg.background.get('data_root', './data/ImageNet9')
    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        data_root, cfg.train.batch_size
    )
    
    # 4. 모델
    model = create_model_factory(
        cfg.model.name, 
        num_classes=cfg.model.num_classes, 
        pretrained=cfg.model.pretrained
    ).to(device)
    
    # 5. Optimizer & Criterion
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )
    
    # 6. 학습
    logger.info("🔥 Starting Training...")
    best_val_acc = 0.0
    
    for epoch in range(cfg.train.epochs):
        train_loss, tr_acc1, tr_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, desc="Train"
        )
        val_acc1, val_acc5 = evaluate_accuracy_only(model, val_loader, device, desc="Val")
        scheduler.step()
        
        wandb.log({
            "train/top1": tr_acc1, 
            "val/top1": val_acc1, 
            "epoch": epoch + 1
        })
        logger.info(f"Epoch {epoch+1}/{cfg.train.epochs} | Train: {tr_acc1:.2f}% | Val: {val_acc1:.2f}%")
        
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            torch.save(model.state_dict(), "./best_bias_model.pth")
            
    # 7. 최종 평가
    logger.info("\n📊 Final Evaluation on Test Sets...")
    model.load_state_dict(torch.load("./best_bias_model.pth"))
    
    results = {}
    
    # [1] Original Test Set 평가 (Baseline)
    orig_acc1, orig_acc5 = evaluate_accuracy_only(model, test_loader, device, desc="Test-Original")
    results['original'] = orig_acc1
    print(f"\n👉 {'Original (Test Set)':<20} Top-1: {orig_acc1:.2f}% | Top-5: {orig_acc5:.2f}%")
    wandb.log({"test/original_top1": orig_acc1})

    # [2] Mixed Rand 평가 (Robustness)
    try:
        mixed_loader = get_eval_loader(data_root, 'mixed_rand', 64)
        mixed_acc1, mixed_acc5 = evaluate_accuracy_only(model, mixed_loader, device, desc="Test-Mixed")
        results['mixed_rand'] = mixed_acc1
        print(f"👉 {'Mixed_Rand':<20} Top-1: {mixed_acc1:.2f}% | Top-5: {mixed_acc5:.2f}%")
        wandb.log({"test/mixed_rand_top1": mixed_acc1})
    except Exception as e:
        print(f"⚠️ Mixed eval failed: {e}")

    # [3] Only FG 평가
    try:
        fg_loader = get_eval_loader(data_root, 'only_fg', 64)
        fg_acc1, fg_acc5 = evaluate_accuracy_only(model, fg_loader, device, desc="Test-FG")
        print(f"👉 {'Only_FG':<20} Top-1: {fg_acc1:.2f}% | Top-5: {fg_acc5:.2f}%")
        wandb.log({"test/only_fg_top1": fg_acc1})
    except Exception:
        pass

    # [4] Gap 계산
    if 'original' in results and 'mixed_rand' in results:
        gap = results['original'] - results['mixed_rand']
        print("-" * 50)
        print(f"📉 Background Gap: {gap:.2f}%p")
        print("=" * 50)
        wandb.log({"background_gap": gap})
        
    wandb.finish()
