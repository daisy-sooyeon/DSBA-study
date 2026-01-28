import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb 

from tqdm import tqdm
from models.factory import create_model_factory
from utils.data_loader import get_loaders, load_config, merge_configs
from utils.metrics import calculate_accuracy
from utils.logger import setup_logger

# ---------------------------------------------------------
# 학습/검증 함수
# ---------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="   Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    top1_avg = 0.0
    top5_avg = 0.0 
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="   Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            
            running_loss += loss.item() * images.size(0)
            top1_avg += acc1.item() * images.size(0)
            top5_avg += acc5.item() * images.size(0) 
            total += images.size(0)
            
    return running_loss / total, top1_avg / total, top5_avg / total

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main(args):
    # 1. 두 개의 Config 파일 로드 및 병합
    data_conf = load_config(f"./configs/data/{args.data}.yaml")
    model_conf = load_config(f"./configs/models/{args.model}.yaml")
    
    # 병합: 초기 config 딕셔너리 생성
    raw_config = merge_configs(data_conf, model_conf)
    
    wandb.init(
        project="cv",  # 프로젝트 이름
        name=f"{args.model}_{args.data}",
        config=raw_config,
        tags=[args.model, args.data, "train"]
    )
    
    config = wandb.config

    # logger 설정
    log_name = f"{config.name}_{config.model_name}_{config.pretrained}"
    logger = setup_logger("./logs", name=log_name)
    logger.info(f"--- Experiment: {args.data} + {args.model} ---")
    logger.info(f"WandB Config: {config}")

    # 2. Setup
    device = torch.device(config.get('device', 'cuda'))
    
    # 3. Data & Model
    train_loader, val_loader, test_loader = get_loaders(dict(config))
    
    model = create_model_factory(
        config.model_name, 
        config.num_classes, 
        pretrained=config.get('pretrained', False)
    )
    model = model.to(device)
    
    # 4. Train Prep
    criterion = nn.CrossEntropyLoss()
    
    opt_name = config.get('optimizer', 'adamw').lower()
    lr = float(config.learning_rate)
    wd = float(config.weight_decay)
    
    if opt_name == 'sgd':
        momentum = float(config.get('momentum', 0.9))
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
    epochs = int(config.epochs)
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, top1, top5 = validate(model, val_loader, criterion, device)        
        # 현재 LR 가져오기 (Scheduler 쓸 때 유용)
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/top1_acc": top1,
            "val/top5_acc": top5,
            "learning_rate": current_lr
        })
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Top1: {top1:.2f}% | Val Top5: {top5:.2f}%")
        
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
    
    final_loss, final_top1, final_top5 = validate(model, test_loader, criterion, device)
    
    logger.info(f"Final Result -> Loss: {final_loss:.4f} | Top-1: {final_top1:.2f}% | Top-5: {final_top5:.2f}%")
    
    wandb.log({
        "final/test_loss": final_loss,
        "final/top1_acc": final_top1,
        "final/top5_acc": final_top5
    })
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 파일명 입력 받기
    parser.add_argument('--data', type=str, required=True, help='Name of the data config file (e.g., cifar10)')
    parser.add_argument('--model', type=str, required=True, help='Name of the model config file (e.g., resnet18)')
    
    args = parser.parse_args()
    
    # logs 폴더 생성
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
        
    main(args)