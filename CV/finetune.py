import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

from models.factory import create_model_factory
from utils.data_loader import get_combined_cifar10c_loader, get_test_only_robustness_loader
from utils.logger import setup_logger
from utils.metrics import calculate_accuracy

CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 
    'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 
    'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 
    'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    top1_sum, top5_sum, total = 0, 0, 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
        running_loss += loss.item() * images.size(0)
        top1_sum += acc1.item() * images.size(0)
        top5_sum += acc5.item() * images.size(0)
        total += labels.size(0)
        
    return running_loss/total, top1_sum/total, top5_sum/total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    top1_sum, top5_sum, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            
            running_loss += loss.item() * images.size(0)
            top1_sum += acc1.item() * images.size(0)
            top5_sum += acc5.item() * images.size(0)
            total += labels.size(0)
            
    return running_loss/total, top1_sum/total, top5_sum/total

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    img_size = config.get('image_size', 32)
    is_pretrained = config.get('pretrained', False)

    wandb.init(
        project="cv",
        name=f"fintune_{config['model_name']}_{is_pretrained}",
        config=config,
        tags=["generalist", "strict_split"]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger("./logs", name="train_generalist_strict")
    
    train_loader, val_loader = get_combined_cifar10c_loader(
        args.data_root, CORRUPTIONS, batch_size=config['batch_size'], image_size=img_size
    )
    
    model = create_model_factory(config['model_name'], num_classes=10, pretrained=False)
    
    if args.weights:
        logger.info(f"Loading weights from {args.weights}")
        ckpt = torch.load(args.weights, map_location=device)
        if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
        
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
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc1 = 0.0
    save_path = f"./checkpoints/generalist_{config['model_name']}_{is_pretrained}_best.pth"
    
    logger.info("🚀 Starting Training...")
    for epoch in range(args.epochs):
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        wandb.log({
            "train/top1": train_acc1, "val/top1": val_acc1,
            "epoch": epoch
        })
        logger.info(f"Epoch {epoch+1} | Train: {train_acc1:.2f}% | Val: {val_acc1:.2f}%")
        
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"⭐️ Best Model Saved!")

    logger.info("\n📊 Starting Final Test Evaluation (Unseen Data)...")
    
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    wandb_table = wandb.Table(columns=["corruption", "severity", "top1_acc", "top5_acc"])
    
    total_top1_sum = 0.0
    total_top5_sum = 0.0
    count = 0
    
    results_list = []

    # 19개 Corruption 순회
    for corruption in CORRUPTIONS:
        print(f"\n👉 Evaluating {corruption}...")
        
        # 1~5 Severity 순회
        for sev in range(1, 6):
            loader = get_test_only_robustness_loader(args.data_root, corruption, sev, batch_size=256, image_size=img_size)
            
            _, acc1, acc5 = evaluate(model, loader, criterion, device)
            
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
            
    final_avg_top1 = total_top1_sum / count
    final_avg_top5 = total_top5_sum / count

    logger.info(f"Eval Complete. Mean Top-1: {final_avg_top1:.2f}% | Mean Top-5: {final_avg_top5:.2f}%")

    wandb.summary["avg_robustness_top1"] = final_avg_top1
    wandb.summary["avg_robustness_top5"] = final_avg_top5

    wandb.log({"finetune_results_table": wandb_table})

    df = pd.DataFrame(results_list)
    csv_save_path = f"./logs/finetune_results_{config['model_name']}_{is_pretrained}.csv"
    
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True) # logs 폴더 없으면 생성
    df.to_csv(csv_save_path, index=False)
    logger.info(f"💾 Results saved to {csv_save_path}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./data/CIFAR-10-C')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    main(args)