import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models.factory import create_model_factory
from utils.logger import setup_logger
from utils.metrics import calculate_accuracy

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        # Subset에서 (img, label)을 꺼냄
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# =========================================================
# ImageNet-9 original데이터 로더
# =========================================================
def get_train_val_test_loaders(data_root, batch_size):
    """
    Original 데이터를 8:1:1로 나누어 Train/Val/Test Loader를 반환합니다.
    """
    target_path = os.path.join(data_root, 'original', 'val')
    full_dataset = ImageFolder(root=target_path, transform=None)
    
    # 크기 계산 (8:1:1)
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, 
        [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Train용
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Val/Test용
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_ds = TransformedSubset(train_ds, train_transform)
    val_ds = TransformedSubset(val_ds, eval_transform)
    test_ds = TransformedSubset(test_ds, eval_transform) 
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# =========================================================
# 나머지 Corruption 데이터 로더 (Mixed 등)
# =========================================================
def get_eval_loader(data_root, mode, batch_size):
    target_path = os.path.join(data_root, mode, 'val')
        
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    dataset = ImageFolder(root=target_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def evaluate(model, loader, device):
    model.eval()
    top1_sum, top5_sum, total = 0.0, 0.0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            top1_sum += acc1.item() * images.size(0)
            top5_sum += acc5.item() * images.size(0)
            total += labels.size(0)
    return top1_sum/total, top5_sum/total

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, top1_sum, top5_sum, total = 0.0, 0.0, 0.0, 0
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

def main(args):
    with open(args.config) as f: config = yaml.safe_load(f)

    is_pretrained = config.get('pretrained', False)

    wandb.init(
        project="cv", 
        name=f"background_{config['model_name']}_{is_pretrained}", 
        config=config, 
        tags=["bias", "split_test"]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger("./logs", name="background_robustness")
    
    logger.info("📦 Preparing Data Splits...")
    train_loader, val_loader, test_loader = get_train_val_test_loaders(args.data_root, config['batch_size'])
    
    model = create_model_factory(config['model_name'], num_classes=9, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    logger.info("🔥 Starting Training...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        train_loss, tr_acc1, tr_acc5 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc1, val_acc5 = evaluate(model, val_loader, device)
        scheduler.step()
        
        wandb.log({
            "train/top1": tr_acc1, "val/top1": val_acc1, "epoch": epoch
        })
        logger.info(f"Epoch {epoch+1} | Train: {tr_acc1:.2f}% | Val: {val_acc1:.2f}%")
        
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            torch.save(model.state_dict(), "./best_bias_model.pth")
            
    logger.info("\n📊 Final Evaluation on Test Sets...")
    model.load_state_dict(torch.load("./best_bias_model.pth"))
    
    results = {}
    
    # [1] Original Test Set 평가 (Baseline)
    orig_acc1, orig_acc5 = evaluate(model, test_loader, device)
    results['original'] = orig_acc1
    print(f"\n👉 {'Original (Test Set)':<20} Top-1: {orig_acc1:.2f}% | Top-5: {orig_acc5:.2f}%")
    wandb.log({"test/original_top1": orig_acc1})

    # [2] Mixed Rand 평가 (Robustness)
    try:
        mixed_loader = get_eval_loader(args.data_root, 'mixed_rand', 64)
        mixed_acc1, mixed_acc5 = evaluate(model, mixed_loader, device)
        results['mixed_rand'] = mixed_acc1
        print(f"👉 {'Mixed_Rand':<20} Top-1: {mixed_acc1:.2f}% | Top-5: {mixed_acc5:.2f}%")
        wandb.log({"test/mixed_rand_top1": mixed_acc1})
    except Exception as e:
        print(f"⚠️ Mixed eval failed: {e}")

    # [3] Only FG 평가
    try:
        fg_loader = get_eval_loader(args.data_root, 'only_fg', 64)
        fg_acc1, fg_acc5 = evaluate(model, fg_loader, device)
        print(f"👉 {'Only_FG':<20} Top-1: {fg_acc1:.2f}% | Top-5: {fg_acc5:.2f}%")
        wandb.log({"test/only_fg_top1": fg_acc1})
    except: pass

    # [4] Gap 계산
    if 'original' in results and 'mixed_rand' in results:
        gap = results['original'] - results['mixed_rand']
        print("-" * 50)
        print(f"📉 Background Gap: {gap:.2f}%p")
        print("=" * 50)
        wandb.log({"background_gap": gap})
        
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/models/resnet50_in9.yaml')
    parser.add_argument('--data_root', type=str, default='./data/ImageNet9')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    main(args)