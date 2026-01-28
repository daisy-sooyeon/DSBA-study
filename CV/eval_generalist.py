import argparse
import yaml
import torch
import torch.nn as nn
import wandb
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

from models.factory import create_model_factory
from utils.data_loader import get_test_only_robustness_loader
from utils.logger import setup_logger
from utils.metrics import calculate_accuracy

# 19가지 Corruption 목록
CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 
    'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 
    'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 
    'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]

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
    # 1. 설정 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)

    img_size = config.get('image_size', 32)
    is_pretrained = config.get('pretrained', False)

    # 2. WandB 초기화 (평가 모드 태그 추가)
    wandb.init(
        project="cv",
        name=f"Eval_Generalist_{config['model_name']}_{is_pretrained}",
        config=config,
        tags=["generalist", "eval_only"]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger("./logs", name="eval_generalist")
    
    # 3. 모델 생성
    model = create_model_factory(config['model_name'], num_classes=10, pretrained=False)
    
    # 4. 가중치 로드 (필수)
    if args.weights and os.path.exists(args.weights):
        logger.info(f"📥 Loading weights from {args.weights}")
        ckpt = torch.load(args.weights, map_location=device)
        
        # 'state_dict' 키가 있으면 꺼내기
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
            
        # Shape Mismatch 방지 로직 (혹시 모를 에러 방지)
        model_state = model.state_dict()
        filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_state and v.shape == model_state[k].shape}
        
        model.load_state_dict(filtered_ckpt, strict=False)
        logger.info(f"✅ Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"❌ Weights file not found: {args.weights}")
        
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # 5. 평가 시작
    logger.info("\n📊 Starting Robustness Evaluation (CIFAR-10-C)...")
    
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
            # Test Only Loader 호출 (image_size 전달)
            loader = get_test_only_robustness_loader(args.data_root, corruption, sev, batch_size=256, image_size=img_size)
            
            _, acc1, acc5 = evaluate(model, loader, criterion, device)
            
            logger.info(f"   Lv.{sev}: Top-1 {acc1:.2f}% | Top-5 {acc5:.2f}%")
            
            # WandB 기록
            wandb.log({
                f"test_detail/{corruption}_s{sev}_top1": acc1,
                f"test_detail/{corruption}_s{sev}_top5": acc5,
            })
            wandb_table.add_data(corruption, sev, acc1, acc5)
            
            # 리스트에 추가 (CSV 저장용)
            results_list.append({
                "corruption": corruption,
                "severity": sev,
                "top1_acc": acc1,
                "top5_acc": acc5
            })
            
            total_top1_sum += acc1
            total_top5_sum += acc5
            count += 1
            
    # 최종 평균 계산
    final_avg_top1 = total_top1_sum / count
    final_avg_top5 = total_top5_sum / count

    logger.info(f"✅ Eval Complete. Mean Top-1: {final_avg_top1:.2f}% | Mean Top-5: {final_avg_top5:.2f}%")

    wandb.summary["avg_robustness_top1"] = final_avg_top1
    wandb.summary["avg_robustness_top5"] = final_avg_top5
    wandb.log({"robustness_results_table": wandb_table})

    # CSV 저장
    df = pd.DataFrame(results_list)
    csv_save_path = f"./logs/eval_results_{config['model_name']}_{is_pretrained}.csv"
    
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    df.to_csv(csv_save_path, index=False)
    logger.info(f"💾 Results saved to {csv_save_path}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--weights', type=str, required=True, help='Path to checkpoint pth')
    parser.add_argument('--data_root', type=str, default='./data/CIFAR-10-C')
    args = parser.parse_args()
    main(args)