import argparse
import yaml
import torch
import torch.nn as nn
import os
import pandas as pd
import wandb
from tqdm import tqdm

from models.factory import create_model_factory
from utils.data_loader import get_robustness_loader, load_config
from utils.metrics import calculate_accuracy
from utils.logger import setup_logger

CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 
    'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 
    'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 
    'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]

def evaluate(model, loader, device):
    model.eval()
    top1_sum = 0.0
    top5_sum = 0.0 
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            
            top1_sum += acc1.item() * images.size(0)
            top5_sum += acc5.item() * images.size(0) 
            total += images.size(0)
            
    return top1_sum / total, top5_sum / total

def main(config_path, weights_path, robustness_data_root):
    config = load_config(config_path)
    is_pretrained = config.get('pretrained', False)

    run = wandb.init(
        project="cv",  
        name=f"eval_robustness_{config['model_name']}_{is_pretrained}",
        config=config,
        tags=["evaluation", "robustness", config['model_name'], str(is_pretrained)]
    )

    device = torch.device(config.get('device', 'cuda'))
    logger = setup_logger("./logs", name=f"eval_robustness_{config['model_name']}")
    
    # Load Model
    logger.info(f"Loading model: {config['model_name']} from {weights_path}")
    model = create_model_factory(config['model_name'], config['num_classes'], pretrained=False)
    
    checkpoint = torch.load(weights_path, map_location=device)
    
    logger.info(f"Checkpoint keys sample: {list(checkpoint.keys())[:5]}") 
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=True)
        
    model.to(device)
    
    results = []
    
    # WandB Table 생성
    wandb_table = wandb.Table(columns=["model", "corruption", "severity", "top1_acc", "top5_acc"])

    logger.info("Starting Robustness Evaluation...")

    total_steps = len(CORRUPTIONS) * 5
    pbar = tqdm(total=total_steps, desc="Evaluating")

    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            try:
                loader = get_robustness_loader(
                    robustness_data_root, 
                    corruption, 
                    severity, 
                    batch_size=config['batch_size'],
                    image_size=config.get('image_size', None)
                )
                
                acc1, acc5 = evaluate(model, loader, device)
                
                result_row = {
                    'model': config['model_name'],
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
                
                wandb_table.add_data(config['model_name'], corruption, severity, acc1, acc5)
                
                logger.info(f"[{corruption} Lv.{severity}] Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}%")
            
            except Exception as e:
                logger.warning(f"Could not evaluate {corruption} s={severity}: {e}")
            
            pbar.update(1)
            
    pbar.close()

    df = pd.DataFrame(results)
    save_path = f"./logs/robustness_results_{config['model_name']}_{is_pretrained}.csv"
    df.to_csv(save_path, index=False)
    
    avg_rob_top1 = df['top1_acc'].mean()
    avg_rob_top5 = df['top5_acc'].mean()
    
    wandb.summary["avg_robustness_top1"] = avg_rob_top1
    wandb.summary["avg_robustness_top5"] = avg_rob_top5
    
    wandb.log({"robustness_results_table": wandb_table})
    
    logger.info(f"Eval Complete. Mean Top-1: {avg_rob_top1:.2f}% | Mean Top-5: {avg_rob_top5:.2f}%")
    logger.info(f"Results saved to {save_path}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to original train config (e.g., configs/models/resnet18.yaml)')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained .pth file')
    parser.add_argument('--data_root', type=str, required=True, help='Root path for corrupted dataset (e.g., ./data/CIFAR-10-C)')
    args = parser.parse_args()
    
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    main(args.config, args.weights, args.data_root)