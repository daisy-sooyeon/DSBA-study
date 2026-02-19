import wandb 
from tqdm import tqdm
import os
import time
import logging
import math

import torch
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import omegaconf
from omegaconf import OmegaConf
import hydra

# 사용자 제공 모듈 임포트
from utils import set_logger, set_seed_all
from model import EncoderForClassification, masked_mean_pooling
from data import get_dataloader

# Optional imports for t-SNE
try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    _TSNE_AVAILABLE = True
except Exception:
    _TSNE_AVAILABLE = False

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def save_checkpoint(model, epoch, metrics, checkpoint_dir, accuracy=None, tag=""):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if accuracy is not None:
        fname = f'{tag}_epoch_{epoch}_acc_{accuracy:.4f}.pt'
    else:
        fname = f'{tag}_epoch_{epoch}.pt'
        
    checkpoint_path = os.path.join(checkpoint_dir, fname)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    return checkpoint_path

def train_iter_accum(model, batch, device, accumulation_steps):
    """
    Accumulation을 위한 단일 Forward/Backward
    """
    inputs = {key : value.to(device) for key, value in batch.items()}
    
    # Forward Pass
    outputs = model(**inputs)
    
    loss = outputs['loss']
    logits = outputs['logits']
    
    # Loss Scaling for Gradient Accumulation
    loss_normalized = loss / accumulation_steps
    loss_normalized.backward()
    
    # Accuracy 계산
    accuracy = calculate_accuracy(logits, inputs['label'])
    
    # 로깅을 위해 Scaled된 Loss 값을 반환 (나중에 더하면 원래 Loss 평균)
    return loss_normalized.item(), accuracy

def valid_iter(model, batch, device):
    inputs = {key : value.to(device) for key, value in batch.items()}
    
    outputs = model(**inputs)
    
    loss = outputs['loss']
    logits = outputs['logits']
    accuracy = calculate_accuracy(logits, inputs['label'])
    
    # t-SNE를 위해 last_hidden_state와 mask 반환 (필요시)
    return loss.item(), accuracy, outputs.get('last_hidden_state'), inputs['attention_mask'], inputs['label']

def run_training(configs: omegaconf.DictConfig, target_batch_size: int, experiment_group_name: str):
    """
    특정 Target Batch Size로 학습을 수행하는 함수
    """
    set_seed_all(configs.train.seed)
    
    # logger
    logger = set_logger(configs)
    
    device = torch.device(configs.train.device if torch.cuda.is_available() else 'cpu')
    
    model = EncoderForClassification(configs)
    model = model.to(device)
    
    train_dataloader = get_dataloader(configs, 'train')
    valid_dataloader = get_dataloader(configs, 'valid')
    test_dataloader = get_dataloader(configs, 'test')
    
    physical_batch_size = configs.train.batch_size # 16
    
    # calculate Gradient Accumulation Steps
    accumulation_steps = target_batch_size // physical_batch_size
    if accumulation_steps < 1:
        accumulation_steps = 1

    base_batch_size = 64 
    base_lr = configs.train.learning_rate 
    
    scaling_factor = math.sqrt(target_batch_size / base_batch_size)
    scaled_lr = base_lr * scaling_factor
    
    print(f"\n" + "="*60)
    print(f"🚀 Experiment Start: Target Batch Size {target_batch_size}")
    print(f"   - Physical Batch Size: {physical_batch_size}")
    print(f"   - Accumulation Steps: {accumulation_steps}")
    print(f"   - Scaled Learning Rate: {scaled_lr}")
    print(f"   - Device: {device}")
    print("="*60 + "\n")

    # Optimizer & WandB Init
    optimizer = torch.optim.Adam(model.parameters(), lr=scaled_lr)
    
    exp_dir = os.path.dirname(os.path.dirname(__file__))
    wandb_dir = os.path.join(exp_dir, 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)
    
    run_name = f"TargetBS_{target_batch_size}_{configs.model.name}"
    
    wandb.init(
        project=configs.logging.wandb_project,
        group=experiment_group_name,
        name=run_name,
        config={
            **OmegaConf.to_container(configs, resolve=True),
            "target_batch_size": target_batch_size,
            "accumulation_steps": accumulation_steps,
            "scaled_learning_rate": scaled_lr
        },
        dir=wandb_dir,
        reinit=True
    )
    
    model_name_clean = configs.model.name.lower().replace('-', '_').replace('.', '_')
    checkpoint_dir = os.path.join(exp_dir, configs.logging.checkpoint_dir, model_name_clean, run_name)

    # Training Loop
    global_step = 0
    best_valid_acc = 0.0
    best_checkpoint_path = None

    for epoch in range(configs.train.epochs):
        model.train()
        accum_loss = 0.0
        accum_acc = 0.0
        
        optimizer.zero_grad()
        
        train_iterator = tqdm(train_dataloader, desc=f"Ep {epoch+1}/{configs.train.epochs} [BS {target_batch_size}]")
        
        step_count = 0
        for i, batch in enumerate(train_iterator):
            # Forward & Backward (Accumulation)
            loss_val, acc_val = train_iter_accum(model, batch, device, accumulation_steps)
            
            accum_loss += loss_val
            accum_acc += acc_val
            step_count += 1
            
            # Step Condition: accumulation_steps에 도달했거나, 에폭의 마지막 배치일 때
            is_update_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_dataloader))
            
            if is_update_step:
                optimizer.step()
                optimizer.zero_grad()
                
                # --- Step 단위 로깅 ---
                log_loss = accum_loss 
                log_acc = accum_acc / step_count 
                lr = optimizer.param_groups[0]['lr']
                
                wandb.log({
                    "train_loss_step": log_loss,
                    "train_accuracy_step": log_acc,
                    "learning_rate": lr,
                    "global_step": global_step,
                    "epoch": epoch + 1
                })
                
                # Reset Accumulators
                accum_loss = 0.0
                accum_acc = 0.0
                step_count = 0
                global_step += 1
        
        # --- Validation Loop ---
        model.eval()
        valid_loss_sum = 0
        valid_acc_sum = 0
        num_valid_batches = 0
        
        # t-SNE용 데이터 수집
        tsne_features = []
        tsne_labels = []

        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Validating", leave=False):
                loss, accuracy, last_hidden, mask, labels = valid_iter(model, batch, device)
                valid_loss_sum += loss
                valid_acc_sum += accuracy
                num_valid_batches += 1
                
                if _TSNE_AVAILABLE and len(tsne_features) < 10:
                    pooled = masked_mean_pooling(last_hidden, mask)
                    tsne_features.append(pooled.cpu())
                    tsne_labels.append(labels.cpu())

        avg_valid_loss = valid_loss_sum / num_valid_batches
        avg_valid_acc = valid_acc_sum / num_valid_batches
        
        print(f" > Valid Loss: {avg_valid_loss:.4f}, Acc: {avg_valid_acc:.4f}")
        logger.info(f"Epoch {epoch+1} Valid Loss: {avg_valid_loss:.4f}, Acc: {avg_valid_acc:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "valid_loss": avg_valid_loss,
            "valid_accuracy": avg_valid_acc
        })
        
        # t-SNE Plotting
        if _TSNE_AVAILABLE and len(tsne_features) > 0:
            try:
                feats = torch.cat(tsne_features, dim=0).numpy()
                lbls = torch.cat(tsne_labels, dim=0).numpy()
                # sampling
                max_samples = 1000
                if feats.shape[0] > max_samples:
                    idx = np.random.choice(feats.shape[0], max_samples, replace=False)
                    feats = feats[idx]
                    lbls = lbls[idx]
                
                tsne = TSNE(n_components=2, random_state=42)
                emb = tsne.fit_transform(feats)
                
                fig, ax = plt.subplots(figsize=(6,6))
                scatter = ax.scatter(emb[:,0], emb[:,1], c=lbls, cmap='coolwarm', s=5, alpha=0.6)
                ax.set_title(f't-SNE Epoch {epoch+1}')
                ax.axis('off')
                wandb.log({f"tsne_epoch_{epoch+1}": wandb.Image(fig)})
                plt.close(fig)
            except Exception as e:
                print(f"t-SNE plotting failed: {e}")

        # Checkpoint
        if avg_valid_acc > best_valid_acc:
            best_valid_acc = avg_valid_acc
            best_checkpoint_path = save_checkpoint(model, epoch+1, {}, checkpoint_dir, accuracy=best_valid_acc, tag="best")
            wandb.log({"best_valid_acc": best_valid_acc})
            print(f"⭐ New Best Checkpoint Saved! Acc: {best_valid_acc:.4f}")

    # --- Test with Best Checkpoint ---
    if best_checkpoint_path:
        print(f"\nTesting with best checkpoint: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        test_acc_sum = 0
        num_test_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):
                _, accuracy, _, _, _ = valid_iter(model, batch, device)
                test_acc_sum += accuracy
                num_test_batches += 1
        
        avg_test_acc = test_acc_sum / num_test_batches
        print(f"Test Accuracy: {avg_test_acc:.4f}")
        wandb.log({"test_accuracy": avg_test_acc})

    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: omegaconf.DictConfig) -> None:
    target_batch_sizes = [64, 256, 1024]
    
    group_name = f"BS_Exp_{int(time.time())}"
    
    print(f"Starting Multi-Batch Experiment Group: {group_name}")
    print(f"Target Batch Sizes: {target_batch_sizes}")
    
    for target_bs in target_batch_sizes:
        try:
            run_training(configs, target_bs, group_name)
        except Exception as e:
            print(f"❌ Error during target_batch_size {target_bs}: {e}")
            import traceback
            traceback.print_exc()
            if wandb.run is not None:
                wandb.finish()
            continue

if __name__ == "__main__":
    main()