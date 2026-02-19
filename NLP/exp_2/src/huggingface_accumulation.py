import wandb 
from tqdm import tqdm
import os
import time
import logging

import torch
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import omegaconf
from omegaconf import OmegaConf
import hydra

# [추가] Accelerate 임포트
from accelerate import Accelerator

from utils import set_logger, set_seed_all
from model import EncoderForClassification, masked_mean_pooling
from data import get_dataloader

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

# [수정] 모델 가중치 저장 시 accelerator.unwrap_model() 사용을 위해 함수 파라미터 수정
def save_checkpoint(model, epoch, metrics, checkpoint_dir, accelerator, accuracy=None, tag=""):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if accuracy is not None:
        fname = f'{tag}_epoch_{epoch}_acc_{accuracy:.4f}.pt'
    else:
        fname = f'{tag}_epoch_{epoch}.pt'
        
    checkpoint_path = os.path.join(checkpoint_dir, fname)
    
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save({
        'epoch': epoch,
        'model_state_dict': unwrapped_model.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    return checkpoint_path

# [삭제] 수동으로 나누던 train_iter_accum 삭제 (메인 루프에 통합됨)

def valid_iter(model, batch):
    outputs = model(**batch)
    
    loss = outputs['loss']
    logits = outputs['logits']
    accuracy = calculate_accuracy(logits, batch['label'])
    
    return loss.item(), accuracy, outputs.get('last_hidden_state'), batch['attention_mask'], batch['label']

def run_training(configs: omegaconf.DictConfig, target_batch_size: int, experiment_group_name: str):
    set_seed_all(configs.train.seed)
    logger = set_logger(configs)
    
    physical_batch_size = configs.train.batch_size # 16
    accumulation_steps = target_batch_size // physical_batch_size
    if accumulation_steps < 1:
        accumulation_steps = 1
        
    # [추가] Accelerator 초기화
    accelerator = Accelerator(gradient_accumulation_steps=accumulation_steps)
    
    print(f"\n" + "="*60)
    print(f"🚀 Experiment Start: Target Batch Size {target_batch_size}")
    print(f"   - Physical Batch Size: {physical_batch_size}")
    print(f"   - Accumulation Steps: {accumulation_steps}")
    print(f"   - Device: {accelerator.device}")
    print("="*60 + "\n")

    model = EncoderForClassification(configs)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.train.learning_rate)
    
    train_dataloader = get_dataloader(configs, 'train')
    valid_dataloader = get_dataloader(configs, 'valid')
    test_dataloader = get_dataloader(configs, 'test')
    
    # Accelerate Prepare (모든 객체를 Accelerator가 관리하도록 넘김)
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    
    # WandB 초기화
    exp_dir = os.path.dirname(os.path.dirname(__file__))
    wandb_dir = os.path.join(exp_dir, 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)
    run_name = f"TargetBS_{target_batch_size}_{configs.model.name}"
    
    if accelerator.is_main_process:
        wandb.init(
            project=configs.logging.wandb_project,
            group=experiment_group_name,
            name=run_name,
            config={
                **OmegaConf.to_container(configs, resolve=True),
                "target_batch_size": target_batch_size,
                "accumulation_steps": accumulation_steps
            },
            dir=wandb_dir,
            reinit=True
        )
    
    model_name_clean = configs.model.name.lower().replace('-', '_').replace('.', '_')
    checkpoint_dir = os.path.join(exp_dir, configs.logging.checkpoint_dir, 'accelerator', model_name_clean, run_name)

    # Training Loop 
    global_step = 0
    best_valid_acc = 0.0
    best_checkpoint_path = None

    for epoch in range(configs.train.epochs):
        model.train()
        accum_loss = 0.0
        accum_acc = 0.0
        step_count = 0
        
        train_iterator = tqdm(train_dataloader, desc=f"Ep {epoch+1}/{configs.train.epochs} [BS {target_batch_size}]", disable=not accelerator.is_local_main_process)
        
        for batch in train_iterator:
            # Accelerate의 Accumulation Context Manager 사용
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs['loss']
                
                accelerator.backward(loss)
                
                optimizer.step()
                optimizer.zero_grad()
                
            accum_loss += loss.item() 
            accum_acc += calculate_accuracy(outputs['logits'], batch['label'])
            step_count += 1
            
            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    log_loss = accum_loss / step_count
                    log_acc = accum_acc / step_count
                    lr = optimizer.param_groups[0]['lr']
                    
                    wandb.log({
                        "train_loss_step": log_loss,
                        "train_accuracy_step": log_acc,
                        "learning_rate": lr,
                        "global_step": global_step,
                        "epoch": epoch + 1
                    })
                    
                # 초기화
                accum_loss = 0.0
                accum_acc = 0.0
                step_count = 0
                global_step += 1
        
        # --- Validation Loop ---
        model.eval()
        valid_loss_sum = 0
        valid_acc_sum = 0
        num_valid_batches = 0
        
        tsne_features = []
        tsne_labels = []

        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Validating", leave=False, disable=not accelerator.is_local_main_process):
                loss, accuracy, last_hidden, mask, labels = valid_iter(model, batch)
                valid_loss_sum += loss
                valid_acc_sum += accuracy
                num_valid_batches += 1
                
                if _TSNE_AVAILABLE and len(tsne_features) < 10:
                    pooled = masked_mean_pooling(last_hidden, mask)
                    tsne_features.append(pooled.cpu())
                    tsne_labels.append(labels.cpu())

        avg_valid_loss = valid_loss_sum / num_valid_batches
        avg_valid_acc = valid_acc_sum / num_valid_batches
        
        if accelerator.is_main_process:
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
                    max_samples = 1000
                    if feats.shape[0] > max_samples:
                        idx = np.random.choice(feats.shape[0], max_samples, replace=False)
                        feats = feats[idx]
                        lbls = lbls[idx]
                    
                    tsne = TSNE(n_components=2, random_state=42)
                    emb = tsne.fit_transform(feats)
                    
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.scatter(emb[:,0], emb[:,1], c=lbls, cmap='coolwarm', s=5, alpha=0.6)
                    ax.set_title(f't-SNE Epoch {epoch+1}')
                    ax.axis('off')
                    wandb.log({f"tsne_epoch_{epoch+1}": wandb.Image(fig)})
                    plt.close(fig)
                except Exception as e:
                    print(f"t-SNE plotting failed: {e}")

            # Checkpoint (accelerator 추가)
            if avg_valid_acc > best_valid_acc:
                best_valid_acc = avg_valid_acc
                best_checkpoint_path = save_checkpoint(model, epoch+1, {}, checkpoint_dir, accelerator, accuracy=best_valid_acc, tag="best")
                wandb.log({"best_valid_acc": best_valid_acc})
                print(f"⭐ New Best Checkpoint Saved! Acc: {best_valid_acc:.4f}")

    # --- Test with Best Checkpoint ---
    if best_checkpoint_path and accelerator.is_main_process:
        print(f"\nTesting with best checkpoint: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path)
        
        accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        test_acc_sum = 0
        num_test_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):
                _, accuracy, _, _, _ = valid_iter(model, batch)
                test_acc_sum += accuracy
                num_test_batches += 1
        
        avg_test_acc = test_acc_sum / num_test_batches
        print(f"Test Accuracy: {avg_test_acc:.4f}")
        wandb.log({"test_accuracy": avg_test_acc})

    if accelerator.is_main_process:
        wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: omegaconf.DictConfig) -> None:
    target_batch_sizes = [64, 256, 1024]
    group_name = f"BS_Exp_{int(time.time())}"
    
    print(f"Starting Multi-Batch Experiment Group: {group_name}")
    
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