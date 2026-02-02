import wandb 
from tqdm import tqdm
import os
import logging
import time

import torch
# Suppress TorchDynamo compilation errors
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import torch.nn
import omegaconf
from omegaconf import OmegaConf
import hydra

from utils import load_config, set_logger, set_seed_all
from model import EncoderForClassification, masked_mean_pooling
from data import get_dataloader

# Optional imports for t-SNE plotting
try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    _TSNE_AVAILABLE = True
except Exception:
    _TSNE_AVAILABLE = False

def train_iter(model, inputs, optimizer, device, epoch):
    """Single training iteration"""
    inputs = {key : value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    loss = outputs['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def valid_iter(model, inputs, device):
    """Single validation iteration"""
    inputs = {key : value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    loss = outputs['loss']
    accuracy = calculate_accuracy(outputs['logits'], inputs['label'])    
    return loss, accuracy

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def save_checkpoint(model, epoch, metrics, checkpoint_dir, accuracy=None):
    """Save model checkpoint with accuracy info in filename"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Include accuracy in filename for easy identification
    if accuracy is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}_acc_{accuracy:.4f}.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    return checkpoint_path

def main(configs : omegaconf.DictConfig) :
    """Main training function"""
    # Set seed for reproducibility
    set_seed_all(configs.train.seed)
    
    # Set device
    device = torch.device(configs.train.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set logger
    logger = set_logger(configs)
    
    # Load model
    model = EncoderForClassification(configs)
    model = model.to(device)
    print(f"Model loaded: {configs.model.name}")
    
    # Load data
    train_dataloader = get_dataloader(configs, 'train')
    valid_dataloader = get_dataloader(configs, 'valid')
    test_dataloader = get_dataloader(configs, 'test')
    print(f"Data loaded - Train: {len(train_dataloader)}, Valid: {len(valid_dataloader)}, Test: {len(test_dataloader)}")
    
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.train.learning_rate)
    
    # WandB initialization
    # Set wandb directory to exp_1/wandb (relative to exp_1 directory)
    exp_1_dir = os.path.dirname(os.path.dirname(__file__))
    wandb_dir = os.path.join(exp_1_dir, 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ['WANDB_DIR'] = wandb_dir
    
    wandb.init(
        project=configs.logging.wandb_project,
        name=f"[imdb] {configs.model.name}",
        config=OmegaConf.to_container(configs, resolve=True),
        dir=wandb_dir
    )
    
    # Create checkpoint directory with model-specific subdirectory
    # Set checkpoint directory to exp_1/checkpoints (relative to exp_1 directory)
    exp_1_dir = os.path.dirname(os.path.dirname(__file__))
    model_name = configs.model.name.lower().replace('-', '_').replace('.', '_')
    checkpoint_dir = os.path.join(exp_1_dir, configs.logging.checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train & validation for each epoch
    best_valid_acc = 0
    best_checkpoint_path = None
    global_step = 0
    
    for epoch in range(configs.train.epochs):
        print(f"\n===== Epoch {epoch+1}/{configs.train.epochs} =====")
        logger.info(f"===== Epoch {epoch+1}/{configs.train.epochs} =====")
        # record epoch start time for timing metrics
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}"):
            loss = train_iter(model, batch, optimizer, device, epoch)
            train_loss += loss.item()
            num_batches += 1
            # Log step-wise loss and learning rate
            lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else None
            try:
                wandb.log({'train_loss_step': loss.item(), 'learning_rate': lr}, step=global_step)
            except Exception:
                wandb.log({'train_loss_step': loss.item(), 'learning_rate': lr})
            global_step += 1
        
        avg_train_loss = train_loss / num_batches
        print(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        wandb.log({'epoch': epoch+1, 'train_loss_epoch': avg_train_loss})
        
        # Validation
        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Valid Epoch {epoch+1}"):
                loss, accuracy = valid_iter(model, batch, device)
                valid_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                valid_accuracy += accuracy
                num_batches += 1
                # collect features for TSNE (use encoder directly)
                try:
                    model_input = {k: v.to(device) for k, v in batch.items() if k != 'label'}
                    enc_out = model.encoder(**model_input)
                    pooled = masked_mean_pooling(enc_out['last_hidden_state'], model_input['attention_mask'])
                    if 'tsne_features' not in locals():
                        tsne_features = []
                        tsne_labels = []
                    tsne_features.append(pooled.detach().cpu())
                    tsne_labels.append(batch['label'])
                except Exception:
                    pass
        
        avg_valid_loss = valid_loss / num_batches
        avg_valid_accuracy = valid_accuracy / num_batches
        
        print(f"Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}")
        logger.info(f"Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}")
        
        # Log to WandB
        wandb.log({
            'epoch': epoch+1,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy
        })
        
        # Save checkpoint for every epoch
        checkpoint_path = save_checkpoint(
            model, 
            epoch+1, 
            {'valid_loss': avg_valid_loss, 'valid_accuracy': avg_valid_accuracy},
            checkpoint_dir,
            accuracy=avg_valid_accuracy
        )
        print(f"Checkpoint saved: {checkpoint_path}")
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Track best checkpoint for testing
        if avg_valid_accuracy > best_valid_acc:
            best_valid_acc = avg_valid_accuracy
            best_checkpoint_path = checkpoint_path
            print(f"⭐ New best checkpoint! (accuracy: {best_valid_acc:.4f})")
            logger.info(f"⭐ New best checkpoint! (accuracy: {best_valid_acc:.4f})")
            # Log best eval accuracy explicitly
            wandb.log({'best_valid_accuracy': best_valid_acc}, step=global_step)

        # Log epoch time
        try:
            epoch_time = time.time() - epoch_start
            wandb.log({'epoch_time': epoch_time, 'epoch': epoch+1})
        except Exception:
            pass

        # t-SNE logging (plot) for validation pooled features
        if _TSNE_AVAILABLE and 'tsne_features' in locals() and len(tsne_features) > 0:
            try:
                feats = torch.cat(tsne_features, dim=0).numpy()
                labels = torch.cat(tsne_labels, dim=0).numpy()
                # limit samples to 1000 for speed
                max_samples = 1000
                if feats.shape[0] > max_samples:
                    inds = np.random.choice(feats.shape[0], max_samples, replace=False)
                    feats = feats[inds]
                    labels = labels[inds]
                tsne = TSNE(n_components=2, random_state=42)
                emb = tsne.fit_transform(feats)
                fig, ax = plt.subplots(figsize=(6,6))
                scatter = ax.scatter(emb[:,0], emb[:,1], c=labels, cmap='coolwarm', s=5)
                ax.set_title(f't-SNE (valid) epoch {epoch+1}')
                ax.axis('off')
                wandb.log({f'tsne_valid_epoch_{epoch+1}': wandb.Image(fig)}, step=global_step)
                plt.close(fig)
            except Exception:
                pass
    
    # Test with best checkpoint
    print("\n===== Testing with best checkpoint =====")
    logger.info("===== Testing with best checkpoint =====")
    
    # Load best checkpoint
    if best_checkpoint_path:
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
    
    # Test
    model.eval()
    test_loss = 0
    test_accuracy = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            loss, accuracy = valid_iter(model, batch, device)
            test_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            test_accuracy += accuracy
            num_batches += 1
    
    avg_test_loss = test_loss / num_batches
    avg_test_accuracy = test_accuracy / num_batches
    
    print(f"\nTest Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
    logger.info(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
    
    # Log test results to WandB
    wandb.log({
        'test_loss': avg_test_loss,
        'test_accuracy': avg_test_accuracy
    })
    
    wandb.finish()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def hydra_main(cfg: omegaconf.DictConfig) -> None:
    main(cfg)

if __name__ == "__main__":
    hydra_main()