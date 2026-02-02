"""
Error Analysis Script for BERT vs ModernBERT Comparison

This script performs:
1. t-SNE boundary error analysis - finds misclassified samples near cluster boundaries
2. Attention map comparison - visualizes attention patterns for both models
"""

import os
import sys
import torch
# Suppress TorchDynamo compilation errors
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import omegaconf
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import EncoderForClassification, masked_mean_pooling
from data import get_dataloader, IMDBDataset


def load_model_and_data(model_name, checkpoint_path, config_path):
    """Load trained model and validation data"""
    # Load base config
    cfg = OmegaConf.load(config_path)
    
    # Load model-specific config
    config_dir = os.path.dirname(config_path)
    if 'bert' in model_name.lower() and 'modern' not in model_name.lower():
        model_config_path = os.path.join(config_dir, 'model', 'bert.yaml')
    else:
        model_config_path = os.path.join(config_dir, 'model', 'modernbert.yaml')
    
    if os.path.exists(model_config_path):
        model_cfg = OmegaConf.load(model_config_path)
        # Merge model config into base config
        cfg = OmegaConf.merge(cfg, model_cfg)
    
    # Override model name
    if 'model' not in cfg:
        cfg.model = OmegaConf.create({})
    cfg.model.name = model_name
    
    # Set device
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = EncoderForClassification(cfg)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    model = model.to(device)
    model.eval()
    
    # Load validation data
    valid_dataloader = get_dataloader(cfg, 'valid')
    dataset = IMDBDataset(cfg, 'valid')
    
    return model, valid_dataloader, dataset, device, cfg


def extract_features_and_predictions(model, dataloader, device):
    """Extract features, predictions, and ground truth labels"""
    features = []
    predictions = []
    labels = []
    indices = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            label = batch['label'].to(device)
            
            # Get encoder output
            encoder_output = model.encoder(**inputs)
            pooled = masked_mean_pooling(encoder_output['last_hidden_state'], inputs['attention_mask'])
            
            # Get predictions
            logits = model.classifier(pooled)
            preds = logits.argmax(dim=-1)
            
            features.append(pooled.cpu())
            predictions.append(preds.cpu())
            labels.append(label.cpu())
            indices.extend(range(idx * dataloader.batch_size, 
                                idx * dataloader.batch_size + len(label)))
    
    features = torch.cat(features, dim=0).numpy()
    predictions = torch.cat(predictions, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    return features, predictions, labels, indices


def find_boundary_samples(emb_2d, labels, predictions, n_samples=10):
    """
    Find samples near cluster boundaries (misclassified or near decision boundary)
    
    Strategy:
    1. Find samples that are misclassified (pred != label)
    2. Among misclassified, find those near the opposite cluster center
    3. Also find correctly classified samples near boundaries
    """
    # Calculate cluster centers
    pos_center = emb_2d[labels == 1].mean(axis=0)
    neg_center = emb_2d[labels == 0].mean(axis=0)
    
    # Find misclassified samples
    misclassified = predictions != labels
    
    boundary_samples = []
    
    # Case 1: Positive label but predicted as negative (in wrong cluster)
    pos_wrong = (labels == 1) & (predictions == 0)
    if pos_wrong.sum() > 0:
        pos_wrong_emb = emb_2d[pos_wrong]
        # Distance to negative cluster center
        dists_to_neg = np.linalg.norm(pos_wrong_emb - neg_center, axis=1)
        # Get closest ones (most confused)
        closest_indices = np.argsort(dists_to_neg)[:min(n_samples//2, len(dists_to_neg))]
        pos_wrong_indices = np.where(pos_wrong)[0][closest_indices]
        for idx in pos_wrong_indices:
            boundary_samples.append({
                'index': idx,
                'label': labels[idx],
                'prediction': predictions[idx],
                'type': 'positive_misclassified_as_negative',
                'distance_to_opposite_center': dists_to_neg[closest_indices[list(pos_wrong_indices).index(idx)]]
            })
    
    # Case 2: Negative label but predicted as positive (in wrong cluster)
    neg_wrong = (labels == 0) & (predictions == 1)
    if neg_wrong.sum() > 0:
        neg_wrong_emb = emb_2d[neg_wrong]
        # Distance to positive cluster center
        dists_to_pos = np.linalg.norm(neg_wrong_emb - pos_center, axis=1)
        closest_indices = np.argsort(dists_to_pos)[:min(n_samples//2, len(dists_to_pos))]
        neg_wrong_indices = np.where(neg_wrong)[0][closest_indices]
        for idx in neg_wrong_indices:
            boundary_samples.append({
                'index': idx,
                'label': labels[idx],
                'prediction': predictions[idx],
                'type': 'negative_misclassified_as_positive',
                'distance_to_opposite_center': dists_to_pos[closest_indices[list(neg_wrong_indices).index(idx)]]
            })
    
    # Case 3: Correctly classified but near boundary (close to opposite cluster)
    correctly_classified = predictions == labels
    if correctly_classified.sum() > 0:
        correct_emb = emb_2d[correctly_classified]
        correct_labels = labels[correctly_classified]
        
        # For positive samples, find those close to negative center
        pos_correct = correct_labels == 1
        if pos_correct.sum() > 0:
            pos_correct_emb = correct_emb[pos_correct]
            dists_to_neg = np.linalg.norm(pos_correct_emb - neg_center, axis=1)
            closest_indices = np.argsort(dists_to_neg)[:min(3, len(dists_to_neg))]
            pos_correct_indices = np.where(correctly_classified & (labels == 1))[0][closest_indices]
            for idx in pos_correct_indices:
                boundary_samples.append({
                    'index': idx,
                    'label': labels[idx],
                    'prediction': predictions[idx],
                    'type': 'positive_near_boundary',
                    'distance_to_opposite_center': dists_to_neg[closest_indices[list(pos_correct_indices).index(idx)]]
                })
        
        # For negative samples, find those close to positive center
        neg_correct = correct_labels == 0
        if neg_correct.sum() > 0:
            neg_correct_emb = correct_emb[neg_correct]
            dists_to_pos = np.linalg.norm(neg_correct_emb - pos_center, axis=1)
            closest_indices = np.argsort(dists_to_pos)[:min(3, len(dists_to_pos))]
            neg_correct_indices = np.where(correctly_classified & (labels == 0))[0][closest_indices]
            for idx in neg_correct_indices:
                boundary_samples.append({
                    'index': idx,
                    'label': labels[idx],
                    'prediction': predictions[idx],
                    'type': 'negative_near_boundary',
                    'distance_to_opposite_center': dists_to_pos[closest_indices[list(neg_correct_indices).index(idx)]]
                })
    
    return boundary_samples


def analyze_tsne_boundary_errors(model_name, checkpoint_path, config_path, n_samples=15, save_dir=None):
    """
    Analyze t-SNE boundary errors and print original texts
    """
    print(f"\n{'='*80}")
    print(f"t-SNE Boundary Error Analysis: {model_name}")
    print(f"{'='*80}\n")
    
    # Create save directory
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analysis_outputs')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model and data
    model, valid_dataloader, dataset, device, cfg = load_model_and_data(
        model_name, checkpoint_path, config_path
    )
    
    # Extract features and predictions
    features, predictions, labels, indices = extract_features_and_predictions(
        model, valid_dataloader, device
    )
    
    # Limit to 1000 samples for t-SNE (same as training script)
    max_samples = 1000
    if len(features) > max_samples:
        sample_indices = np.random.choice(len(features), max_samples, replace=False)
        features_subset = features[sample_indices]
        labels_subset = labels[sample_indices]
        predictions_subset = predictions[sample_indices]
        original_indices = sample_indices
    else:
        features_subset = features
        labels_subset = labels
        predictions_subset = predictions
        original_indices = np.arange(len(features))
    
    # Compute t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(features_subset)
    
    # Find boundary samples
    boundary_samples = find_boundary_samples(
        emb_2d, labels_subset, predictions_subset, n_samples=n_samples
    )
    
    # Prepare output text
    model_short_name = model_name.split('/')[-1].replace('-', '_')
    output_lines = []
    output_lines.append(f"{'='*80}")
    output_lines.append(f"t-SNE Boundary Error Analysis: {model_name}")
    output_lines.append(f"{'='*80}\n")
    output_lines.append(f"Total samples analyzed: {len(features_subset)}")
    output_lines.append(f"Total misclassified: {(predictions_subset != labels_subset).sum()}/{len(labels_subset)}")
    output_lines.append(f"Boundary samples found: {len(boundary_samples)}\n")
    output_lines.append(f"{'='*80}\n")
    
    # Print and save results
    print(f"\n{'='*80}")
    print(f"Found {len(boundary_samples)} boundary samples")
    print(f"{'='*80}\n")
    
    for i, sample in enumerate(boundary_samples[:n_samples], 1):
        original_idx = original_indices[sample['index']]
        text = dataset.valid_data[original_idx]['text']
        label_name = "Positive" if sample['label'] == 1 else "Negative"
        pred_name = "Positive" if sample['prediction'] == 1 else "Negative"
        
        sample_text = f"\n[Sample {i}]\n"
        sample_text += f"Type: {sample['type']}\n"
        sample_text += f"Label: {label_name} | Prediction: {pred_name}\n"
        sample_text += f"Distance to opposite center: {sample['distance_to_opposite_center']:.4f}\n"
        sample_text += f"Text:\n"
        sample_text += "-" * 80 + "\n"
        sample_text += text + "\n"
        sample_text += "-" * 80 + "\n"
        
        print(sample_text)
        output_lines.append(sample_text)
    
    # Save to file
    log_file = os.path.join(save_dir, f'tsne_boundary_errors_{model_short_name}.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\n✅ Results saved to: {log_file}")
    
    return boundary_samples, emb_2d, labels_subset, predictions_subset, dataset, original_indices


def extract_attention_weights(model, tokenizer, text, device, max_length=128):
    """
    Extract attention weights from model for a given text
    Returns attention weights from all layers
    """
    # Tokenize
    inputs = tokenizer(text, truncation=True, padding='max_length', 
                      max_length=max_length, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get attention weights
    model.eval()
    with torch.no_grad():
        # Forward pass with output_attentions=True
        try:
            outputs = model.encoder(**inputs, output_attentions=True)
            attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
        except Exception as e:
            print(f"Warning: Could not extract attention weights: {e}")
            attentions = None
    
    # Check if attentions are available
    if attentions is None or len(attentions) == 0:
        # Fallback: create dummy attention (uniform attention)
        seq_len = inputs['input_ids'].shape[1]
        dummy_attention = np.ones((seq_len, seq_len)) / seq_len
        layer_attentions = [dummy_attention]
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())
        print("Warning: Attention weights not available, using uniform attention as fallback")
        return layer_attentions, tokens
    
    # Average over heads for each layer
    # attentions: tuple of (batch_size, num_heads, seq_len, seq_len)
    layer_attentions = []
    for layer_att in attentions:
        # Average over heads: (batch_size, num_heads, seq_len, seq_len) -> (batch_size, seq_len, seq_len)
        if layer_att is not None and len(layer_att.shape) >= 3:
            avg_att = layer_att[0].mean(dim=0).cpu().numpy()  # (seq_len, seq_len)
            layer_attentions.append(avg_att)
        else:
            # Fallback for this layer
            seq_len = inputs['input_ids'].shape[1]
            dummy_attention = np.ones((seq_len, seq_len)) / seq_len
            layer_attentions.append(dummy_attention)
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())
    
    return layer_attentions, tokens


def visualize_attention_comparison(bert_model, modernbert_model, bert_tokenizer, modernbert_tokenizer,
                                   text, device, save_path=None):
    """
    Compare attention patterns between BERT and ModernBERT for the same text
    """
    # Extract attention weights
    bert_attentions, bert_tokens = extract_attention_weights(
        bert_model, bert_tokenizer, text, device
    )
    modernbert_attentions, modernbert_tokens = extract_attention_weights(
        modernbert_model, modernbert_tokenizer, text, device
    )
    
    # Use last layer attention (most relevant for classification)
    bert_last_att = bert_attentions[-1]
    modernbert_last_att = modernbert_attentions[-1]
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # BERT attention
    im1 = axes[0].imshow(bert_last_att, cmap='Blues', aspect='auto')
    axes[0].set_title('BERT Attention (Last Layer)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Token Position')
    axes[0].set_ylabel('Token Position')
    plt.colorbar(im1, ax=axes[0])
    
    # ModernBERT attention
    im2 = axes[1].imshow(modernbert_last_att, cmap='Reds', aspect='auto')
    axes[1].set_title('ModernBERT Attention (Last Layer)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Token Position')
    axes[1].set_ylabel('Token Position')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    
    plt.close()
    
    # Print top attended tokens for each model
    print("\n" + "="*80)
    print("Top Attended Token Pairs (Last Layer)")
    print("="*80)
    
    # BERT top attention
    print("\nBERT - Top 10 attention pairs:")
    bert_flat = bert_last_att.flatten()
    bert_top_indices = np.argsort(bert_flat)[-10:][::-1]
    for idx in bert_top_indices:
        i, j = np.unravel_index(idx, bert_last_att.shape)
        if i < len(bert_tokens) and j < len(bert_tokens):
            print(f"  {bert_tokens[i][:20]:20s} -> {bert_tokens[j][:20]:20s} : {bert_last_att[i,j]:.4f}")
    
    # ModernBERT top attention
    print("\nModernBERT - Top 10 attention pairs:")
    modernbert_flat = modernbert_last_att.flatten()
    modernbert_top_indices = np.argsort(modernbert_flat)[-10:][::-1]
    for idx in modernbert_top_indices:
        i, j = np.unravel_index(idx, modernbert_last_att.shape)
        if i < len(modernbert_tokens) and j < len(modernbert_tokens):
            print(f"  {modernbert_tokens[i][:20]:20s} -> {modernbert_tokens[j][:20]:20s} : {modernbert_last_att[i,j]:.4f}")
    
    return bert_attentions, modernbert_attentions, bert_tokens, modernbert_tokens


def main():
    """
    Main analysis function
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['tsne', 'attention', 'both'], 
                       default='both', help='Analysis mode')
    parser.add_argument('--bert_checkpoint', type=str, 
                       default='checkpoints/bert_base_uncased/epoch_2_acc_0.8810.pt',
                       help='Path to BERT checkpoint')
    parser.add_argument('--modernbert_checkpoint', type=str,
                       default='checkpoints/answerdotai/modernbert_base/epoch_2_acc_0.9092.pt',
                       help='Path to ModernBERT checkpoint')
    parser.add_argument('--n_samples', type=int, default=15,
                       help='Number of boundary samples to analyze')
    parser.add_argument('--text', type=str, default=None,
                       help='Text for attention analysis (if None, uses a sample from boundary errors)')
    
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    cfg = OmegaConf.load(config_path)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    
    if args.mode in ['tsne', 'both']:
        print("\n" + "="*80)
        print("PART 1: t-SNE Boundary Error Analysis")
        print("="*80)
        
        # Create save directory
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analysis_outputs')
        os.makedirs(save_dir, exist_ok=True)
        
        # Analyze BERT
        bert_checkpoint = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.bert_checkpoint)
        bert_samples, bert_emb, bert_labels, bert_preds, bert_dataset, bert_indices = analyze_tsne_boundary_errors(
            'bert-base-uncased', bert_checkpoint, config_path, n_samples=args.n_samples, save_dir=save_dir
        )
        
        # Analyze ModernBERT
        modernbert_checkpoint = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.modernbert_checkpoint)
        modernbert_samples, modernbert_emb, modernbert_labels, modernbert_preds, modernbert_dataset, modernbert_indices = analyze_tsne_boundary_errors(
            'answerdotai/ModernBERT-base', modernbert_checkpoint, config_path, n_samples=args.n_samples, save_dir=save_dir
        )
        
        # Summary
        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append("SUMMARY: Error Analysis Comparison")
        summary_lines.append("="*80)
        summary_lines.append(f"\nBERT:")
        summary_lines.append(f"  Total misclassified: {(bert_preds != bert_labels).sum()}/{len(bert_labels)}")
        summary_lines.append(f"  Misclassification rate: {(bert_preds != bert_labels).sum()/len(bert_labels)*100:.2f}%")
        summary_lines.append(f"  Boundary samples found: {len(bert_samples)}")
        
        summary_lines.append(f"\nModernBERT:")
        summary_lines.append(f"  Total misclassified: {(modernbert_preds != modernbert_labels).sum()}/{len(modernbert_labels)}")
        summary_lines.append(f"  Misclassification rate: {(modernbert_preds != modernbert_labels).sum()/len(modernbert_labels)*100:.2f}%")
        summary_lines.append(f"  Boundary samples found: {len(modernbert_samples)}")
        
        # Print summary
        summary_text = '\n'.join(summary_lines)
        print("\n" + summary_text)
        
        # Save summary
        summary_file = os.path.join(save_dir, 'error_analysis_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\n✅ Summary saved to: {summary_file}")
    
    if args.mode in ['attention', 'both']:
        print("\n" + "="*80)
        print("PART 2: Attention Map Comparison")
        print("="*80)
        
        # Load tokenizers
        from transformers import AutoTokenizer
        
        # BERT
        bert_cfg = OmegaConf.load(config_path)
        bert_model_config_path = os.path.join(os.path.dirname(config_path), 'model', 'bert.yaml')
        if os.path.exists(bert_model_config_path):
            bert_model_cfg = OmegaConf.load(bert_model_config_path)
            bert_cfg = OmegaConf.merge(bert_cfg, bert_model_cfg)
        if 'model' not in bert_cfg:
            bert_cfg.model = OmegaConf.create({})
        bert_cfg.model.name = 'bert-base-uncased'
        bert_model = EncoderForClassification(bert_cfg).to(device)
        bert_checkpoint = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.bert_checkpoint)
        if os.path.exists(bert_checkpoint):
            checkpoint = torch.load(bert_checkpoint, map_location=device)
            bert_model.load_state_dict(checkpoint['model_state_dict'])
        bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
        
        # ModernBERT
        modernbert_cfg = OmegaConf.load(config_path)
        modernbert_model_config_path = os.path.join(os.path.dirname(config_path), 'model', 'modernbert.yaml')
        if os.path.exists(modernbert_model_config_path):
            modernbert_model_cfg = OmegaConf.load(modernbert_model_config_path)
            modernbert_cfg = OmegaConf.merge(modernbert_cfg, modernbert_model_cfg)
        if 'model' not in modernbert_cfg:
            modernbert_cfg.model = OmegaConf.create({})
        modernbert_cfg.model.name = 'answerdotai/ModernBERT-base'
        modernbert_model = EncoderForClassification(modernbert_cfg).to(device)
        modernbert_checkpoint = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.modernbert_checkpoint)
        if os.path.exists(modernbert_checkpoint):
            checkpoint = torch.load(modernbert_checkpoint, map_location=device)
            modernbert_model.load_state_dict(checkpoint['model_state_dict'])
        modernbert_tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base', use_fast=True)
        
        # Get text for analysis
        if args.text is None:
            # Use a sample from boundary errors (prefer misclassified)
            if args.mode == 'both' and 'bert_samples' in locals() and len(bert_samples) > 0:
                sample = bert_samples[0]
                original_idx = bert_indices[sample['index']]
                text = bert_dataset.valid_data[original_idx]['text']
                print(f"\nUsing boundary error sample (BERT misclassified):")
                print(f"Label: {'Positive' if sample['label'] == 1 else 'Negative'}")
                print(f"Prediction: {'Positive' if sample['prediction'] == 1 else 'Negative'}")
            else:
                # Use a complex sentence with irony/sarcasm
                text = "This movie is so good that I fell asleep halfway through. The acting was terrible, the plot was predictable, and the ending was the best part because it meant the movie was finally over."
        else:
            text = args.text
        
        print(f"\nAnalyzing text:")
        print("-" * 80)
        print(text[:200] + ("..." if len(text) > 200 else ""))
        print("-" * 80)
        
        # Visualize attention
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analysis_outputs')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'attention_comparison.png')
        
        visualize_attention_comparison(
            bert_model, modernbert_model,
            bert_tokenizer, modernbert_tokenizer,
            text, device, save_path=save_path
        )
        
        # Save attention analysis log
        attention_log_lines = []
        attention_log_lines.append("="*80)
        attention_log_lines.append("Attention Analysis Complete")
        attention_log_lines.append("="*80)
        
        attention_log_text = '\n'.join(attention_log_lines)
        print("\n" + attention_log_text)
        
        # Save attention log
        attention_log_file = os.path.join(save_dir, 'attention_analysis_log.txt')
        with open(attention_log_file, 'w', encoding='utf-8') as f:
            f.write(attention_log_text)
        print(f"\n✅ Attention analysis log saved to: {attention_log_file}")


if __name__ == "__main__":
    main()


