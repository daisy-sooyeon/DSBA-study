import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModel

os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import EncoderForClassification, masked_mean_pooling
from data import get_dataloader, IMDBDataset

def get_valid_attention(matrix, tokens, exclude_tokens=['[PAD]', '[CLS]', '[SEP]', '<s>', '</s>', '<pad>', '<mask>']):
    """특수 토큰을 제외한 실제 단어들 간의 어텐션 행렬만 추출"""
    valid_indices = [i for i, t in enumerate(tokens) if t not in exclude_tokens]
    if not valid_indices:
        return matrix, tokens
    
    filtered_matrix = matrix[valid_indices, :][:, valid_indices]
    filtered_tokens = [tokens[i] for i in valid_indices]
    
    row_sums = filtered_matrix.sum(axis=-1, keepdims=True)
    filtered_matrix = np.divide(filtered_matrix, row_sums, out=np.zeros_like(filtered_matrix), where=row_sums!=0)
    
    return filtered_matrix, filtered_tokens

def load_model_with_eager(model_name, checkpoint_path, config_path):
    """어텐션 추출을 위해 attn_implementation='eager'로 모델 로드"""
    cfg = OmegaConf.load(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_cfg_path = os.path.join(os.path.dirname(config_path), 'model', 
                                 'bert.yaml' if 'modern' not in model_name.lower() else 'modernbert.yaml')
    if os.path.exists(model_cfg_path):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(model_cfg_path))
    
    cfg.model.name = model_name
    
    model = EncoderForClassification(cfg)
    
    try:
        model.encoder = AutoModel.from_pretrained(
            model_name, 
            attn_implementation="eager",
            add_pooling_layer=False 
        )
    except TypeError:
        model.encoder = AutoModel.from_pretrained(
            model_name, 
            attn_implementation="eager"
        )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ Loaded weights: {checkpoint_path}")
        print(f"   Missing keys (can ignore if pooler): {msg.missing_keys}")
    
    model = model.to(device).eval()
    return model, cfg, device

def extract_features(model, dataloader, device):
    features, predictions, labels, confidences = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            target = batch['label'].to(device)
            
            outputs = model.encoder(**inputs)
            pooled = masked_mean_pooling(outputs['last_hidden_state'], inputs['attention_mask'])
            logits = model.classifier(pooled)
            
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            
            features.append(pooled.cpu())
            predictions.append(pred.cpu())
            labels.append(target.cpu())
            confidences.append(conf.cpu())
            
    return (torch.cat(features).numpy(), torch.cat(predictions).numpy(), 
            torch.cat(labels).numpy(), torch.cat(confidences).numpy())

def visualize_attention(bert_model, modern_model, bert_tok, modern_tok, text, device, save_path):
    """BERT와 ModernBERT의 어텐션 맵 비교 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(24, 10)) 
    
    models = [(bert_model, bert_tok, "BERT (12 Layers)", "Blues"), 
              (modern_model, modern_tok, "ModernBERT (22 Layers)", "Reds")]
    
    for i, (m, tok, name, cmap) in enumerate(models):
        inputs = tok(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = m.encoder(**inputs, output_attentions=True)
            if outputs.attentions is None:
                print(f"⚠️ {name} failed to return attentions. Skipping.")
                continue
                
            attentions = outputs.attentions[-1][0].mean(dim=0).cpu().numpy() 
            tokens = tok.convert_ids_to_tokens(inputs['input_ids'][0])
        
        filtered_att, filtered_tokens = get_valid_attention(attentions, tokens)

        step = 3
            
        sparse_labels = [t if idx % step == 0 else "" for idx, t in enumerate(filtered_tokens)]
        
        sns.heatmap(filtered_att, 
                    xticklabels=sparse_labels, 
                    yticklabels=sparse_labels,
                    cmap=cmap, ax=axes[i], annot=False, cbar_kws={'shrink': .8})
        
        axes[i].set_title(f"{name} Final Layer Attention", fontsize=16, fontweight='bold')
        
        axes[i].tick_params(axis='x', rotation=90, labelsize=9)
        axes[i].tick_params(axis='y', rotation=0, labelsize=9)

    plt.suptitle(f"Attention Comparison on Error Sample (Label Overlap Fixed)", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"📊 Filtered attention map saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_ckpt', type=str, required=True)
    parser.add_argument('--modern_ckpt', type=str, required=True)
    args = parser.parse_args()

    config_path = "configs/config.yaml"
    save_dir = "analysis_outputs"
    os.makedirs(save_dir, exist_ok=True)

    print("\n[Step 1] Loading models in Eager Mode...")
    bert_model, _, device = load_model_with_eager('bert-base-uncased', args.bert_ckpt, config_path)
    modern_model, cfg, _ = load_model_with_eager('answerdotai/ModernBERT-base', args.modern_ckpt, config_path)
    
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    modern_tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    
    valid_loader = get_dataloader(cfg, 'valid')
    dataset = IMDBDataset(cfg, 'valid')

    print("\n[Step 2] Running Error Analysis...")
    feat, pred, label, conf = extract_features(modern_model, valid_loader, device)
    
    errors = np.where((pred != label) & (conf > 0.8))[0]
    if len(errors) == 0: errors = np.where(pred != label)[0]
    
    top_error_idx = errors[0]
    error_text = dataset.valid_data[int(top_error_idx)]['text']
    
    print(f"📍 Targeting Boundary Error (Index {top_error_idx}):")
    print(f"   Label: {label[top_error_idx]} | Pred: {pred[top_error_idx]} | Confidence: {conf[top_error_idx]:.4f}")
    print(f"   Text Snippet: {error_text[:150]}...")

    print("\n[Step 3] Visualizing Attention (Eager mode enabled)...")
    visualize_attention(bert_model, modern_model, bert_tokenizer, modern_tokenizer, 
                        error_text, device, f"{save_dir}/filtered_attention_compare.png")

if __name__ == "__main__":
    main()