import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModel

# 경고 무시 및 환경 설정
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# model, data 모듈 임포트
from model import EncoderForClassification, masked_mean_pooling
from data import get_dataloader, IMDBDataset

def get_valid_attention(matrix, tokens, exclude_tokens=['[PAD]', '<pad>', '<mask>']):
    """[PAD]만 제외하고 [CLS], [SEP] 등은 포함하여 어텐션 행렬 추출"""
    valid_indices = [i for i, t in enumerate(tokens) if t not in exclude_tokens]
    if not valid_indices:
        return matrix, tokens
    
    filtered_matrix = matrix[valid_indices, :][:, valid_indices]
    filtered_tokens = [tokens[i] for i in valid_indices]
    
    return filtered_matrix, filtered_tokens

def load_model_with_eager(model_name, checkpoint_path, config_path):
    """어텐션 추출을 위해 attn_implementation='eager'로 모델 로드"""
    
    if os.path.exists(config_path):
        base_cfg = OmegaConf.load(config_path)
    else:
        base_cfg = OmegaConf.create()
        
    temp_cfg = OmegaConf.create({
        'model': {
            'name': model_name,
            'hidden_size': 768,   # 필수: Base 모델 히든 사이즈
            'num_labels': 2,      # 필수: IMDB 분류 클래스 수
            'dropout_prob': 0.1   # 필수: 드롭아웃 (기본값)
        },
        'data': {
            'max_len': 128        
        }
    })
    cfg = OmegaConf.merge(base_cfg, temp_cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # 체크포인트 로드
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # 키 매칭이 안 될 수 있으므로(classifier 등) strict=False로 로드
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
    
    model = model.to(device).eval()
    return model, cfg, device

def extract_features(model, dataloader, device):
    """
    에러 분석을 위해 모델의 예측값과 실제 정답을 추출하는 함수
    """
    features, predictions, labels, confidences = [], [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting predictions"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            
            # ModernBERT는 token_type_ids를 받지 않음
            if "modern" in model.model_name.lower() and "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            target = batch['label'].to(device)
            
            # Forward Pass
            outputs = model.encoder(**inputs)
            
            # Pooling
            if hasattr(model, 'pooler'):
                 # 만약 내부에 pooler가 따로 정의되어 있다면
                pooled = model.pooler(outputs['last_hidden_state'], inputs['attention_mask'])
            else:
                # 직접 pooling 수행
                pooled = masked_mean_pooling(outputs['last_hidden_state'], inputs['attention_mask'])
            
            logits = model.classifier(pooled)
            
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            
            predictions.append(pred.cpu())
            labels.append(target.cpu())
            confidences.append(conf.cpu())
            
    return (None, torch.cat(predictions).numpy(), 
            torch.cat(labels).numpy(), torch.cat(confidences).numpy())

def visualize_all_layers(model, tokenizer, text, device, model_name, save_dir):
    """
    단일 모델의 모든 레이어 어텐션을 시각화하여 저장
    """
    # Forward Pass
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
    
    with torch.no_grad():
        # ModernBERT용 token_type_ids 제거
        if "modern" in model_name.lower() and "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        
        outputs = model.encoder(**inputs, output_attentions=True)
        
        if outputs.attentions is None:
            print(f"⚠️ {model_name} attentions not found. (Check 'output_attentions=True')")
            return

        all_attentions = outputs.attentions
        num_layers = len(all_attentions)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print(f"🎨 Visualizing {num_layers} layers for {model_name}...")

    # Plotting Setup
    cols = 4
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for i, layer_attn in enumerate(all_attentions):
        # Head Averaging
        avg_attn = layer_attn[0].mean(dim=0).cpu().numpy()
        
        # 유효 토큰 필터링
        filtered_att, filtered_tokens = get_valid_attention(avg_attn, tokens)
        
        # Heatmap
        sns.heatmap(filtered_att, 
                    xticklabels=filtered_tokens, 
                    yticklabels=filtered_tokens,
                    cmap="Blues" if "BERT" in model_name else "Reds",
                    ax=axes[i], annot=False, cbar=False)
        
        axes[i].set_title(f"Layer {i+1}", fontsize=12, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=90, labelsize=3)
        axes[i].tick_params(axis='y', rotation=0, labelsize=3)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{model_name} - All Layers Attention Flow", fontsize=20, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"attn_all_layers_{model_name.replace('/', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved to: {save_path}")
    plt.close()

def visualize_last_layer(model, tokenizer, text, device, model_name, save_dir):
    """
    단일 모델의 마지막 레이어 어텐션만 시각화하여 크게 저장
    """
    # Forward Pass
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
    
    with torch.no_grad():
        # ModernBERT용 token_type_ids 제거
        if "modern" in model_name.lower() and "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        
        outputs = model.encoder(**inputs, output_attentions=True)
        
        if outputs.attentions is None:
            print(f"⚠️ {model_name} attentions not found. (Check 'output_attentions=True')")
            return

        # 마지막 레이어만 선택 (-1 인덱스)
        last_layer_attn = outputs.attentions[-1]
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print(f"🎨 Visualizing LAST layer for {model_name}...")

    # Head Averaging
    avg_attn = last_layer_attn[0].mean(dim=0).cpu().numpy()
    
    # 유효 토큰 필터링
    filtered_att, filtered_tokens = get_valid_attention(avg_attn, tokens)
    
    plt.figure(figsize=(10, 8))

    display_tokens = [t.replace('Ġ', '') for t in filtered_tokens]
    
    sns.heatmap(filtered_att, 
                xticklabels=display_tokens, 
                yticklabels=display_tokens,
                cmap="Blues" if "BERT" in model_name else "Reds",
                annot=False, cbar=True)
    
    plt.title(f"{model_name} - Last Layer Attention", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"attn_last_layer_{model_name.replace('/', '_')}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"📊 Saved to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_ckpt', type=str, required=True)
    parser.add_argument('--modern_ckpt', type=str, required=True)
    args = parser.parse_args()

    config_path = "configs/config.yaml"
    save_dir = "analysis_outputs"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 모델 & 토크나이저 로드
    print("\n[Step 1] Loading models...")
    bert_model, _, device = load_model_with_eager('bert-base-uncased', args.bert_ckpt, config_path)
    modern_model, cfg, _ = load_model_with_eager('answerdotai/ModernBERT-base', args.modern_ckpt, config_path)
    
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    modern_tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    
    # 2. 맞춘 정답 데이터 찾기 (High Confidence Correct Prediction)
    print("\n[Step 2] Finding a correct prediction sample...")
    
    try:
        valid_loader = get_dataloader(cfg, 'valid')
        dataset = IMDBDataset(cfg, 'valid')
        
        _, pred, label, conf = extract_features(modern_model, valid_loader, device)
        
        corrects = np.where((pred == label) & (conf > 0.9))[0]
        
        if len(corrects) == 0: 
            print("⚠️ No >0.9 confidence correct predictions found. Using a random correct prediction.")
            corrects = np.where(pred == label)[0]
        
        if len(corrects) > 0:
            target_idx = corrects[0]
            target_text = dataset.valid_data[int(target_idx)]['text']
            print(f"📍 Targeting Correct Index {target_idx}")
            print(f"   Prediction: {pred[target_idx]}, Actual Label: {label[target_idx]}, Confidence: {conf[target_idx]:.4f}")
            print(f"   Snippet: {target_text[:100]}...")
        else:
            print("⚠️ No correct predictions found at all! Using a dummy text.")
            target_text = "This movie is absolutely wonderful and I highly recommend it to everyone."
            
    except Exception as e:
        print(f"⚠️ Error during feature extraction: {e}")
        print("➡️ Using dummy text for visualization.")
        target_text = "This movie is absolutely wonderful and I highly recommend it to everyone."

    # 3. 시각화 실행
    print("\n[Step 3] Visualizing All Layers...")
    visualize_all_layers(bert_model, bert_tokenizer, target_text, device, "BERT-Base", save_dir)
    visualize_all_layers(modern_model, modern_tokenizer, target_text, device, "ModernBERT-Base", save_dir)

    print("\n[Step 4] Visualizing Last Layer Only...")
    visualize_last_layer(bert_model, bert_tokenizer, target_text, device, "BERT-Base", save_dir)
    visualize_last_layer(modern_model, modern_tokenizer, target_text, device, "ModernBERT-Base", save_dir)
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()