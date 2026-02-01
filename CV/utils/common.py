"""
공통 학습/평가 함수들을 통합한 모듈
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import calculate_accuracy


def train_one_epoch(model, loader, criterion, optimizer, device, desc="Training"):
    """
    한 에포크 동안 모델을 학습합니다.
    
    Args:
        model: 학습할 모델
        loader: 학습용 DataLoader
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스 (cuda/cpu)
        desc: 진행률 표시 설명
    
    Returns:
        tuple: (평균 손실, top1 정확도, top5 정확도)
    """
    model.train()
    running_loss = 0.0
    top1_sum, top5_sum, total = 0.0, 0.0, 0
    
    for images, labels in tqdm(loader, desc=desc, leave=False):
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
    
    return running_loss / total, top1_sum / total, top5_sum / total


def evaluate(model, loader, criterion, device, desc="Evaluating"):
    """
    모델을 평가합니다 (손실 포함).
    
    Args:
        model: 평가할 모델
        loader: 평가용 DataLoader
        criterion: 손실 함수
        device: 디바이스 (cuda/cpu)
        desc: 진행률 표시 설명
    
    Returns:
        tuple: (평균 손실, top1 정확도, top5 정확도)
    """
    model.eval()
    running_loss = 0.0
    top1_sum, top5_sum, total = 0.0, 0.0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            running_loss += loss.item() * images.size(0)
            top1_sum += acc1.item() * images.size(0)
            top5_sum += acc5.item() * images.size(0)
            total += labels.size(0)
    
    return running_loss / total, top1_sum / total, top5_sum / total


def evaluate_accuracy_only(model, loader, device, desc="Evaluating"):
    """
    모델을 평가합니다 (정확도만).
    
    Args:
        model: 평가할 모델
        loader: 평가용 DataLoader
        device: 디바이스 (cuda/cpu)
        desc: 진행률 표시 설명
    
    Returns:
        tuple: (top1 정확도, top5 정확도)
    """
    model.eval()
    top1_sum, top5_sum, total = 0.0, 0.0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            top1_sum += acc1.item() * images.size(0)
            top5_sum += acc5.item() * images.size(0)
            total += labels.size(0)
    
    return top1_sum / total, top5_sum / total
