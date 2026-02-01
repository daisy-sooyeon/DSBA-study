"""
ImageNet-9 데이터셋 처리 관련 유틸리티
"""
import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class TransformedSubset(Dataset):
    """Transform이 적용된 Subset 클래스"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_train_val_test_loaders(data_root, batch_size):
    """
    Original 데이터를 8:1:1로 나누어 Train/Val/Test Loader를 반환합니다.
    
    Args:
        data_root: ImageNet-9 데이터 루트 경로
        batch_size: 배치 크기
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
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


def get_eval_loader(data_root, mode, batch_size):
    """
    평가용 데이터 로더를 생성합니다 (Mixed, Only_FG 등).
    
    Args:
        data_root: ImageNet-9 데이터 루트 경로
        mode: 데이터 모드 (e.g., 'mixed_rand', 'only_fg')
        batch_size: 배치 크기
    
    Returns:
        DataLoader: 평가용 데이터 로더
    """
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
