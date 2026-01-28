import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from PIL import Image

# ---------------------------------------------------------
# Config 유틸리티
# ---------------------------------------------------------
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(config_data, config_model):
    """
    두 개의 딕셔너리를 병합합니다.
    Key가 겹칠 경우 model 설정이 data 설정을 덮어씁니다.
    """
    config = config_data.copy()
    config.update(config_model)
    return config

# =========================================================
# CIFAR-10-C (.npy) 전용 데이터셋 클래스
# =========================================================
class CIFAR10C_Dataset(Dataset):
    def __init__(self, root, corruption, severity, transform=None):
        self.transform = transform
        
        npy_path = os.path.join(root, corruption + '.npy')
        label_path = os.path.join(root, 'labels.npy')
        
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"❌ NPY file not found: {npy_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"❌ Labels file not found: {label_path}")
        
        # .npy 파일 로드
        # CIFAR-10-C 구조: [Severity 1 (1만장) | Severity 2 | ... | Severity 5] 순서로 연결됨
        loaded_images = np.load(npy_path)
        loaded_labels = np.load(label_path)
        
        # Severity(1~5)에 맞춰서 데이터 자르기 (Slicing)
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        self.data = loaded_images[start_idx:end_idx]
        self.targets = loaded_labels[start_idx:end_idx]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        # numpy array -> PIL Image 변환 (transforms 적용을 위해 필수)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

# =========================================================
# 학습용 Loader (Train/Val/Test)
# =========================================================
def get_loaders(config):
    data_root = config.get('data_root', './data')
    dataset_name = config.get('dataset_name', 'cifar10').lower()
    batch_size = config.get('batch_size', 128)
    num_workers = config.get('num_workers', 4)
    
    # 이미지 크기 우선순위: config['image_size'] > 데이터셋 기본값
    if 'image_size' in config:
        size = int(config['image_size'])
    elif dataset_name == 'cifar10':
        size = 32
    else:
        size = 224

    # 데이터셋별 정규화 값
    if dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 데이터셋 로드
    if dataset_name == 'cifar10':
        full_train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
        
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        val_dataset.dataset.transform = test_transform 

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# =========================================================
# Robustness 평가용 Loader
# =========================================================
def get_robustness_loader(data_root, corruption, severity, batch_size=128, image_size=None):
        
    size = image_size if image_size else 32
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = CIFAR10C_Dataset(data_root, corruption, severity, transform=transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# =========================================================
# CIFAR-10-C Fine-tuning 전용 Loader
# =========================================================
def get_combined_cifar10c_loader(data_root, corruptions, batch_size=128, split_ratio=0.9, image_size=32):
    """
    [Generalist 학습용]
    모든 Corruption의 '앞쪽 90%' 데이터만 긁어모아서 Train/Val Loader로 만듭니다.
    image_size가 224면 Resize를 수행합니다 (ViT용).
    """
    all_images = []
    all_labels = []
    
    print(f"🔄 Loading Training Data (Combined) | Target Size: {image_size}x{image_size}")
    
    for corruption in corruptions:
        npy_path = os.path.join(data_root, corruption + '.npy')
        label_path = os.path.join(data_root, 'labels.npy')
        
        if os.path.exists(npy_path):
            imgs = np.load(npy_path)
            lbls = np.load(label_path)
            
            if len(lbls) == 10000: lbls = np.concatenate([lbls] * 5)
            
            # 앞쪽 90%
            split_idx = int(len(imgs) * split_ratio)
            all_images.append(imgs[:split_idx])
            all_labels.append(lbls[:split_idx])
        else:
            print(f"⚠️ Warning: {corruption} not found.")

    full_images = np.concatenate(all_images, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    if image_size == 224:
        # ViT용 전처리 (Resize 필수)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # ResNet(CIFAR)용 전처리 (32x32 유지)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    class FullCIFAR10C(Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform
        def __len__(self): return len(self.images)
        def __getitem__(self, idx):
            img = self.images[idx]
            if img.shape[0] == 3 and img.shape[2] != 3: img = img.transpose(1, 2, 0)
            img = Image.fromarray(img)
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]

    dataset = FullCIFAR10C(full_images, full_labels, transform=None)
    
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    
    train_ds = Subset(dataset, range(0, train_size))
    val_ds = Subset(dataset, range(train_size, len(dataset)))
    
    # Wrapper로 Transform 적용
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            return self.transform(img), label

    train_loader = DataLoader(TransformedSubset(train_ds, train_transform), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(TransformedSubset(val_ds, eval_transform), batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

# =========================================================
# CIFAR-10-C Fine-tuning 평가용 Loader
# =========================================================
def get_test_only_robustness_loader(data_root, corruption, severity, batch_size=128, image_size=32):
    """
    [평가용 Loader] 
    image_size 인자를 받아서 224일 경우 Resize를 수행합니다.
    """
    npy_path = os.path.join(data_root, corruption + '.npy')
    label_path = os.path.join(data_root, 'labels.npy')
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"❌ File not found: {npy_path}")
        
    imgs = np.load(npy_path)
    lbls = np.load(label_path)
    
    if len(lbls) == 10000: 
        lbls = np.concatenate([lbls] * 5)
    
    # Severity 별 인덱스 계산 (0~10000, 10000~20000 ...)
    start_idx_sev = (severity - 1) * 10000
    end_idx_sev = severity * 10000
    
    sev_imgs = imgs[start_idx_sev:end_idx_sev]
    sev_lbls = lbls[start_idx_sev:end_idx_sev]
    
    # 뒤쪽 10% (Test Set)만 사용
    split_point = int(10000 * 0.9)
    test_imgs = sev_imgs[split_point:]
    test_lbls = sev_lbls[split_point:]
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # Transform 분기
    if image_size == 224:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    class TestDataset(Dataset):
        def __init__(self, i, l, t):
            self.i = i; self.l = l; self.t = t
        def __len__(self): return len(self.i)
        def __getitem__(self, idx):
            img = self.i[idx]
            img = Image.fromarray(img)
            return self.t(img), self.l[idx]
            
    dataset = TestDataset(test_imgs, test_lbls, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)