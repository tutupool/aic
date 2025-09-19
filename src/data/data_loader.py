import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WebFG400Dataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        WebFG-400 数据集加载器
        
        Args:
            root_dir: 数据集根目录
            transform: 数据增强变换
            mode: 'train', 'val', 或 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self.labels = []
        
        # 遍历所有类别目录
        class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                for img_name in images:
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(class_idx)
        
        # 划分训练集和验证集
        if mode in ['train', 'val']:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                self.image_paths, self.labels, test_size=0.2, random_state=42, stratify=self.labels
            )
            
            if mode == 'train':
                self.image_paths = train_paths
                self.labels = train_labels
            else:
                self.image_paths = val_paths
                self.labels = val_labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 读取图像，处理损坏的文件
        image = cv2.imread(img_path)
        if image is None:
            # 如果图像读取失败，创建一个黑色占位图像
            print(f"警告: 无法读取图像 {img_path}，使用占位图像")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

def get_transforms(mode='train', img_size=224):
    """获取数据增强变换"""
    if mode == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_dataloaders(data_dir, batch_size=32, num_workers=2, img_size=224):
    """创建数据加载器"""
    train_transform = get_transforms('train', img_size)
    val_transform = get_transforms('val', img_size)
    
    train_dataset = WebFG400Dataset(data_dir, transform=train_transform, mode='train')
    val_dataset = WebFG400Dataset(data_dir, transform=val_transform, mode='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 400