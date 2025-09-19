import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Webinat5000Dataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Webinat5000 数据集加载器
        
        Args:
            root_dir: 数据集根目录（包含train目录）
            transform: 数据增强变换
            mode: 'train', 'val', 或 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self.labels = []
        
        # Webinat5000的数据集结构：root_dir/train/0000/, root_dir/train/0001/, ...
        train_dir = os.path.join(root_dir, 'train')
        
        if not os.path.exists(train_dir):
            raise ValueError(f"训练目录不存在: {train_dir}")
        
        # 遍历所有类别目录（0000-0998）
        class_dirs = sorted([d for d in os.listdir(train_dir) 
                           if os.path.isdir(os.path.join(train_dir, d)) and d.isdigit()])
        
        print(f"找到 {len(class_dirs)} 个类别")
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(train_dir, class_dir)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                for img_name in images:
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(int(class_dir))  # 使用实际的类别编号（0000=0, 0001=1, ...）
        
        print(f"总共 {len(self.image_paths)} 张图像")
        
        # 划分训练集和验证集（仅对训练模式）
        if mode in ['train', 'val'] and len(self.image_paths) > 0:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                self.image_paths, self.labels, test_size=0.2, random_state=42, stratify=self.labels
            )
            
            if mode == 'train':
                self.image_paths = train_paths
                self.labels = train_labels
                print(f"训练集: {len(self.image_paths)} 张图像")
            else:
                self.image_paths = val_paths
                self.labels = val_labels
                print(f"验证集: {len(self.image_paths)} 张图像")
    
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

def get_webinat5000_transforms(mode='train', img_size=224):
    """获取Webinat5000数据增强变换"""
    if mode == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_webinat5000_dataloaders(data_dir, batch_size=32, num_workers=2, img_size=224):
    """创建Webinat5000数据加载器"""
    train_transform = get_webinat5000_transforms('train', img_size)
    val_transform = get_webinat5000_transforms('val', img_size)
    
    train_dataset = Webinat5000Dataset(data_dir, transform=train_transform, mode='train')
    val_dataset = Webinat5000Dataset(data_dir, transform=val_transform, mode='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, 5000  # Webinat5000有5000个类别