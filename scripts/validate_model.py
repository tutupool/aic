#!/usr/bin/env python3
"""
模型验证脚本 - 在测试集上评估模型性能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_loader import WebFG400Dataset, get_transforms
from src.models.model import create_model

def validate_model():
    parser = argparse.ArgumentParser(description='模型验证脚本')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='模型权重文件路径')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='测试集目录路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='数据加载工作线程数')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据加载器
    test_transform = get_transforms('val', img_size=224)
    test_dataset = WebFG400Dataset(args.test_dir, transform=test_transform, mode='val')
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"测试集: {len(test_dataset)} 张图片")
    print(f"类别数: {len(set(test_dataset.labels))}")
    
    # 创建模型
    model = create_model(num_classes=400, pretrained=False)
    
    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # 验证模型
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    print("=" * 50)
    print(f"模型验证结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"正确数/总数: {correct}/{total}")
    print("=" * 50)

if __name__ == "__main__":
    validate_model()