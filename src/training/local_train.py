#!/usr/bin/env python3
"""
本地训练脚本 - 针对WebFG-400数据集优化
支持梯度累积，适应各种GPU内存配置
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.data.data_loader import create_dataloaders
from src.models.model import create_model
import argparse
import os
from datetime import datetime
import time

def train_model():
    parser = argparse.ArgumentParser(description='WebFG-400 Local Training')
    parser.add_argument('--data_dir', type=str, default='data/raw/webfg400_train/train', 
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for gradient accumulation')
    parser.add_argument('--accumulation_steps', type=int, default=2, 
                       help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=30, 
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2, 
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./output', 
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, num_classes = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=224
    )
    
    print(f"训练集: {len(train_loader.dataset)} 张图片")
    print(f"验证集: {len(val_loader.dataset)} 张图片")
    
    # 创建模型
    print("创建模型...")
    model = create_model(num_classes=400, pretrained=True)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练记录
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("开始训练...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / args.accumulation_steps  # 梯度累积
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 统计
            running_loss += loss.item() * args.accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 打印进度
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{args.epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item() * args.accumulation_steps:.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 验证阶段
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        # 保存每个epoch的模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': history
        }, os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch: {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
              f'Time: {epoch_time:.1f}s')
        print("-" * 60)
    
    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%")
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'history': history
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # 保存训练历史
    save_training_history(history, args.output_dir)

def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def save_training_history(history, output_dir):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    # 保存历史数据到文本文件
    with open(os.path.join(output_dir, 'training_history.txt'), 'w') as f:
        f.write("Epoch\tTrain_Loss\tTrain_Acc\tVal_Loss\tVal_Acc\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1}\t{history['train_loss'][i]:.4f}\t"
                   f"{history['train_acc'][i]:.2f}\t"
                   f"{history['val_loss'][i]:.4f}\t"
                   f"{history['val_acc'][i]:.2f}\n")

if __name__ == "__main__":
    train_model()