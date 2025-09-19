import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_loader import create_dataloaders
from model import create_model

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播和优化
        loss.backward()
        
        # 梯度累积：每2个batch更新一次
        if (batch_idx + 1) % 2 == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        
        running_loss += loss.item()
        
        # 计算准确率
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': accuracy_score(all_labels, all_preds)
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': accuracy_score(all_labels, all_preds)
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_model(config):
    """训练模型主函数"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, num_classes = create_dataloaders(
        config['data_dir'], 
        config['batch_size'], 
        config['num_workers'],
        config['img_size']
    )
    
    # 创建模型
    model = create_model(num_classes=num_classes, pretrained=config['pretrained'])
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # 训练循环
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None)
        
        # 验证
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, os.path.join(config['output_dir'], 'best_model.pth'))
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
        
        # 保存每个epoch的模型
        if config['save_all_epochs']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, os.path.join(config['output_dir'], f'model_epoch_{epoch+1}.pth'))
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, os.path.join(config['output_dir'], 'final_model.pth'))
    
    return model, history

def plot_training_history(history, output_dir):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

if __name__ == "__main__":
    # 配置参数
    config = {
        'data_dir': 'data/raw/webfg400_train/train',  # 本地数据路径
        'output_dir': './output',  # 本地输出目录
        'batch_size': 16,  # 减小batch size以适应本地GPU内存
        'accumulation_steps': 2,  # 梯度累积步数
        'num_workers': 2,
        'img_size': 224,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 20,
        'pretrained': True,
        'save_all_epochs': False,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 开始训练
    model, history = train_model(config)
    
    # 绘制训练历史
    plot_training_history(history, config['output_dir'])
    
    print("训练完成！")