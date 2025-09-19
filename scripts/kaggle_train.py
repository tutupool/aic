#!/usr/bin/env python3
"""
Kaggle WebFG-400 训练脚本
适用于Kaggle平台的GPU训练
"""

import os
import torch
import argparse
from train import train_model

def main():
    parser = argparse.ArgumentParser(description='WebFG-400 Kaggle Training')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/webfg400-train/train',
                       help='训练数据目录路径')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working',
                       help='输出目录路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='数据加载工作线程数')
    parser.add_argument('--img_size', type=int, default=224,
                       help='图像尺寸')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--pretrained', type=bool, default=True,
                       help='是否使用预训练权重')
    parser.add_argument('--save_all_epochs', type=bool, default=False,
                       help='是否保存所有epoch的模型')
    
    args = parser.parse_args()
    
    # 配置参数
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'img_size': args.img_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'pretrained': args.pretrained,
        'save_all_epochs': args.save_all_epochs
    }
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    else:
        print("警告: 未检测到GPU，使用CPU训练")
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("开始训练WebFG-400模型...")
    print(f"数据目录: {config['data_dir']}")
    print(f"输出目录: {config['output_dir']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"学习率: {config['lr']}")
    print(f"训练轮数: {config['epochs']}")
    
    # 开始训练
    model, history = train_model(config)
    
    print("训练完成！")
    print(f"最佳验证准确率: {max(history['val_acc']):.4f}")
    
    # 保存训练配置
    config_path = os.path.join(config['output_dir'], 'training_config.txt')
    with open(config_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write(f"best_val_acc: {max(history['val_acc']):.4f}\n")
    
    print(f"配置已保存到: {config_path}")

if __name__ == "__main__":
    main()