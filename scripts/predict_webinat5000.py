#!/usr/bin/env python3
"""
Webinat5000 测试集预测脚本
专门用于处理Webinat5000测试集
生成符合比赛要求的 pred_results_webinat5000.csv 文件
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.append('src')
from models.model import create_model

def load_model(model_path, num_classes=5000, device='cuda'):
    """加载训练好的Webinat5000模型"""
    # 创建模型架构
    model = create_model(num_classes=num_classes, pretrained=False)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def get_test_transforms(img_size=224):
    """获取测试集数据预处理"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def predict_on_test_set(model, test_dir, device='cuda', batch_size=16, img_size=224):
    """在测试集上进行预测"""
    # 获取所有测试图像文件
    image_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    print(f"找到 {len(image_files)} 张测试图像")
    
    # 数据预处理
    transform = get_test_transforms(img_size)
    
    all_preds = []
    all_filenames = []
    
    # 批量预测
    for i in tqdm(range(0, len(image_files), batch_size), desc='预测进度'):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        # 加载和预处理图像
        for file_path in batch_files:
            try:
                image = Image.open(file_path).convert('RGB')
                image = transform(image)
                batch_images.append(image)
            except Exception as e:
                print(f"无法加载图像 {file_path}: {e}")
                continue
        
        if not batch_images:
            continue
            
        # 转换为tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(batch_tensor)
            preds = outputs.argmax(dim=1)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_filenames.extend([os.path.basename(f) for f in batch_files])
    
    return all_filenames, all_preds

def get_class_names_from_train_dir(train_dir):
    """从训练目录获取类别名称列表"""
    class_dirs = sorted([d for d in os.listdir(train_dir) 
                       if os.path.isdir(os.path.join(train_dir, d))])
    return class_dirs

def main():
    parser = argparse.ArgumentParser(description='Webinat5000 测试集预测脚本')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的Webinat5000模型文件路径')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Webinat5000测试数据目录路径')
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Webinat5000训练数据目录路径（用于获取类别名称）')
    parser.add_argument('--output_csv', type=str, default='pred_results_webinat5000.csv',
                       help='输出CSV文件名')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='预测批次大小')
    parser.add_argument('--img_size', type=int, default=224,
                       help='图像尺寸')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查目录是否存在
    if not os.path.exists(args.test_dir):
        print(f"错误: 测试目录不存在: {args.test_dir}")
        return
    
    if not os.path.exists(args.train_dir):
        print(f"错误: 训练目录不存在: {args.train_dir}")
        return
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    
    # 手动指定类别数量为5000（模型是为5000个类别训练的）
    num_classes = 5000
    print(f"使用固定类别数量: {num_classes}")
    
    model = load_model(args.model_path, num_classes=num_classes, device=device)
    
    # 生成5000个类别的名称（四位数字格式）
    print("生成5000个类别的名称...")
    class_names = [str(i).zfill(4) for i in range(5000)]
    print(f"生成 {len(class_names)} 个类别名称")
    
    # 进行预测
    print("开始预测Webinat5000测试集...")
    filenames, pred_labels = predict_on_test_set(
        model, args.test_dir, device, args.batch_size, args.img_size
    )
    
    # 创建符合比赛要求的预测结果DataFrame
    print("生成符合比赛要求的预测结果...")
    
    # 确保类别名称为四位数字格式（前面补0）
    predicted_classes = []
    for label in pred_labels:
        class_name = class_names[label]
        # 确保类别名称为4位数字格式
        if len(class_name) < 4:
            class_name = class_name.zfill(4)
        predicted_classes.append(class_name)
    
    # 创建只包含两列的DataFrame：图片文件名和四位数字类别名
    results_df = pd.DataFrame({
        'image_filename': filenames,
        'predicted_class': predicted_classes
    })
    
    # 保存为CSV文件（只包含两列，符合比赛要求）
    results_df.to_csv(args.output_csv, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {args.output_csv}")
    
    # 显示文件信息
    print(f"\n输出文件信息:")
    print(f"文件路径: {os.path.abspath(args.output_csv)}")
    print(f"数据行数: {len(results_df)}")
    print(f"数据列数: {len(results_df.columns)}")
    print(f"列名: {list(results_df.columns)}")
    
    # 显示预测统计
    print(f"\n预测统计:")
    print(f"总测试图像: {len(results_df)}")
    
    # 显示前几个预测结果示例（符合比赛格式）
    print(f"\n前5个预测结果示例（比赛格式）:")
    for i in range(min(5, len(results_df))):
        filename = results_df.iloc[i]['image_filename']
        class_name = results_df.iloc[i]['predicted_class']
        print(f"{filename}, {class_name}")

if __name__ == "__main__":
    main()