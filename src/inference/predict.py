#!/usr/bin/env python3
"""
WebFG-400 预测脚本
用于生成 pred_results_web400.csv 预测结果文件
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_loader import WebFG400Dataset, get_transforms
from model import create_model

def load_model(model_path, num_classes=400, device='cuda'):
    """加载训练好的模型"""
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

def predict_on_dataset(model, data_loader, device='cuda'):
    """在整个数据集上进行预测"""
    all_preds = []
    all_probs = []
    all_image_paths = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc='Predicting')):
            data = data.to(device)
            
            # 获取预测结果
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 获取批次中的图像路径
            batch_start = batch_idx * data_loader.batch_size
            batch_end = min((batch_idx + 1) * data_loader.batch_size, len(data_loader.dataset))
            
            for i in range(len(data)):
                idx_in_dataset = batch_start + i
                if idx_in_dataset < len(data_loader.dataset):
                    all_image_paths.append(data_loader.dataset.image_paths[idx_in_dataset])
                    all_labels.append(data_loader.dataset.labels[idx_in_dataset])
    
    return all_image_paths, all_labels, all_preds, all_probs

def create_prediction_dataframe(image_paths, true_labels, pred_labels, pred_probs, class_names=None):
    """创建预测结果的DataFrame"""
    
    # 提取文件名
    filenames = [os.path.basename(path) for path in image_paths]
    
    # 提取目录名（类别）
    dir_names = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    
    # 获取预测概率
    max_probs = [np.max(prob) for prob in pred_probs]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'image_filename': filenames,
        'true_class': true_labels,
        'predicted_class': pred_labels,
        'confidence': max_probs,
        'directory': dir_names
    })
    
    # 如果有类别名称，添加类别名称列
    if class_names:
        df['true_class_name'] = [class_names[label] for label in true_labels]
        df['predicted_class_name'] = [class_names[label] for label in pred_labels]
    
    return df

def get_class_names(data_dir):
    """获取类别名称列表"""
    class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return class_dirs

def main():
    parser = argparse.ArgumentParser(description='WebFG-400 Prediction Script')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型文件路径')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='测试数据目录路径')
    parser.add_argument('--output_csv', type=str, default='pred_results_web400.csv',
                       help='输出CSV文件名')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='预测批次大小')
    parser.add_argument('--img_size', type=int, default=224,
                       help='图像尺寸')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = load_model(args.model_path, device=device)
    
    # 创建测试数据加载器
    print(f"加载测试数据: {args.data_dir}")
    test_transform = get_transforms('val', args.img_size)
    test_dataset = WebFG400Dataset(args.data_dir, transform=test_transform, mode='train')  # 使用train模式获取所有数据
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # 进行预测
    print("开始预测...")
    image_paths, true_labels, pred_labels, pred_probs = predict_on_dataset(model, test_loader, device)
    
    # 获取类别名称
    class_names = get_class_names(args.data_dir)
    
    # 创建预测结果DataFrame
    print("生成预测结果...")
    results_df = create_prediction_dataframe(image_paths, true_labels, pred_labels, pred_probs, class_names)
    
    # 保存为CSV文件
    results_df.to_csv(args.output_csv, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {args.output_csv}")
    
    # 计算并显示准确率
    accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
    print(f"整体准确率: {accuracy:.4f}")
    
    # 显示预测结果统计
    print("\n预测结果统计:")
    print(f"总样本数: {len(results_df)}")
    print(f"正确预测数: {sum(results_df['true_class'] == results_df['predicted_class'])}")
    print(f"平均置信度: {results_df['confidence'].mean():.4f}")

if __name__ == "__main__":
    main()