#!/usr/bin/env python3
"""
Kaggle WebFG-400 预测脚本
在Kaggle平台上生成 pred_results_web400.csv 预测结果文件
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_loader import WebFG400Dataset, get_transforms
from model import create_model

def main():
    # Kaggle环境配置
    model_path = '/kaggle/working/best_model.pth'  # Kaggle输出目录中的最佳模型
    data_dir = '/kaggle/input/webfg400-train/webfg400_train/train'  # Kaggle数据集路径
    output_csv = '/kaggle/working/pred_results_web400.csv'  # 输出文件
    
    print("=" * 60)
    print("WebFG-400 Kaggle 预测脚本")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
        return
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请确保已正确上传webfg400-train数据集")
        return
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    try:
        model = create_model(num_classes=400, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 创建测试数据加载器
    print(f"\n加载测试数据: {data_dir}")
    try:
        test_transform = get_transforms('val', 224)
        test_dataset = WebFG400Dataset(data_dir, transform=test_transform, mode='train')
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False,
            num_workers=2, pin_memory=True
        )
        print(f"✓ 数据加载成功，总样本数: {len(test_dataset)}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return
    
    # 进行预测
    print("\n开始预测...")
    all_preds = []
    all_probs = []
    all_image_paths = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc='预测进度')):
            data = data.to(device)
            
            # 获取预测结果
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # 获取图像路径
            batch_start = batch_idx * test_loader.batch_size
            for i in range(len(data)):
                idx_in_dataset = batch_start + i
                if idx_in_dataset < len(test_dataset):
                    all_image_paths.append(test_dataset.image_paths[idx_in_dataset])
    
    # 获取类别名称
    class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # 创建预测结果DataFrame
    print("\n生成预测结果文件...")
    filenames = [os.path.basename(path) for path in all_image_paths]
    dir_names = [os.path.basename(os.path.dirname(path)) for path in all_image_paths]
    max_probs = [np.max(prob) for prob in all_probs]
    
    results_df = pd.DataFrame({
        'image_id': [f.split('.')[0] for f in filenames],  # 去除扩展名的图像ID
        'image_filename': filenames,
        'true_label': all_labels,
        'predicted_label': all_preds,
        'confidence': max_probs,
        'true_class': [class_dirs[label] for label in all_labels],
        'predicted_class': [class_dirs[label] for label in all_preds],
        'directory': dir_names
    })
    
    # 保存为CSV文件
    results_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✓ 预测结果已保存到: {output_csv}")
    
    # 计算准确率
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"\n预测完成!")
    print(f"整体准确率: {accuracy:.4f}")
    print(f"总样本数: {len(results_df)}")
    print(f"正确预测数: {sum(results_df['true_label'] == results_df['predicted_label'])}")
    print(f"平均置信度: {results_df['confidence'].mean():.4f}")
    
    # 显示文件信息
    file_size = os.path.getsize(output_csv) / 1024 / 1024  # MB
    print(f"\n输出文件信息:")
    print(f"文件路径: {output_csv}")
    print(f"文件大小: {file_size:.2f} MB")
    print(f"数据列数: {len(results_df.columns)}")
    print(f"数据行数: {len(results_df)}")
    
    print("\n" + "=" * 60)
    print("预测完成！您现在可以下载 pred_results_web400.csv 文件提交")
    print("=" * 60)

if __name__ == "__main__":
    main()