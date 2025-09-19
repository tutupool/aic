#!/usr/bin/env python3
"""
Webinat5000 完整处理流程脚本
包括数据准备、训练和预测
"""

import os
import subprocess
import sys
import argparse

def setup_webinat5000():
    """设置Webinat5000数据集"""
    print("=" * 60)
    print("Webinat5000 数据集处理流程")
    print("=" * 60)
    
    # 创建目录
    os.makedirs("data/raw/webinat5000_train", exist_ok=True)
    os.makedirs("data/raw/webinat5000_test_A", exist_ok=True)
    os.makedirs("outputs/models_webinat5000", exist_ok=True)
    
    # 检查数据集文件
    train_zip = "data/external/webinat5000_train.zip"
    test_zip = "data/external/webinat5000_test_A.zip"
    
    if not os.path.exists(train_zip):
        print(f"错误: 训练集文件不存在: {train_zip}")
        return False
    
    if not os.path.exists(test_zip):
        print(f"错误: 测试集文件不存在: {test_zip}")
        return False
    
    print("✓ 目录结构已创建")
    
    # 解压训练集
    print("解压Webinat5000训练集...")
    try:
        import zipfile
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall("data/raw/webinat5000_train")
        print("✓ 训练集解压完成")
    except Exception as e:
        print(f"✗ 训练集解压失败: {e}")
        return False
    
    # 解压测试集
    print("解压Webinat5000测试集...")
    try:
        import zipfile
        with zipfile.ZipFile(test_zip, 'r') as zip_ref:
            zip_ref.extractall("data/raw/webinat5000_test_A")
        print("✓ 测试集解压完成")
    except Exception as e:
        print(f"✗ 测试集解压失败: {e}")
        return False
    
    # 检查数据集结构
    train_dir = "data/raw/webinat5000_train"
    if os.path.exists(train_dir):
        class_dirs = [d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))]
        print(f"✓ 训练集包含 {len(class_dirs)} 个类别")
    
    test_dir = "data/raw/webinat5000_test_A"
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"✓ 测试集包含 {len(test_files)} 张图像")
    
    return True

def train_webinat5000():
    """训练Webinat5000模型"""
    print("\n" + "=" * 60)
    print("开始训练Webinat5000模型")
    print("=" * 60)
    
    # 运行训练脚本
    cmd = [
        sys.executable, "scripts/train_webinat5000.py",
        "--data_dir", "data/raw/webinat5000_train",
        "--batch_size", "8",
        "--accumulation_steps", "4",
        "--epochs", "20",
        "--lr", "0.0005",
        "--output_dir", "outputs/models_webinat5000"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ 训练完成")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 训练失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def predict_webinat5000():
    """预测Webinat5000测试集"""
    print("\n" + "=" * 60)
    print("开始预测Webinat5000测试集")
    print("=" * 60)
    
    # 查找最佳模型
    best_model = "outputs/models_webinat5000/best_model_webinat5000.pth"
    if not os.path.exists(best_model):
        print("找不到最佳模型，使用最终模型")
        best_model = "outputs/models_webinat5000/final_model_webinat5000.pth"
    
    if not os.path.exists(best_model):
        print("✗ 找不到训练好的模型")
        return False
    
    # 运行预测脚本
    cmd = [
        sys.executable, "scripts/predict_webinat5000.py",
        "--model_path", best_model,
        "--test_dir", "data/raw/webinat5000_test_A",
        "--train_dir", "data/raw/webinat5000_train",
        "--output_csv", "pred_results_webinat5000.csv",
        "--batch_size", "16"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ 预测完成")
        print(result.stdout)
        
        # 检查预测结果
        if os.path.exists("pred_results_webinat5000.csv"):
            import pandas as pd
            df = pd.read_csv("pred_results_webinat5000.csv")
            print(f"\n预测结果文件信息:")
            print(f"文件路径: {os.path.abspath('pred_results_webinat5000.csv')}")
            print(f"数据行数: {len(df)}")
            print(f"数据列数: {len(df.columns)}")
            print(f"列名: {list(df.columns)}")
            print(f"\n前5个预测结果:")
            print(df.head())
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 预测失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Webinat5000完整处理流程')
    parser.add_argument('--setup_only', action='store_true',
                       help='仅设置数据集，不进行训练和预测')
    parser.add_argument('--train_only', action='store_true',
                       help='仅进行训练，不进行预测')
    parser.add_argument('--predict_only', action='store_true',
                       help='仅进行预测，假设已有训练好的模型')
    
    args = parser.parse_args()
    
    # 设置数据集
    if not args.predict_only:
        if not setup_webinat5000():
            return
    
    # 训练模型
    if not args.predict_only and not args.setup_only:
        if not train_webinat5000():
            return
    
    # 预测测试集
    if not args.train_only and not args.setup_only:
        if not predict_webinat5000():
            return
    
    print("\n" + "=" * 60)
    print("Webinat5000处理流程完成!")
    print("=" * 60)
    print("生成的预测文件: pred_results_webinat5000.csv")
    print("可以提交到官方网站进行评分")

if __name__ == "__main__":
    main()