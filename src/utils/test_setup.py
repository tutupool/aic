#!/usr/bin/env python3
"""
测试脚本 - 验证代码库设置是否正确
"""

import os
import sys
import torch
import torchvision
import numpy as np

def test_imports():
    """测试所有必要的导入"""
    print("测试导入...")
    
    try:
        import data_loader
        import model
        import train
        print("✓ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_torch():
    """测试PyTorch环境"""
    print("\n测试PyTorch环境...")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"TorchVision版本: {torchvision.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    
    return True

def test_model():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        from model import create_model, count_parameters
        
        # 创建模型
        model = create_model(num_classes=400, pretrained=False)
        param_count = count_parameters(model)
        
        print(f"✓ 模型创建成功")
        print(f"模型参数量: {param_count:,}")
        
        # 测试前向传播
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
        print("✓ 前向传播测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False

def test_data_loader():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    
    try:
        from data_loader import WebFG400Dataset, get_transforms
        
        # 测试数据变换
        transform = get_transforms('train', 224)
        print("✓ 数据变换创建成功")
        
        # 测试数据集类（使用虚拟路径）
        dataset = WebFG400Dataset(
            root_dir="./dummy_data",
            transform=transform,
            mode='train'
        )
        print("✓ 数据集类创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("WebFG-400 Kaggle训练代码库测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_torch,
        test_model,
        test_data_loader
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    
    if all(results):
        print("🎉 所有测试通过！代码库设置正确。")
        print("\n下一步:")
        print("1. 将webfg400_train.zip上传到Kaggle数据集")
        print("2. 在Kaggle创建新内核")
        print("3. 上传本代码库所有文件")
        print("4. 运行 kaggle_train.py 开始训练")
        return True
    else:
        print("❌ 部分测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)