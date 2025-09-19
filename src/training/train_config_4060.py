#!/usr/bin/env python3
"""
NVIDIA RTX 4060 显卡优化训练配置
针对8GB显存进行优化
"""

import torch

def get_optimal_config():
    """根据4060显卡特性返回最优配置"""
    
    # 检查GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        print(f"检测到GPU: {gpu_name}")
        print(f"显存大小: {gpu_memory:.1f} GB")
        
        # RTX 4060 8GB 优化配置
        if "4060" in gpu_name and gpu_memory <= 8:
            config = {
                'batch_size': 16,           # 适合8GB显存
                'accumulation_steps': 2,    # 梯度累积步数
                'img_size': 224,            # 输入图像尺寸
                'num_workers': 4,           # 数据加载线程数
                'mixed_precision': True,    # 启用混合精度训练
                'gradient_clip': 1.0,       # 梯度裁剪
                'optimizer': 'adamw',       # 优化器
                'lr': 1e-4,                # 学习率
                'weight_decay': 1e-4,       # 权重衰减
            }
            print("使用RTX 4060 8GB优化配置")
            return config
    
    # 默认配置
    default_config = {
        'batch_size': 8,
        'accumulation_steps': 4,
        'img_size': 224,
        'num_workers': 2,
        'mixed_precision': True,
        'gradient_clip': 1.0,
        'optimizer': 'adam',
        'lr': 1e-4,
        'weight_decay': 1e-4,
    }
    print("使用默认配置")
    return default_config

def setup_mixed_precision():
    """设置混合精度训练"""
    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        print("混合精度训练已启用")
        return scaler, autocast
    except ImportError:
        print("混合精度训练不可用")
        return None, None

def check_system_resources():
    """检查系统资源"""
    import psutil
    
    # CPU信息
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=1)
    
    # 内存信息
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024**3)
    available_memory = memory.available / (1024**3)
    
    print(f"CPU核心: {cpu_cores}物理核心, {cpu_threads}逻辑核心")
    print(f"CPU使用率: {cpu_usage}%")
    print(f"总内存: {total_memory:.1f} GB")
    print(f"可用内存: {available_memory:.1f} GB")
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_used = torch.cuda.memory_allocated() / (1024**3)
        gpu_free = gpu_memory - gpu_used
        
        print(f"GPU显存: {gpu_memory:.1f} GB (已用: {gpu_used:.1f} GB, 可用: {gpu_free:.1f} GB)")

if __name__ == "__main__":
    print("=" * 50)
    print("NVIDIA RTX 4060 训练配置检查")
    print("=" * 50)
    
    # 检查系统资源
    check_system_resources()
    
    print("\n" + "=" * 50)
    print("推荐训练配置:")
    print("=" * 50)
    
    config = get_optimal_config()
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # 设置混合精度
    scaler, autocast = setup_mixed_precision()
    
    print("\n" + "=" * 50)
    print("训练命令:")
    print("=" * 50)
    print("python local_train.py \\")
    print(f"  --batch_size {config['batch_size']} \\")
    print(f"  --accumulation_steps {config['accumulation_steps']} \\")
    print(f"  --lr {config['lr']}")
    
    print("\n" + "=" * 50)
    print("开始训练前请确保:")
    print("1. 已运行 setup_local_env.bat 配置环境")
    print("2. 数据集路径正确: e:\\AIC1\\webfg400_train\\train")
    print("=" * 50)