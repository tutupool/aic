#!/usr/bin/env python3
"""
训练状态监视脚本
用于监视Webinat5000训练进度
"""

import os
import time
import glob

def monitor_training():
    """监视训练状态"""
    model_dir = "outputs/models_webinat5000"
    
    print("=== Webinat5000训练状态监视 ===")
    
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        print("可能训练尚未开始或目录路径不正确")
        return
    
    # 获取所有模型文件
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    
    print(f"📁 模型目录: {model_dir}")
    print(f"📊 模型文件数量: {len(model_files)}")
    
    if model_files:
        # 按修改时间排序
        model_files.sort(key=os.path.getmtime)
        
        print("\n📋 模型文件列表:")
        for i, file_path in enumerate(model_files[-5:], 1):  # 显示最后5个文件
            file_name = os.path.basename(file_path)
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file_path)))
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {i}. {file_name} ({size_mb:.1f} MB) - {mtime}")
        
        if len(model_files) > 5:
            print(f"  ... 还有 {len(model_files) - 5} 个更早的文件")
    else:
        print("\n⚠️  目录中没有找到模型文件")
        print("可能训练刚刚开始或遇到问题")
    
    # 检查是否有日志文件
    log_dir = "outputs/logs"
    if os.path.exists(log_dir):
        log_files = glob.glob(os.path.join(log_dir, "*.txt"))
        if log_files:
            print(f"\n📝 日志文件数量: {len(log_files)}")

if __name__ == "__main__":
    monitor_training()