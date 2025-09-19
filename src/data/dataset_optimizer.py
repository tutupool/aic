#!/usr/bin/env python3
"""
数据集优化工具 - 减小数据集大小，便于上传
"""

import os
import zipfile
import argparse
from pathlib import Path
import shutil

def analyze_dataset_size(dataset_path):
    """分析数据集大小"""
    total_size = 0
    image_count = 0
    class_count = 0
    
    print("分析数据集大小...")
    
    for class_dir in Path(dataset_path).iterdir():
        if class_dir.is_dir():
            class_count += 1
            for img_file in class_dir.glob('*.jpg'):
                total_size += img_file.stat().st_size
                image_count += 1
            for img_file in class_dir.glob('*.png'):
                total_size += img_file.stat().st_size
                image_count += 1
            for img_file in class_dir.glob('*.jpeg'):
                total_size += img_file.stat().st_size
                image_count += 1
    
    print(f"数据集路径: {dataset_path}")
    print(f"类别数量: {class_count}")
    print(f"图像总数: {image_count}")
    print(f"总大小: {total_size / (1024**3):.2f} GB")
    print(f"平均每张图像: {total_size / image_count / 1024:.1f} KB")
    
    return total_size, image_count, class_count

def create_optimized_dataset(source_path, target_path, max_size_gb=2):
    """创建优化后的数据集"""
    max_size_bytes = max_size_gb * 1024**3
    
    print(f"创建优化数据集，目标大小: {max_size_gb} GB")
    
    # 创建目标目录
    os.makedirs(target_path, exist_ok=True)
    
    total_copied = 0
    class_count = 0
    image_count = 0
    
    for class_dir in Path(source_path).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            target_class_dir = os.path.join(target_path, class_name)
            os.makedirs(target_class_dir, exist_ok=True)
            
            class_count += 1
            class_images = 0
            
            # 获取所有图像文件
            image_files = []
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                image_files.extend(list(class_dir.glob(ext)))
            
            # 每个类别最多复制50张图像
            max_per_class = min(50, len(image_files))
            
            for img_file in image_files[:max_per_class]:
                if total_copied + img_file.stat().st_size > max_size_bytes:
                    break
                    
                shutil.copy2(img_file, target_class_dir)
                total_copied += img_file.stat().st_size
                image_count += 1
                class_images += 1
            
            print(f"类别 {class_name}: 复制了 {class_images} 张图像")
    
    print(f"优化完成!")
    print(f"总图像数: {image_count}")
    print(f"总大小: {total_copied / (1024**3):.2f} GB")
    
    return total_copied, image_count, class_count

def compress_dataset(dataset_path, output_zip):
    """压缩数据集"""
    print(f"压缩数据集到 {output_zip}...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dataset_path)
                zipf.write(file_path, arcname)
    
    zip_size = os.path.getsize(output_zip) / (1024**3)
    print(f"压缩完成! 压缩包大小: {zip_size:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description='数据集优化工具')
    parser.add_argument('--source', type=str, default='e:\\AIC1\\webfg400_train\\train',
                       help='源数据集路径')
    parser.add_argument('--target', type=str, default='e:\\AIC1\\webfg400_train_optimized',
                       help='目标数据集路径')
    parser.add_argument('--max_size', type=float, default=2.0,
                       help='最大数据集大小(GB)')
    parser.add_argument('--compress', action='store_true',
                       help='是否压缩数据集')
    
    args = parser.parse_args()
    
    # 分析原始数据集
    print("=" * 50)
    print("原始数据集分析:")
    analyze_dataset_size(args.source)
    
    print("\n" + "=" * 50)
    print("创建优化数据集:")
    
    # 创建优化数据集
    create_optimized_dataset(args.source, args.target, args.max_size)
    
    # 分析优化后的数据集
    print("\n" + "=" * 50)
    print("优化后数据集分析:")
    analyze_dataset_size(args.target)
    
    # 压缩数据集
    if args.compress:
        print("\n" + "=" * 50)
        output_zip = args.target + '.zip'
        compress_dataset(args.target, output_zip)
        
        # 分析压缩包
        zip_size = os.path.getsize(output_zip) / (1024**3)
        print(f"最终压缩包大小: {zip_size:.2f} GB")
        print(f"可以上传到Kaggle!")

if __name__ == "__main__":
    main()