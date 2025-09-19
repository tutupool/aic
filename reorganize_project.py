#!/usr/bin/env python3
"""
项目结构重组脚本
自动将杂乱的文件按功能分类到标准目录结构中
"""

import os
import shutil
from pathlib import Path

def create_directories():
    """创建所有需要的目录"""
    directories = [
        'src/data', 'src/models', 'src/training', 'src/inference', 'src/utils', 'src/config',
        'scripts',
        'data/raw', 'data/processed', 'data/external',
        'outputs/models', 'outputs/logs', 'outputs/predictions',
        'docs/competition',
        'backups'
    ]
    
    print("创建目录结构...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ 创建: {directory}")

def move_files():
    """移动文件到对应的目录"""
    file_mapping = {
        # 数据文件
        'webfg400_train': 'data/raw/',
        'webinat5000_train.zip': 'data/external/',
        
        # 源代码 - 数据处理
        'data_loader.py': 'src/data/',
        'dataset_optimizer.py': 'src/data/',
        
        # 源代码 - 模型
        'model.py': 'src/models/',
        
        # 源代码 - 训练
        'train.py': 'src/training/',
        'local_train.py': 'src/training/',
        'train_config_4060.py': 'src/training/',
        
        # 源代码 - 推理
        'predict.py': 'src/inference/',
        'kaggle_predict.py': 'src/inference/',
        
        # 源代码 - 工具
        'validate_code.py': 'src/utils/',
        'test_setup.py': 'src/utils/',
        
        # 源代码 - 配置
        'kernel-metadata.json': 'src/config/',
        
        # 脚本
        'setup_local_env.bat': 'scripts/',
        'kaggle_train.py': 'scripts/',
        
        # 文档
        'README.md': 'docs/',
        'LOCAL_TRAINING_GUIDE.md': 'docs/',
        'PROJECT_REORGANIZATION_PLAN.md': 'docs/',
        '【2025】AIC挑战赛赛题规则_20250717145839_29-34(1).pdf': 'docs/competition/',
        
        # 备份文件
        'AIC1.zip': 'backups/',
        'webfg400_train.zip': 'backups/',
    }
    
    print("\n移动文件...")
    moved_count = 0
    skipped_count = 0
    
    for source_file, target_dir in file_mapping.items():
        if os.path.exists(source_file):
            target_path = os.path.join(target_dir, source_file)
            
            # 如果是目录，使用shutil.move
            if os.path.isdir(source_file):
                shutil.move(source_file, target_dir)
                print(f"  ✓ 移动目录: {source_file} -> {target_dir}")
            else:
                shutil.move(source_file, target_path)
                print(f"  ✓ 移动文件: {source_file} -> {target_path}")
            moved_count += 1
        else:
            print(f"  ⚠ 跳过: {source_file} (不存在)")
            skipped_count += 1
    
    print(f"\n移动完成: {moved_count} 个文件已移动, {skipped_count} 个文件跳过")

def update_file_references():
    """更新文件中的路径引用"""
    print("\n更新文件中的路径引用...")
    
    # 需要更新的文件及其路径映射
    update_files = {
        'src/training/local_train.py': {
            r"e:\\AIC1\\webfg400_train\\train": "data/raw/webfg400_train/train",
            r"\./output": "outputs/models"
        },
        'src/training/train.py': {
            r"e:\\AIC1\\webfg400_train\\train": "data/raw/webfg400_train/train",
            r"\./output": "outputs/models"
        },
        'src/inference/predict.py': {
            r"\./output": "outputs/models"
        },
        'scripts/setup_local_env.bat': {
            r"python local_train.py": "python ../src/training/local_train.py"
        }
    }
    
    updated_count = 0
    
    for file_path, replacements in update_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 执行替换
                original_content = content
                for old_path, new_path in replacements.items():
                    content = content.replace(old_path, new_path)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✓ 更新: {file_path}")
                    updated_count += 1
                else:
                    print(f"  ⚠ 无需更新: {file_path}")
                    
            except Exception as e:
                print(f"  ❌ 更新失败: {file_path} - {e}")
        else:
            print(f"  ⚠ 文件不存在: {file_path}")
    
    print(f"路径引用更新完成: {updated_count} 个文件已更新")

def create_new_requirements():
    """创建新的requirements.txt在根目录"""
    print("\n创建新的requirements.txt...")
    
    requirements_content = """torch>=2.0.0
 torchvision>=0.15.0
 torchaudio>=2.0.0
 matplotlib>=3.7.0
 tqdm>=4.65.0
 opencv-python>=4.8.0
 pandas>=2.0.0
 scikit-learn>=1.3.0
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    
    print("  ✓ 创建: requirements.txt")

def verify_structure():
    """验证重组后的结构"""
    print("\n" + "="*60)
    print("验证项目结构...")
    print("="*60)
    
    expected_files = [
        'requirements.txt',
        'src/data/data_loader.py',
        'src/models/model.py', 
        'src/training/local_train.py',
        'src/inference/predict.py',
        'scripts/setup_local_env.bat',
        'data/raw/webfg400_train',
        'docs/README.md'
    ]
    
    all_exists = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"  ✓ 存在: {file_path}")
        else:
            print(f"  ❌ 缺失: {file_path}")
            all_exists = False
    
    if all_exists:
        print("\n🎉 项目重组成功！结构验证通过！")
    else:
        print("\n⚠️  项目重组完成，但有些文件缺失")
    
    return all_exists

def main():
    """主函数"""
    print("="*60)
    print("🤖 WebFG-400 项目结构重组工具")
    print("="*60)
    
    # 备份警告
    print("⚠️  警告: 此操作将移动文件，建议先备份项目！")
    response = input("是否继续? (y/n): ")
    
    if response.lower() != 'y':
        print("操作取消")
        return
    
    # 执行重组步骤
    create_directories()
    move_files()
    update_file_references()
    create_new_requirements()
    success = verify_structure()
    
    print("\n" + "="*60)
    if success:
        print("✅ 项目重组完成！")
        print("\n下一步操作:")
        print("1. 测试训练: python src/training/local_train.py --help")
        print("2. 测试预测: python src/inference/predict.py --help")
        print("3. 配置环境: scripts/setup_local_env.bat")
    else:
        print("⚠️  重组完成，但需要手动检查缺失文件")
    print("="*60)

if __name__ == "__main__":
    main()