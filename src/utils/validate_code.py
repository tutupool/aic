#!/usr/bin/env python3
"""
代码语法验证脚本 - 不依赖外部包
"""

import ast
import os

def validate_python_syntax(file_path):
    """验证Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        ast.parse(source_code)
        print(f"✓ {file_path} 语法正确")
        return True
    except SyntaxError as e:
        print(f"✗ {file_path} 语法错误: {e}")
        return False
    except Exception as e:
        print(f"? {file_path} 读取错误: {e}")
        return False

def check_file_exists(file_path):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print(f"✓ {file_path} 存在")
        return True
    else:
        print(f"✗ {file_path} 不存在")
        return False

def main():
    """主验证函数"""
    print("=" * 50)
    print("WebFG-400 代码库验证")
    print("=" * 50)
    
    files_to_check = [
        'data_loader.py',
        'model.py', 
        'train.py',
        'kaggle_train.py',
        'requirements.txt',
        'kernel-metadata.json',
        'README.md',
        'test_setup.py'
    ]
    
    print("\n检查文件存在性:")
    existence_results = []
    for file in files_to_check:
        existence_results.append(check_file_exists(file))
    
    print("\n验证Python语法:")
    syntax_results = []
    for file in files_to_check:
        if file.endswith('.py'):
            syntax_results.append(validate_python_syntax(file))
    
    print("\n" + "=" * 50)
    print("验证结果汇总:")
    print("=" * 50)
    
    all_files_exist = all(existence_results)
    all_syntax_ok = all(syntax_results) if syntax_results else True
    
    if all_files_exist and all_syntax_ok:
        print("🎉 所有文件存在且语法正确！")
        print("\n代码库结构完整，可以上传到Kaggle。")
        print("\n上传步骤:")
        print("1. 将 webfg400_train.zip 上传为Kaggle数据集")
        print("2. 创建新内核，上传本代码库所有文件")
        print("3. 在Kaggle环境中安装依赖: pip install -r requirements.txt")
        print("4. 运行: python kaggle_train.py")
        return True
    else:
        print("❌ 验证失败，请检查缺失的文件或语法错误。")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)