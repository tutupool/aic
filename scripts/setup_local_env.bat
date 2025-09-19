@echo off
echo ========================================
echo    WebFG-400 本地训练环境配置脚本
echo ========================================
echo.

echo 1. 创建Anaconda虚拟环境
echo.
conda create -n webfg400 python=3.9 -y

echo.
echo 2. 激活虚拟环境
call conda activate webfg400

echo.
echo 3. 安装PyTorch (CUDA 11.8版本，适配4060显卡)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 4. 安装其他依赖包
pip install matplotlib tqdm opencv-python pandas scikit-learn --trusted-host pypi.tuna.tsinghua.edu.cn -i http://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo 5. 验证GPU是否可用
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU型号: {torch.cuda.get_device_name(0)}'); print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

echo.
echo ========================================
echo    环境配置完成！
echo    使用以下命令激活环境：
echo    conda activate webfg400
echo    然后运行：python ../src/training/local_train.py
echo ========================================

pause