# WebFG-400 Kaggle 训练代码

本代码库提供了在Kaggle平台上训练WebFG-400细粒度图像分类模型的完整解决方案。

## 数据集

WebFG-400 是一个包含400个细粒度类别的图像分类数据集，每个类别包含约600张从网络收集的图像。

## 项目结构

```
AIC1/
├── data_loader.py      # 数据加载和预处理
├── model.py           # 模型定义
├── train.py          # 训练脚本
├── kaggle_train.py   # Kaggle专用训练入口
├── requirements.txt  # 依赖包
├── kernel-metadata.json  # Kaggle内核配置
└── README.md         # 说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 2.0.0+
- TorchVision 0.15.0+
- CUDA 11.7+ (GPU训练)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 在Kaggle上运行

### 1. 上传数据集

将WebFG-400训练数据上传到Kaggle数据集：
- 数据集名称: `webfg400-train`
- 数据路径: `/kaggle/input/webfg400-train/train`

### 2. 创建Kaggle内核

1. 上传本代码库所有文件到Kaggle
2. 确保内核设置中启用GPU加速
3. 设置互联网访问权限（用于下载预训练权重）

### 3. 运行训练

内核会自动执行 `kaggle_train.py`，或者可以手动运行：

```bash
python kaggle_train.py \
    --data_dir /kaggle/input/webfg400-train/train \
    --output_dir /kaggle/working \
    --batch_size 32 \
    --epochs 20 \
    --lr 1e-4
```

## 训练参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `/kaggle/input/webfg400-train/train` | 训练数据目录 |
| `--output_dir` | `/kaggle/working` | 输出文件目录 |
| `--batch_size` | 32 | 训练批次大小 |
| `--num_workers` | 2 | 数据加载线程数 |
| `--img_size` | 224 | 输入图像尺寸 |
| `--lr` | 1e-4 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--epochs` | 20 | 训练轮数 |
| `--pretrained` | True | 使用预训练权重 |

## 模型架构

- 基础模型: ResNet-50 (ImageNet预训练)
- 分类头: 2层全连接网络 + Dropout
- 输入尺寸: 224×224×3
- 输出类别: 400

## 数据增强

训练时使用以下数据增强：
- 随机水平翻转
- 随机亮度对比度调整
- 标准化 (ImageNet均值标准差)

## 输出文件

训练完成后会在输出目录生成：
- `best_model.pth` - 最佳验证准确率的模型权重
- `final_model.pth` - 最终训练完成的模型权重
- `training_history.png` - 训练过程可视化
- `training_config.txt` - 训练配置参数

### 生成预测结果文件

训练完成后，使用以下命令生成 `pred_results_web400.csv` 预测结果文件：

```bash
# 在Kaggle上运行预测
python kaggle_predict.py

# 或者使用通用预测脚本
python predict.py --model_path /kaggle/working/best_model.pth --data_dir /kaggle/input/webfg400-train/webfg400_train/train --output_csv pred_results_web400.csv
```

预测结果文件包含以下列：
- `image_id` - 图像ID（不含扩展名）
- `image_filename` - 图像文件名
- `true_label` - 真实标签编号
- `predicted_label` - 预测标签编号
- `confidence` - 预测置信度
- `true_class` - 真实类别名称
- `predicted_class` - 预测类别名称
- `directory` - 图像所在目录

## 性能指标

- 使用准确率(Accuracy)作为主要评估指标
- 支持混淆矩阵和分类报告
- 自动保存最佳模型

## 注意事项

1. 确保Kaggle内核有足够的GPU内存（建议P100或V100）
2. 训练时间约2-4小时（20个epoch）
3. 验证集自动从训练集划分（80%训练，20%验证）
4. 模型会自动使用混合精度训练加速

## 自定义修改

- 修改 `model.py` 更换模型架构
- 调整 `data_loader.py` 中的数据增强策略
- 在 `train.py` 中修改训练逻辑和评估指标

## 许可证

本项目基于MIT许可证开源。

## 支持

如有问题请在Kaggle讨论区提问或提交GitHub Issue。