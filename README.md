# 花卉分类项目

## 项目简介

本项目是一个基于 Ultralytics YOLO26 模型的花卉分类系统，旨在通过深度学习技术对不同种类的花卉进行自动识别和分类。项目利用了 Ultralytics 框架的强大功能，结合了多种注意力机制模块，以提高分类精度和模型性能。

## 项目结构

```
flower_classification/
├── flower_photos/         # 原始花卉数据集
├── flower_photos_split/   # 划分后的训练/验证/测试数据集
├── ultralytics/           # Ultralytics 框架代码
│   ├── cfg/models/26/Att_yaml/  # 注意力机制模型配置文件
│   ├── nn/Attmodules/     # 注意力机制模块实现
├── weights/               # 预训练模型和训练权重
│   ├── trained_weights/   # 训练后的模型权重
├── performance_reports/   # 性能报告
├── train_model.py         # 模型训练脚本
├── split_dataset.py       # 数据集划分脚本
├── model_comparison.py    # 模型性能比较脚本
├── train_logs.py          # 训练日志生成脚本
├── pyproject.toml         # 项目配置和依赖
├── requirements.txt       # 依赖项列表
└── README.md              # 项目说明文档
```

## 依赖项

项目基于以下主要依赖项：

- Python 3.8+
- PyTorch 1.8.0+
- NumPy 1.23.0+
- OpenCV 4.6.0+
- Matplotlib 3.3.0+
- Ultralytics 框架

详细依赖项请参考 `pyproject.toml` 文件。

## 数据集

项目使用花卉数据集进行训练和测试，数据集包含多个类别的花卉图像。数据集结构如下：

- `flower_photos/`：原始数据集，包含各类花卉的图像
- `flower_photos_split/`：划分后的数据集，包含 train、val 和 test 子目录

数据集已通过 `split_dataset.py` 脚本进行了划分，确保训练、验证和测试数据的合理分布。

## 训练方法

### 1. 准备数据集

确保 `flower_photos_split` 目录存在并包含划分好的数据集。如果需要重新划分数据集，可以运行：

```bash
python split_dataset.py
```

### 2. 选择模型

项目支持多种带有注意力机制的 YOLO26 模型，配置文件位于 `ultralytics/cfg/models/26/Att_yaml/` 目录，例如：

- `yolo26n-SE-cls.yaml`：带有 SE 注意力机制的 YOLO26n 分类模型
- `yolo26n-CBAM-cls.yaml`：带有 CBAM 注意力机制的 YOLO26n 分类模型
- 等其他模型

### 3. 训练模型

运行训练脚本开始训练模型：

```bash
python train_model.py
```

训练参数可在 `train_model.py` 文件中修改，包括：
- 训练轮数 (epochs)
- 批次大小 (batch)
- 图像大小 (imgsz)
- 设备选择 (device)

### 4. 模型评估

训练完成后，模型会自动进行验证，并生成详细的训练日志和性能报告。

## 模型比较

项目包含 `model_comparison.py` 脚本，用于比较不同模型的性能指标，包括准确率、损失值、推理时间等。

## 注意力机制模块

项目实现了多种注意力机制模块，位于 `ultralytics/nn/Attmodules/` 目录：

- SE (Squeeze-and-Excitation)
- CBAM (Convolutional Block Attention Module)
- 等其他注意力机制

这些模块可以通过修改模型配置文件来使用。

## 使用预训练模型

项目提供了预训练的 YOLO26 模型权重，位于 `weights/` 目录：

- `yolo26n.pt`：基础 YOLO26n 模型
- `yolo26s.pt`：基础 YOLO26s 模型
- `yolo26n-cls.pt`：YOLO26n 分类模型
- `yolo26s-cls.pt`：YOLO26s 分类模型

## 性能报告

训练完成后，详细的性能报告将保存在 `performance_reports/` 目录中，包括：

- 训练和验证准确率
- 损失曲线
- 模型参数量
- 推理时间
- 各类别分类性能
