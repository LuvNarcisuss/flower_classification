# 花卉分类项目

## 项目简介

本项目是一个基于 Ultralytics YOLO26 模型的花卉分类系统，旨在通过深度学习技术对不同种类的花卉进行自动识别和分类。项目利用了 Ultralytics 框架的强大功能，结合了多种注意力机制模块，以提高分类精度和模型性能。此外，项目还包含一个基于Flask的Web应用，提供了用户友好的界面来使用花卉分类模型。

## 项目结构

```
flower_classification/
├── flower_photos/         # 原始花卉数据集
├── flower_photos_split/   # 划分后的训练/验证/测试数据集
├── static/                # Web应用静态资源
│   ├── uploads/           # 上传的图像文件
├── templates/             # Web应用HTML模板
├── ultralytics/           # Ultralytics 框架代码
│   ├── cfg/models/26/Att_yaml/  # 注意力机制模型配置文件
│   ├── nn/Attmodules/     # 注意力机制模块实现
├── weights/               # 预训练模型和训练权重
│   ├── trained_weights/   # 训练后的模型权重
├── performance_reports/   # 性能报告
├── app.py                 # Web应用主脚本
├── stats_load_history.py  # 历史记录和统计分析脚本
├── train_model.py         # 模型训练脚本
├── split_dataset.py       # 数据集划分脚本
├── model_comparison.py    # 模型性能比较脚本
├── train_logs.py          # 训练日志生成脚本
├── flower_classes.csv     # 花卉类别中英文映射
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
- Flask 3.0.0+ (Web应用)
- pandas 2.0.0+ (数据处理)

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

## Web应用

项目包含一个基于Flask的Web应用，提供了用户友好的界面来使用花卉分类模型。

### 功能特点

- **用户认证**：支持用户注册和登录
- **花卉识别**：上传花卉图像进行分类识别
- **历史记录**：保存用户的识别历史
- **统计分析**：提供识别结果的统计分析
- **模型管理**：支持切换不同的训练模型
- **性能展示**：展示模型的性能指标
- **训练过程可视化**：展示模型训练过程的指标

### 如何运行Web应用

1. 确保安装了所有依赖项
2. 运行Web应用：

```bash
python app.py
```

3. 在浏览器中访问 `http://localhost:1104`

### Web应用使用指南

1. **注册/登录**：首先注册一个账户或登录现有账户
2. **上传图像**：在首页上传花卉图像
3. **查看结果**：查看模型的分类结果和置信度
4. **查看历史**：在历史记录页面查看过去的识别结果
5. **导出数据**：将识别历史导出为CSV文件
6. **切换模型**：在性能页面切换不同的模型
7. **查看性能**：查看模型的详细性能指标
8. **查看训练过程**：查看模型的训练过程数据

## 数据库

Web应用使用数据库存储用户信息、识别历史和纠正信息。数据库连接配置在 `stats_load_history.py` 文件中。

## 许可证

本项目基于 AGPL-3.0 许可证，详情请参阅 `LICENSE` 文件。

## 贡献

欢迎对项目进行贡献，包括但不限于：

- 改进模型性能
- 添加新的注意力机制
- 优化训练流程
- 扩展数据集
- 改进Web应用功能
- 添加新的可视化效果
- 优化用户界面

## 参考

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLO26: Next-Gen Real-Time Object Detection](https://docs.ultralytics.com/models/yolo26/)
- [Flask Web Framework](https://flask.palletsprojects.com/)

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目维护者：Ultralytics Team
- 官方文档：https://docs.ultralytics.com
- GitHub 仓库：https://github.com/ultralytics/ultralytics
