from multiprocessing.dummy import freeze_support
from ultralytics import YOLO
import torch
from pathlib import Path
from datetime import datetime
from train_logs import train_log

# 禁用确定性算法
# torch.use_deterministic_algorithms(False)
# 忽略特定警告
import warnings
warnings.filterwarnings("ignore", message="adaptive_max_pool2d_backward_cuda does not have a deterministic implementation")

def train_model(model_name):
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent

    if model_name.endswith('.yaml'):
        # 加载yolo模型
        model = YOLO('ultralytics/cfg/models/26/Att_yaml/' + model_name)
        # 分解模型名称
        parts = model_name.replace('.yaml', '').split('-')
        train_name = f"{parts[0]}_{parts[1]}_flower_cls"
    elif model_name.endswith('.pt'):
        model = YOLO('weights/' + model_name)
        # 分解模型名称
        parts = model_name.replace('.pt', '').split('-')
        train_name = f"{parts[0]}_flower_cls"
    else:
        print("请输入正确的模型")
        return

    # 指定具体GPU（确保GPU存在）
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.cuda.get_device_name(0)  # 使用第一个GPU
        print("使用 GPU 训练")
    else:
        device = 'cpu'
        print("使用 CPU 训练")

    """# 检查原数据集
    dataset_path = current_dir / 'flower_photos'
    classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    classes.sort()
    print(f"\n数据集中的类别: {classes}")

    dataset_info = []
    total_images = 0
    for cls in classes:
        cls_path = dataset_path / cls
        count = len([f for f in cls_path.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']])
        total_images += count
        dataset_info.append(f"{cls}: {count} 张图片")
        print(f"{cls}: {count} 张图片")
    print(f"总计: {total_images} 张图片")"""

    # 检查数据集结构（使用已划分的数据集）
    dataset_path = current_dir / 'flower_photos_split'

    # 检查 train、val、test 子目录
    splits_info = []
    all_classes = set()
    dataset_info = []

    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if split_dir.exists():
            classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
            classes.sort()
            all_classes.update(classes)

            print(f"\n{split} 目录：")
            total_in_split = 0
            for cls in classes:
                cls_path = split_dir / cls
                count = len([f for f in cls_path.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']])
                total_in_split += count
                print(f"  - {cls}: {count} 张图片")
                dataset_info.append(f"{split}/{cls}: {count} 张图片")
            print(f"  {split} 总计：{total_in_split} 张图片")
            splits_info.append((split, total_in_split))
        else:
            print(f"\n{split} 目录不存在")

    classes = sorted(list(all_classes))
    print(f"\n数据集中的类别：{classes}")

    # 计算总图片数
    total_images = sum(info[1] for info in splits_info)
    print(f"\n所有数据集总计：{total_images} 张图片")

    # 训练参数
    train_params = {
        'data': str(dataset_path),        # 数据集目录，分类任务会自动读取子目录作为类别
        'epochs': 100,                    # 训练轮数
        'batch': 16,                      # 批次大小
        'imgsz': 320,                     # 图像大小
        'device': device,                 # 使用GPU训练，更通用
        'amp': True,                      # 开启混合精度训练
        'project': 'flower_detection',    # 项目名称
        'name': train_name,               # 训练结果名称
        'exist_ok': True,                 # 如果项目已存在，覆盖它
        'task': 'classify',               # 指定任务类型为分类
        'classes': classes                # 显式指定类别顺序，确保与预测时一致
    }
    
    print("\n训练参数:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")
    
    # 开始训练时间
    start_time = datetime.now()
    print(f"\n开始训练时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 训练模型
    print("\n开始训练模型...")
    results = model.train(**train_params)
    
    # 训练结束时间
    end_time = datetime.now()
    print(f"\n训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练总时长: {end_time - start_time}")
    
    # 验证模型
    print("\n开始验证模型...")
    metrics = model.val(device=device)  # 使用与训练相同的设备验证
    print(f"验证结果: {metrics}")
    
    # 获取模型信息
    model_info = {}
    model_layers_info = []
    try:
        # 获取模型结构信息
        model_info['model_type'] = type(model).__name__
        model_info['model_config'] = 'ultralytics/cfg/models/26/Att_yaml/' + model_name if model_name.endswith('.yaml') else 'ultralytics/cfg/models/26/yolo26.yaml'
        # 尝试获取模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        model_info['total_params'] = f"{total_params:,}"
        # 尝试获取可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_info['trainable_params'] = f"{trainable_params:,}"
        
        # 获取模型层信息
        if hasattr(model, 'model') and hasattr(model.model, 'modules'):
            layers = list(model.model.modules())
            for i, layer in enumerate(layers):
                if hasattr(layer, '__class__'):
                    layer_info = {
                        'index': i,
                        'name': layer.__class__.__name__,
                        'module': str(layer.__class__.__module__),
                        'params': sum(p.numel() for p in layer.parameters()) if hasattr(layer, 'parameters') else 0
                    }
                    # 尝试获取层的参数
                    if hasattr(layer, '__dict__'):
                        # 只保存一些关键参数
                        if hasattr(layer, 'in_channels'):
                            layer_info['in_channels'] = layer.in_channels
                        if hasattr(layer, 'out_channels'):
                            layer_info['out_channels'] = layer.out_channels
                        if hasattr(layer, 'kernel_size'):
                            layer_info['kernel_size'] = layer.kernel_size
                        if hasattr(layer, 'stride'):
                            layer_info['stride'] = layer.stride
                    model_layers_info.append(layer_info)
    except Exception as e:
        model_info['error'] = str(e)
    
    # 保存模型
    model_name = train_name + '_model.pt'
    model_path = current_dir / 'weights/trained_weights' / model_name
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    print(f"模型大小: {model_path.stat().st_size / 1024 / 1024:.2f} MB\n")
    
    # 生成训练日志文件
    log_name = f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # 合并训练参数和模型信息
    full_train_info = {
        **train_params,
        'model_info': model_info,
        'model_layers_info': model_layers_info
    }
    # 生成训练日志文件
    train_log(current_dir, log_name, classes, dataset_info, total_images, full_train_info,
              start_time, end_time, metrics, model_path)

    print("\n模型训练完成！\n")
    
if __name__ == '__main__':
    freeze_support()  # 解决Windows下多进程问题

    att_yaml_dir = Path(__file__).parent / 'ultralytics/cfg/models/26/Att_yaml'

    train_model('yolo26n-SE-cls.yaml')
    train_model('yolo26n-CBAM-cls.yaml')

    # # 遍历目录中的所有分类模型的 yaml 文件
    # yaml_files = [f.name for f in att_yaml_dir.iterdir()
    #               if f.suffix.lower() == '.yaml'
    #               and f.name.endswith('-cls.yaml')]
    # print(f"找到 {len(yaml_files)} 个 YAML 配置文件:\n")
    #
    # # 开始遍历训练每个模型
    # for model_name in yaml_files:
    #     if model_name.startswith('yolo26-'):
    #         model_name = model_name.replace('yolo26-', 'yolo26n-', 1)
    #     print(model_name)
    #     # 使用配置好的依赖训练模型
    #     train_model(model_name)