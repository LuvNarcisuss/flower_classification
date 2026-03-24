from datetime import datetime
import torch
from pathlib import Path

def train_log(current_dir, log_name, classes, dataset_info, total_images, train_params,
              start_time, end_time, metrics, model_path):
    # 生成训练日志文件
    # 创建日志目录
    logs_dir = current_dir / 'train_logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件路径

    log_file = logs_dir / f'{log_name}.txt'

    # 写入日志
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("花朵识别模型训练日志\n")
        f.write("=" * 80 + "\n\n")

        # 训练基本信息
        f.write("1. 训练基本信息\n")
        f.write("-" * 40 + "\n")
        f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练时长: {end_time - start_time}\n")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
        else:
            device_name = 'CPU'
        f.write(f"训练设备: {device_name}\n\n")

        # 数据集信息
        f.write("2. 数据集信息\n")
        f.write("-" * 40 + "\n")
        f.write(f"类别数量: {len(classes)}\n")
        f.write(f"类别列表: {', '.join(classes)}\n")
        f.write(f"总图像数: {total_images}\n\n")
        f.write("类别分布:\n")
        for info in dataset_info:
            f.write(f"  {info}\n")
        f.write("\n")

        # 训练参数
        f.write("3. 训练参数\n")
        f.write("-" * 40 + "\n")
        # 分离模型信息和其他训练参数
        model_info = train_params.pop('model_info', {})
        for key, value in train_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # 模型信息
        if model_info:
            f.write("4. 模型信息\n")
            f.write("-" * 40 + "\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # 模型层信息
        model_layers_info = train_params.pop('model_layers_info', [])
        if model_layers_info:
            f.write("5. 模型层信息\n")
            f.write("-" * 40 + "\n")
            # 写入表头
            f.write(f"{'index':>6} {'params':>10} {'module':<40} {'arguments':<40}\n")
            f.write("-" * 120 + "\n")
            # 写入层信息
            for layer in model_layers_info:
                # 构建参数字符串
                args = []
                if 'in_channels' in layer and 'out_channels' in layer:
                    args.extend([layer['in_channels'], layer['out_channels']])
                    if 'kernel_size' in layer:
                        args.append(layer['kernel_size'])
                    if 'stride' in layer:
                        args.append(layer['stride'])
                args_str = str(args) if args else "[]"
                
                # 写入层信息
                module_name = layer['module'].split('.')[-1] + '.' + layer['name']
                f.write(f"{layer['index']:>6} {layer['params']:>10} {module_name:<40} {args_str:<40}\n")
            f.write("\n")
            
        # 训练结果
        f.write("6. 训练结果\n")
        f.write("-" * 40 + "\n")
        f.write(f"验证结果: {metrics}\n\n")
        
        # 模型保存信息
        f.write("7. 模型保存信息\n")
        f.write("-" * 40 + "\n")
        if model_path.exists():
            f.write(f"模型保存路径: {model_path}\n")
            f.write(f"模型大小: {model_path.stat().st_size / 1024 / 1024:.2f} MB\n")
        else:
            f.write("模型文件未找到\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("训练日志生成完成\n")
        f.write("=" * 80 + "\n")

    print(f"训练日志已生成到: {log_file}")
    print(f"日志文件大小: {log_file.stat().st_size / 1024:.2f} KB")

if __name__ == '__main__':
    current = Path(__file__).parent
    log_name = f'test_log'
    train_log(current, log_name, [], [], 0, {}, datetime.now(), datetime.now(), {}, current / 'model.pt')
