from pathlib import Path

weights_dir = Path('weights/trained_weights')
model_paths = [str(p) for p in weights_dir.glob('*.pt')]

if not model_paths:
    print(f"警告：在 {weights_dir} 目录下未找到任何 .pt 模型文件")

print(f"找到 {len(model_paths)} 个模型文件:")

print(model_paths)