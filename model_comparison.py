from ultralytics import YOLO
import torch
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda.memory')

class ModelPerformanceAnalyzer:
    def __init__(self, model_paths, test_data_path, device=None):
        """初始化模型性能分析器
        
        Args:
            model_paths: 模型文件路径列表
            test_data_path: 测试数据路径
            device: 运行设备
        """
        self.model_paths = model_paths
        self.test_data_path = test_data_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.performance_metrics = {}
        self.model_names = {}
        
        # 加载模型
        for model_path in model_paths:
            model_name = Path(model_path).stem.replace(
                '_model', '', 1)
            self.model_names[model_path] = model_name
            self.models[model_name] = YOLO(model_path)
            self.models[model_name].to(self.device)
            print(f"已加载模型: {model_name}")
    
    def analyze_performance(self):
        """分析模型性能"""
        for model_path, model_name in self.model_names.items():
            print(f"\n分析模型: {model_name}")
            metrics = {}
            
            # 1. 计算模型参数量
            model = self.models[model_name]
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            metrics['total_params'] = total_params
            metrics['trainable_params'] = trainable_params
            metrics['params_mb'] = total_params * 4 / (1024 * 1024)  # 粗略计算模型大小
            
            # 2. 测试推理速度
            speed_metrics = self._test_inference_speed(model)
            metrics.update(speed_metrics)
            
            # 3. 测试模型准确率
            accuracy_metrics = self._test_accuracy(model)
            metrics.update(accuracy_metrics)
            
            # 4. 计算内存使用
            memory_metrics = self._test_memory_usage(model)
            metrics.update(memory_metrics)
            
            self.performance_metrics[model_name] = metrics
            print(f"模型 {model_name} 分析完成")
        
    def _test_inference_speed(self, model, iterations=100, batch_size=1):
        """测试模型推理速度"""
        # 创建随机输入
        input_tensor = torch.randn(batch_size, 3, 320, 320).to(self.device)
        
        # 预热
        for _ in range(10):
            _ = model(input_tensor)
        
        # 测试推理时间
        start_time = time.time()
        for _ in range(iterations):
            _ = model(input_tensor)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        fps = iterations / total_time
        
        return {
            'inference_time_ms': avg_time * 1000,
            'fps': fps
        }

    def _test_accuracy(self, model):
        """测试模型准确率"""
        # 初始化所有指标为0
        metrics = {
            'accuracy_top1': 0,
            'accuracy_top5': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        
        try:
            # 首先尝试使用YOLO的val()方法获取基本指标
            results = model.val(data=self.test_data_path, device=self.device)
            metrics['accuracy_top1'] = results.top1
            metrics['accuracy_top5'] = results.top5
            
            print(f"模型准确率: Top-1={results.top1:.4f}, Top-5={results.top5:.4f}")
            
            # 尝试手动计算其他指标
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                import os
                import glob
                
                # 获取测试数据
                test_dir = os.path.join(self.test_data_path, 'val')
                if not os.path.exists(test_dir):
                    test_dir = os.path.join(self.test_data_path, 'test')
                
                if os.path.exists(test_dir):
                    y_true = []
                    y_pred = []
                    
                    # 获取所有类别
                    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
                    if not classes:
                        print("测试目录中没有类别文件夹")
                        return metrics
                    
                    class_to_idx = {cls: i for i, cls in enumerate(classes)}
                    
                    # 遍历测试数据
                    for cls in classes:
                        cls_dir = os.path.join(test_dir, cls)
                        img_paths = glob.glob(os.path.join(cls_dir, '*.*'))
                        
                        for img_path in img_paths:
                            try:
                                # 预测
                                result = model.predict(img_path, device=self.device, verbose=False)[0]
                                # 获取预测类别
                                pred_idx = result.probs.top1
                                if pred_idx < len(classes):
                                    pred_cls = classes[pred_idx]
                                    # 记录真实标签和预测标签
                                    y_true.append(class_to_idx[cls])
                                    y_pred.append(class_to_idx[pred_cls])
                            except Exception as pred_error:
                                print(f"预测图像时出错: {pred_error}")
                                continue
                    
                    # 计算指标
                    if y_true and y_pred:
                        metrics['accuracy'] = accuracy_score(y_true, y_pred)
                        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                        print(f"手动计算的指标: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
                    else:
                        print("没有有效的预测数据，无法计算指标")
                else:
                    print("测试目录不存在")
            except ImportError:
                print("缺少scikit-learn库，无法计算详细指标")
            except Exception as calc_error:
                print(f"计算指标时出错: {calc_error}")
            
        except Exception as e:
            print(f"测试准确率时出错: {e}")
        
        return metrics
    
    def _test_memory_usage(self, model):
        """测试模型内存使用"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            
            # 创建输入
            input_tensor = torch.randn(1, 3, 320, 320).to(self.device)
            
            # 前向传播
            _ = model(input_tensor)
            
            memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.empty_cache()
        else:
            memory_allocated = 0
        
        return {
            'memory_usage_mb': memory_allocated
        }
    
    def generate_report(self, output_dir=None):
        """生成性能报告，更新或添加模型数据"""
        if output_dir is None:
            output_dir = Path(__file__).parent / 'performance_reports'
        output_dir.mkdir(exist_ok=True)
        
        report_file = output_dir / '性能报告.csv'
        
        # 检查文件是否存在
        file_exists = report_file.exists()
        
        # 读取现有数据
        existing_data = {}
        if file_exists:
            import csv
            with open(report_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model_name = row['模型名称']
                    existing_data[model_name] = row
        
        # 更新数据
        for model_name, metrics in self.performance_metrics.items():
            params_million = metrics['total_params'] / 1e6
            existing_data[model_name] = {
                '模型名称': model_name,
                '参数量 (百万)': f"{params_million:.2f}",
                '模型大小 (MB)': f"{metrics['params_mb']:.2f}",
                '推理时间 (ms)': f"{metrics['inference_time_ms']:.2f}",
                'FPS': f"{metrics['fps']:.2f}",
                'Top-1 准确率': f"{metrics['accuracy_top1']:.4f}",
                'Top-5 准确率': f"{metrics['accuracy_top5']:.4f}",
                '准确率': f"{metrics['accuracy']:.4f}",
                '精确率': f"{metrics['precision']:.4f}",
                '召回率': f"{metrics['recall']:.4f}",
                'F1 值': f"{metrics['f1']:.4f}",
                '内存使用 (MB)': f"{metrics['memory_usage_mb']:.2f}",
                '分析时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # 写入CSV文件
        import csv
        fieldnames = ['模型名称', '参数量 (百万)', '模型大小 (MB)', '推理时间 (ms)', 'FPS', 
                     'Top-1 准确率', 'Top-5 准确率', '准确率', '精确率', '召回率', 'F1 值', '内存使用 (MB)', '分析时间']
        
        with open(report_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for model_data in existing_data.values():
                writer.writerow(model_data)
        
        print(f"性能报告已更新: {report_file}")
        return report_file
    


if __name__ == '__main__':
    # 模型路径
    # model_paths = [
    #     'weights/trained_weights/yolo26s_SE_flower_cls_model.pt',
    #     'weights/trained_weights/yolo26s_CBAM_flower_cls_model.pt',
    #     'weights/trained_weights/yolo26s_flower_cls_model.pt']

    weights_dir = Path('weights/trained_weights')
    model_paths = [str(p) for p in weights_dir.glob('*.pt')]

    if not model_paths:
        print("未找到训练权重文件，请确保weights/trained_weights目录中有模型文件")
        model_paths = ['weights/trained_weights/yolo26s_flower_cls_model.pt']

    print(f"找到 {len(model_paths)} 个模型文件:")
    for model_path in model_paths:
        print(f"- {Path(model_path).name}")

    # 测试数据路径
    test_data_path = 'flower_photos_split'
    
    # 初始化分析器
    analyzer = ModelPerformanceAnalyzer(model_paths, test_data_path)
    
    # 分析性能
    analyzer.analyze_performance()
    
    # 生成报告
    analyzer.generate_report()
    
    print("\n模型性能分析完成！")

