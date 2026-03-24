from ultralytics import YOLO
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
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
            model_name = Path(model_path).stem
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
        try:
            results = model.val(data=self.test_data_path, device=self.device)
            metrics = {
                'accuracy_top1': results.top1,
                'accuracy_top5': results.top5
            }
            
            # 尝试手动计算其他指标
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
                class_to_idx = {cls: i for i, cls in enumerate(classes)}
                
                # 遍历测试数据
                for cls in classes:
                    cls_dir = os.path.join(test_dir, cls)
                    img_paths = glob.glob(os.path.join(cls_dir, '*.*'))
                    
                    for img_path in img_paths:
                        # 预测
                        result = model.predict(img_path, device=self.device, verbose=False)[0]
                        # 获取预测类别
                        pred_idx = result.probs.top1
                        pred_cls = classes[pred_idx] if pred_idx < len(classes) else 'unknown'
                        
                        # 记录真实标签和预测标签
                        y_true.append(class_to_idx[cls])
                        y_pred.append(class_to_idx.get(pred_cls, -1))
                
                # 计算指标
                if y_true and y_pred:
                    metrics['accuracy'] = accuracy_score(y_true, y_pred)
                    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    print(f"手动计算的指标: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
                else:
                    # 如果没有测试数据，使用top1作为默认值
                    metrics['accuracy'] = results.top1
                    metrics['precision'] = results.top1
                    metrics['recall'] = results.top1
                    metrics['f1'] = results.top1
            else:
                # 如果测试目录不存在，使用top1作为默认值
                metrics['accuracy'] = results.top1
                metrics['precision'] = results.top1
                metrics['recall'] = results.top1
                metrics['f1'] = results.top1
            
            print(f"模型准确率: Top-1={results.top1:.4f}, Top-5={results.top5:.4f}")
            
        except Exception as e:
            print(f"测试准确率时出错: {e}")
            metrics = {
                'accuracy_top1': 0,
                'accuracy_top5': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
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
        """生成性能报告"""
        if output_dir is None:
            output_dir = Path(__file__).parent / 'performance_reports'
        output_dir.mkdir(exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f'性能报告.md'
        
        # 生成Markdown报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 模型性能分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 性能指标表格
            f.write("## 性能指标对比\n")
            f.write("| 模型名称 | 参数量 (百万) | 模型大小 (MB) | 推理时间 (ms) | FPS | Top-1 准确率 | Top-5 准确率 | 准确率 | 精确率 | 召回率 | F1 值 | 内存使用 (MB) |\n")
            f.write("|---------|------------|-------------|-------------|-----|------------|------------|-------|-------|-------|-------|------------|\n")
            
            for model_name, metrics in self.performance_metrics.items():
                params_million = metrics['total_params'] / 1e6
                f.write(f"| {model_name} | {params_million:.2f} | {metrics['params_mb']:.2f} | {metrics['inference_time_ms']:.2f} | {metrics['fps']:.2f} | {metrics['accuracy_top1']:.4f} | {metrics['accuracy_top5']:.4f} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['memory_usage_mb']:.2f} |\n")
            
            # 详细指标
            f.write("\n## 详细指标\n")
            for model_name, metrics in self.performance_metrics.items():
                f.write(f"### {model_name}\n")
                f.write(f"- 总参数量: {metrics['total_params']:,}\n")
                # f.write(f"- 可训练参数量: {metrics['trainable_params']:,}\n")
                f.write(f"- 模型大小: {metrics['params_mb']:.2f} MB\n")
                f.write(f"- 推理时间: {metrics['inference_time_ms']:.2f} ms\n")
                f.write(f"- FPS: {metrics['fps']:.2f}\n")
                f.write(f"- Top-1 准确率: {metrics['accuracy_top1']:.4f}\n")
                f.write(f"- Top-5 准确率: {metrics['accuracy_top5']:.4f}\n")
                f.write(f"- 准确率: {metrics['accuracy']:.4f}\n")
                f.write(f"- 精确率: {metrics['precision']:.4f}\n")
                f.write(f"- 召回率: {metrics['recall']:.4f}\n")
                f.write(f"- F1 值: {metrics['f1']:.4f}\n")
                f.write(f"- 内存使用: {metrics['memory_usage_mb']:.2f} MB\n\n")
        
        print(f"性能报告已生成: {report_file}")
        return report_file
    
    def plot_results(self, output_dir=None):
        """绘制性能对比图表"""
        if output_dir is None:
            output_dir = Path(__file__).parent / 'performance_reports'
        output_dir.mkdir(exist_ok=True)
        
        # 准备数据
        model_names = list(self.performance_metrics.keys())
        metrics = list(self.performance_metrics.values())
        
        # 提取数据
        params_million = [m['total_params'] / 1e6 for m in metrics]
        inference_time = [m['inference_time_ms'] for m in metrics]
        fps = [m['fps'] for m in metrics]
        top1_acc = [m['accuracy_top1'] for m in metrics]
        top5_acc = [m['accuracy_top5'] for m in metrics]
        accuracy = [m['accuracy'] for m in metrics]
        precision = [m['precision'] for m in metrics]
        recall = [m['recall'] for m in metrics]
        f1 = [m['f1'] for m in metrics]
        memory_usage = [m['memory_usage_mb'] for m in metrics]
        
        # 1. 准确率对比
        plt.figure(figsize=(10, 6))
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, top1_acc, width, label='Top-1 准确率')
        plt.bar(x + width/2, top5_acc, width, label='Top-5 准确率')
        plt.xlabel('模型')
        plt.ylabel('准确率')
        plt.title('不同注意力机制模型的准确率对比')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        acc_plot_path = output_dir / '不同注意力机制模型的准确率对比.png'
        plt.savefig(acc_plot_path)
        plt.close()
        
        # 2. 分类性能指标对比
        plt.figure(figsize=(12, 8))
        
        # 定义指标名称和颜色
        metrics_names = ['准确率', '精确率', '召回率', 'F1 值']
        metrics_data = [accuracy, precision, recall, f1]
        colors = plt.cm.tab10(np.arange(len(model_names)))
        
        # 设置条形图宽度
        bar_width = 0.8 / len(model_names)
        
        # 绘制每个指标的条形图
        for i, (metric_name, metric_data) in enumerate(zip(metrics_names, metrics_data)):
            # 为每个指标创建一组条形图
            for j, (model_name, value) in enumerate(zip(model_names, metric_data)):
                # 计算每个模型在该指标组中的位置
                x_pos = i + j * bar_width
                plt.bar(x_pos, value, bar_width, color=colors[j], label=model_name if i == 0 else "")
        
        # 设置x轴标签
        plt.xlabel('指标')
        plt.ylabel('指标值')
        plt.title('不同注意力机制模型的分类性能指标对比')
        # 调整x轴刻度位置和标签
        plt.xticks(np.arange(len(metrics_names)) + bar_width * (len(model_names) - 1) / 2, metrics_names)
        
        # 添加图例
        plt.legend()
        plt.tight_layout()
        classification_plot_path = output_dir / '不同注意力机制模型的分类性能指标对比.png'
        plt.savefig(classification_plot_path)
        plt.close()
        
        # 3. 速度对比
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, fps, color='green')
        plt.xlabel('模型')
        plt.ylabel('FPS')
        plt.title('不同注意力机制模型的推理速度对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        speed_plot_path = output_dir / '不同注意力机制模型的推理速度对比.png'
        plt.savefig(speed_plot_path)
        plt.close()
        
        # 4. 参数量对比
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, params_million, color='blue')
        plt.xlabel('模型')
        plt.ylabel('参数量 (百万)')
        plt.title('不同注意力机制模型的参数量对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        params_plot_path = output_dir / '不同注意力机制模型的参数量对比.png'
        plt.savefig(params_plot_path)
        plt.close()
        
        # 5. 内存使用对比
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, memory_usage, color='purple')
        plt.xlabel('模型')
        plt.ylabel('内存使用 (MB)')
        plt.title('不同注意力机制模型的内存使用对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        memory_plot_path = output_dir / '不同注意力机制模型的内存使用对比.png'
        plt.savefig(memory_plot_path)
        plt.close()
        
        # 6. 综合性能雷达图
        plt.figure(figsize=(12, 8))
        
        # 打印原始数据，便于调试
        print("\n雷达图原始数据:")
        print(f"准确率: {accuracy}")
        print(f"精确率: {precision}")
        print(f"召回率: {recall}")
        print(f"F1值: {f1}")
        
        # 准备雷达图数据，只使用准确率、精确率、召回率、F1值
        metrics_to_plot = {
            '准确率': accuracy,
            '精确率': precision,
            '召回率': recall,
            'F1值': f1
        }
        
        num_metrics = len(metrics_to_plot)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        ax = plt.subplot(111, polar=True)
        
        # 确保数据长度与模型数量一致
        print(f"\n模型数量: {len(model_names)}")
        print(f"数据长度 - 准确率: {len(accuracy)}")
        print(f"数据长度 - 精确率: {len(precision)}")
        print(f"数据长度 - 召回率: {len(recall)}")
        print(f"数据长度 - F1值: {len(f1)}")
        
        for i, model_name in enumerate(model_names):
            # 确保索引不越界
            if i < len(accuracy) and i < len(precision) and i < len(recall) and i < len(f1):
                values = [metrics_to_plot[metric][i] for metric in metrics_to_plot.keys()]
                values += values[:1]  # 闭合
                print(f"绘制模型 {model_name} 的雷达图数据: {values}")
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
                ax.fill(angles, values, alpha=0.1)
            else:
                print(f"模型 {model_name} 的索引 {i} 越界，跳过绘制")
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(list(metrics_to_plot.keys()))
        
        # 准确率、精确率、召回率、F1值都是0-1之间的值，设置范围为0.5-1以增强对比
        ax.set_ylim(0.5, 1)
        plt.title('不同注意力机制模型的综合性能对比', size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        radar_plot_path = output_dir / '不同注意力机制模型的综合性能对比.png'
        plt.savefig(radar_plot_path)
        plt.close()
        
        print(f"性能对比图表已生成到: {output_dir}")
        return output_dir

def main():
    # 模型路径
    model_paths = [
        'weights/trained_weights/yolo26n_SE_flower_cls_model.pt',
        'weights/trained_weights/yolo26n_CBAM_flower_cls_model.pt',
        'weights/trained_weights/yolo26n_flower_cls_model.pt']

    # weights_dir = Path('weights/trained_weights')
    # model_paths = [str(p) for p in weights_dir.glob('*.pt')]
    #
    # if not model_paths:
    #     print("未找到训练权重文件，请确保weights/trained_weights目录中有模型文件")
    #     return
    #
    # print(f"找到 {len(model_paths)} 个模型文件:")
    # for model_path in model_paths:
    #     print(f"- {Path(model_path).name}")
    
    # 测试数据路径
    test_data_path = 'flower_photos_split'
    
    # 初始化分析器
    analyzer = ModelPerformanceAnalyzer(model_paths, test_data_path)
    
    # 分析性能
    analyzer.analyze_performance()
    
    # 生成报告
    analyzer.generate_report()
    
    # 绘制结果
    analyzer.plot_results()
    
    print("\n模型性能分析完成！")

if __name__ == '__main__':
    main()
