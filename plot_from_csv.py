import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore', category=FutureWarning)

class ModelPerformancePlotter:
    def __init__(self, csv_file):
        """初始化模型性能绘图器
        
        Args:
            csv_file: 性能报告CSV文件路径
        """
        self.csv_file = Path(csv_file)
        self.model_data = {}
        self.load_data()
    
    def load_data(self):
        """从CSV文件加载模型性能数据"""
        if not self.csv_file.exists():
            print(f"错误: 找不到性能报告文件 {self.csv_file}")
            return
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row['模型名称']
                self.model_data[model_name] = {
                    '参数量 (百万)': float(row['参数量 (百万)']),
                    '模型大小 (MB)': float(row['模型大小 (MB)']),
                    '推理时间 (ms)': float(row['推理时间 (ms)']),
                    'FPS': float(row['FPS']),
                    'Top-1 准确率': float(row['Top-1 准确率']),
                    'Top-5 准确率': float(row['Top-5 准确率']),
                    '准确率': float(row['准确率']),
                    '精确率': float(row['精确率']),
                    '召回率': float(row['召回率']),
                    'F1 值': float(row['F1 值']),
                    '内存使用 (MB)': float(row['内存使用 (MB)']),
                    '分析时间': row['分析时间']
                }
        
        # print(f"已加载 {len(self.model_data)} 个模型的性能数据")
        # print("可用模型:")
        # for i, model_name in enumerate(self.model_data.keys(), 1):
        #     print(f"{i}. {model_name}")
    
    def plot_comparison(self, model_names, output_dir=None):
        """绘制指定模型的性能对比图表
        
        Args:
            model_names: 要对比的模型名称列表
            output_dir: 输出目录
        """
        if not model_names:
            print("错误: 未指定要对比的模型")
            return
        
        # 验证模型名称是否存在
        valid_model_names = []
        for model_name in model_names:
            if model_name in self.model_data:
                valid_model_names.append(model_name)
            else:
                print(f"警告: 模型 {model_name} 不存在，将被跳过")
        
        if not valid_model_names:
            print("错误: 没有有效的模型名称")
            return
        
        if output_dir is None:
            output_dir = Path(__file__).parent / 'performance_reports'
        output_dir.mkdir(exist_ok=True)
        
        # 准备数据
        metrics = []
        for model_name in valid_model_names:
            metrics.append(self.model_data[model_name])
        
        # 提取数据
        params_million = [m['参数量 (百万)'] for m in metrics]
        inference_time = [m['推理时间 (ms)'] for m in metrics]
        fps = [m['FPS'] for m in metrics]
        top1_acc = [m['Top-1 准确率'] for m in metrics]
        top5_acc = [m['Top-5 准确率'] for m in metrics]
        accuracy = [m['准确率'] for m in metrics]
        precision = [m['精确率'] for m in metrics]
        recall = [m['召回率'] for m in metrics]
        f1 = [m['F1 值'] for m in metrics]
        memory_usage = [m['内存使用 (MB)'] for m in metrics]
        
        # 1. 准确率对比
        plt.figure(figsize=(12, 8))
        
        # 定义指标名称和颜色
        metrics_names = ['Top-1 准确率', 'Top-5 准确率']
        metrics_data = [top1_acc, top5_acc]
        colors = plt.cm.tab10(np.arange(len(valid_model_names)))
        
        # 设置条形图宽度
        bar_width = 0.8 / len(valid_model_names)
        
        # 绘制每个指标的条形图
        for i, (metric_name, metric_data) in enumerate(zip(metrics_names, metrics_data)):
            # 为每个指标创建一组条形图
            for j, (model_name, value) in enumerate(zip(valid_model_names, metric_data)):
                # 计算每个模型在该指标组中的位置
                x_pos = i + j * bar_width
                plt.bar(x_pos, value, bar_width, color=colors[j], label=model_name if i == 0 else "")
        
        # 设置x轴标签
        plt.xlabel('指标')
        plt.ylabel('准确率')
        plt.title('不同注意力机制模型的准确率对比')
        # 调整x轴刻度位置和标签
        plt.xticks(np.arange(len(metrics_names)) + bar_width * (len(valid_model_names) - 1) / 2, metrics_names)
        
        # 添加图例
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
        colors = plt.cm.tab10(np.arange(len(valid_model_names)))
        
        # 设置条形图宽度
        bar_width = 0.8 / len(valid_model_names)
        
        # 绘制每个指标的条形图
        for i, (metric_name, metric_data) in enumerate(zip(metrics_names, metrics_data)):
            # 为每个指标创建一组条形图
            for j, (model_name, value) in enumerate(zip(valid_model_names, metric_data)):
                # 计算每个模型在该指标组中的位置
                x_pos = i + j * bar_width
                plt.bar(x_pos, value, bar_width, color=colors[j], label=model_name if i == 0 else "")
        
        # 设置x轴标签
        plt.xlabel('指标')
        plt.ylabel('指标值')
        plt.title('不同注意力机制模型的分类性能指标对比')
        # 调整x轴刻度位置和标签
        plt.xticks(np.arange(len(metrics_names)) + bar_width * (len(valid_model_names) - 1) / 2, metrics_names)
        
        # 添加图例
        plt.legend()
        plt.tight_layout()
        classification_plot_path = output_dir / '不同注意力机制模型的分类性能指标对比.png'
        plt.savefig(classification_plot_path)
        plt.close()
        
        # 3. 速度对比
        plt.figure(figsize=(10, 6))
        plt.bar(valid_model_names, fps, color='green')
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
        plt.bar(valid_model_names, params_million, color='blue')
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
        plt.bar(valid_model_names, memory_usage, color='purple')
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
        print(f"\n模型数量: {len(valid_model_names)}")
        print(f"数据长度 - 准确率: {len(accuracy)}")
        print(f"数据长度 - 精确率: {len(precision)}")
        print(f"数据长度 - 召回率: {len(recall)}")
        print(f"数据长度 - F1值: {len(f1)}")
        
        for i, model_name in enumerate(valid_model_names):
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
        
        # 设置范围为0.6-1以增强对比
        ax.set_ylim(0.6, 1)
        plt.title('不同注意力机制模型的综合性能对比', size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        radar_plot_path = output_dir / '不同注意力机制模型的综合性能对比.png'
        plt.savefig(radar_plot_path)
        plt.close()
        
        print(f"性能对比图表已生成到: {output_dir}")
        return output_dir

if __name__ == '__main__':
    # 性能报告文件路径
    report_file = Path(__file__).parent / 'performance_reports' / '性能报告.csv'
    # print(report_file)
    
    # 初始化绘图器
    plotter = ModelPerformancePlotter(report_file)

    # 请根据实际情况修改以下列表，添加要对比的模型名称
    # selected_models = [
    #     'yolo26s_SE_flower_cls_model',
    #     'yolo26s_CBAM_flower_cls_model',
    #     'yolo26s_flower_cls_model'
    # ]

    selected_models = list(plotter.model_data.keys())

    print(f"从 CSV 文件读取到的模型名称列表:")
    for model in selected_models:
        print(f"  - {model}")
    print(f"\n共{len(selected_models)}条模型数据")

    
    print(f"\n要对比的模型: {selected_models}")
    
    # 绘制对比图表
    plotter.plot_comparison(selected_models)

    print("图像绘制完成")
