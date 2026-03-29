from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
from ultralytics import YOLO
import os
from pathlib import Path
import json
import csv
import shutil
import pandas as pd
import random
from datetime import datetime
from io import StringIO, BytesIO
from stats_load_history import load_history, save_history, calculate, get_db_connection, calculate_class_distribution, calculate_confidence_by_class, calculate_temporal_trends
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 用于session管理

# 设置上传文件夹
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 项目根目录
root_dir = Path(__file__).parent
print(f"项目根目录: {root_dir}")

# 选择模型
model_name = 'yolo26s_SE_flower_cls'

# 加载YOLO分类模型（如果没有训练好的模型，可以使用预训练模型）
# 暂时注释掉模型加载，以测试应用程序启动
try:
    # 尝试使用训练好的分类模型
    print("使用训练好的花朵分类模型")
    model_path = os.path.join(root_dir, f'weights/trained_weights/{model_name}_model.pt')
except Exception as e:
    # 如果没有训练好的模型，使用预训练模型
    print(f"无法加载训练好的模型: {e}")
    model_path = os.path.join(root_dir, 'weights/yolo26n-cls.pt')

model = YOLO(model_path)

# 从模型中获取类别映射
CLASSES = model.names
print(f"从模型中获取的类别映射: {CLASSES}")

# 从CSV文件加载类别映射
def load_class_mapping():
    try:
        csv_path = os.path.join(root_dir, 'flower_classes.csv')
        df = pd.read_csv(csv_path)
        # 创建id到中文名称的映射
        id_to_zh = dict(zip(df['id'], df['name_zh']))
        # 创建id到英文名称的映射
        id_to_en = dict(zip(df['id'], df['name']))
        # 创建英文到中文的映射（保持向后兼容）
        en_to_zh = dict(zip(df['name'], df['name_zh']))
        # 创建中文到英文的映射（保持向后兼容）
        zh_to_en = dict(zip(df['name_zh'], df['name']))
    except Exception as e:
        print(f"加载CSV文件时出错: {e}")
        id_to_zh, id_to_en, en_to_zh, zh_to_en = {}, {}, {}, {}
    return id_to_zh, id_to_en, en_to_zh, zh_to_en

# 加载类别映射
ID_TO_ZH, ID_TO_EN, CHINESE_CLASSES, CLASS_MAPPING = load_class_mapping()
print(f"从CSV文件加载的ID到中文映射: {ID_TO_ZH}")

# 登录装饰器
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    decorated_function = wraps(f)(decorated_function)
    return decorated_function

# 检查用户是否登录
def is_logged_in():
    return 'user_id' in session

# 用户登录
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        if not conn:
            flash('数据库连接失败，请稍后再试', 'danger')
            return redirect(url_for('login'))
        
        try:
            with conn.cursor() as cursor:
                # 检查用户是否存在
                cursor.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()
                
                if user and user['password'] == password:
                    # 登录成功
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    flash('登录成功', 'success')
                    return redirect(url_for('index'))
                else:
                    flash('用户名或密码错误', 'danger')
        except Exception as e:
            print(f"登录时出错: {e}")
            flash('登录失败，请稍后再试', 'danger')
        finally:
            if conn:
                conn.close()
    
    return render_template('login.html')

# 用户注册
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        if not conn:
            flash('数据库连接失败，请稍后再试', 'danger')
            return redirect(url_for('register'))
        
        try:
            with conn.cursor() as cursor:
                # 检查用户名是否已存在
                cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
                if cursor.fetchone():
                    flash('用户名已存在', 'danger')
                    return redirect(url_for('register'))
                
                # 插入新用户
                cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
                conn.commit()
                
                # 自动登录
                cursor.execute("SELECT id, username FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()
                session['user_id'] = user['id']
                session['username'] = user['username']
                
                flash('注册成功', 'success')
                return redirect(url_for('index'))
        except Exception as e:
            print(f"注册时出错: {e}")
            if conn:
                conn.rollback()
            flash('注册失败，请稍后再试', 'danger')
        finally:
            if conn:
                conn.close()
    
    return render_template('register.html')

# 用户登出
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('已成功登出', 'success')
    return redirect(url_for('index'))

@app.route('/')
def index():
    # 从性能报告CSV文件读取可用模型列表
    performance_report_path = os.path.join(root_dir, 'performance_reports', '性能报告.csv')
    available_models = []
    
    try:
        if os.path.exists(performance_report_path):
            with open(performance_report_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model_name_only = row['模型名称'].replace('_model', '')
                    if model_name_only not in available_models:
                        available_models.append(model_name_only)
    except Exception as e:
        print(f"读取模型列表时出错: {e}")
    
    return render_template('index.html', is_logged_in=is_logged_in(), username=session.get('username'), model_name=model_name, available_models=available_models)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('file')
    
    # 检查文件列表是否为空
    if not files or all(file.filename == '' for file in files):
        return redirect(request.url)
    
    # 处理所有上传的文件
    all_results = []
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'files': []
    }
    
    total_confidence = 0
    total_predictions = 0
    
    for file in files:
        if file.filename == '':
            continue
        
        # 保存上传的文件
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 使用真实的模型预测
        results = model.predict(filepath)
        
        # 获取预测结果
        predictions = []
        if results:
            # 处理模型预测结果
            for result in results:
                # 获取预测的类别索引
                class_idx = result.probs.top1
                # 将索引转换为对应的ID（如0对应c0, 1对应c1等）
                class_id = f'c{class_idx}'
                # 获取中文类别名称
                class_name = ID_TO_ZH.get(class_id, f'类别 {class_idx}')
                # 获取置信度
                confidence_value = round(float(result.probs.top1conf), 2)
                predictions.append({
                    'class': class_name,
                    'confidence': confidence_value
                })
                total_confidence += confidence_value
                total_predictions += 1
        
        # 如果没有预测结果，返回默认结果
        if not predictions:
            predictions.append({
                'class': 'Unknown',
                'confidence': 0.0
            })
            total_predictions += 1
        
        all_results.append({
            'filename': filename,
            'predictions': predictions
        })
        
        # 添加到历史记录
        history_entry['files'].append({
            'filename': filename,
            'predictions': predictions
        })
        
    
    # 计算平均置信度
    average_confidence = 0.0
    if total_predictions > 0:
        average_confidence = round(total_confidence / total_predictions, 2)
    
    # 保存历史记录
    user_id = session.get('user_id')
    save_history(history_entry, user_id=user_id)
    
    return render_template('result.html', all_results=all_results, average_confidence=average_confidence, is_logged_in=is_logged_in(), username=session.get('username'))

@app.route('/history')
@login_required
def history():
    user_id = session.get('user_id')
    history = load_history(user_id)
    
    # 使用 calculate_stats 模块计算历史统计数据
    total_files, average_confidence = calculate(history)
    
    # 计算类别分布
    class_distribution = calculate_class_distribution(history)
    
    # 计算各类别平均置信度
    confidence_by_class = calculate_confidence_by_class(history)
    
    # 计算时间趋势
    temporal_trends = calculate_temporal_trends(history)
    
    return render_template('history.html', 
                           history=history, 
                           total_files=total_files, 
                           average_confidence=average_confidence,
                           class_distribution=class_distribution,
                           confidence_by_class=confidence_by_class,
                           temporal_trends=temporal_trends,
                           is_logged_in=is_logged_in(), 
                           username=session.get('username'))

@app.route('/export/csv')
@login_required
def export_csv():
    user_id = session.get('user_id')
    history = load_history(user_id)
    
    # 创建CSV内容
    output = StringIO()
    writer = csv.writer(output)
    
    # 写入表头
    writer.writerow(['时间', '文件名', '花朵种类', '置信度'])
    
    # 写入数据
    for entry in history:
        timestamp = entry['timestamp']
        for file_entry in entry['files']:
            filename = file_entry['filename']
            for pred in file_entry['predictions']:
                writer.writerow([timestamp, filename, pred['class'], pred['confidence']])
    
    # 转换为BytesIO
    output.seek(0)
    csv_data = output.getvalue().encode('utf-8')
    csv_bytes = BytesIO(csv_data)
    
    return send_file(
        csv_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'flower_recognition_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

@app.route('/correct', methods=['POST'])
@login_required
def correct():
    """处理用户纠正识别结果的请求"""
    filename = request.form.get('filename')
    predicted_class = request.form.get('predicted_class')
    correct_class = request.form.get('correct_class')
    user_id = session.get('user_id')
    
    # 使用从CSV文件加载的中文到英文的映射
    
    # 保存纠正信息到数据库
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                # 插入纠正记录
                cursor.execute(
                    "INSERT INTO corrections (filename, predicted_class, correct_class, user_id, timestamp) VALUES (%s, %s, %s, %s, %s)",
                    (filename, predicted_class, correct_class, user_id, datetime.now())
                )
                conn.commit()
                
                # 将纠正后的图片复制到对应数据集目录
                if correct_class in CLASS_MAPPING:
                    # 源文件路径
                    src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    # 目标目录
                    dataset_dir = os.path.join('..', 'flower_photos', CLASS_MAPPING[correct_class])
                    # 确保目标目录存在
                    os.makedirs(dataset_dir, exist_ok=True)
                    # 目标文件路径
                    dst_path = os.path.join(dataset_dir, filename)
                    # 复制文件
                    shutil.copy2(src_path, dst_path)
                    print(f"已将图片 {filename} 复制到 {dataset_dir}")
                
                flash('纠正信息已提交，感谢您的反馈！', 'success')
        except Exception as e:
            print(f"保存纠正信息时出错: {e}")
            if conn:
                conn.rollback()
            flash('提交纠正信息失败，请稍后再试', 'danger')
        finally:
            if conn:
                conn.close()
    else:
        flash('数据库连接失败，请稍后再试', 'danger')
    
    # 重定向回结果页面
    return redirect(request.referrer or url_for('index'))

# 更换模型
@app.route('/change_model', methods=['POST'])
@login_required
def change_model():
    global model_name
    new_model = request.form.get('model_name')
    if new_model:
        model_name = new_model
    # 获取来源页面，默认为首页
    referrer = request.referrer
    if referrer and 'performance' in referrer:
        return redirect(url_for('performance'))
    else:
        return redirect(url_for('index'))

# 模型性能数据
@app.route('/performance')
@login_required
def performance():
    # 从性能报告CSV文件读取数据
    import csv
    import os
    
    performance_report_path = os.path.join(root_dir, 'performance_reports', '性能报告.csv')
    performance_data = {
        'model_name': model_name,
        'params': 0.0,
        'model_size': 0.0,
        'inference_time': 0.0,
        'fps': 0.0,
        'top1_accuracy': 0.0,
        'top5_accuracy': 0.0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'memory_usage': 0.0,
        'analysis_time': '',
        'available_models': []
    }
    
    try:
        if os.path.exists(performance_report_path):
            with open(performance_report_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                target_model_name = f'{model_name}'
                found = False
                
                for row in reader:
                    # 收集所有可用模型
                    model_name_only = row['模型名称']
                    if model_name_only not in performance_data['available_models']:
                        performance_data['available_models'].append(model_name_only)
                    
                    if row['模型名称'] == target_model_name:
                        # 提取性能数据
                        performance_data['params'] = float(row['参数量 (百万)'])
                        performance_data['model_size'] = float(row['模型大小 (MB)'])
                        performance_data['inference_time'] = float(row['推理时间 (ms)'])
                        performance_data['fps'] = float(row['FPS'])
                        performance_data['top1_accuracy'] = float(row['Top-1 准确率'])
                        performance_data['top5_accuracy'] = float(row['Top-5 准确率'])
                        performance_data['accuracy'] = float(row['准确率'])
                        performance_data['precision'] = float(row['精确率'])
                        performance_data['recall'] = float(row['召回率'])
                        performance_data['f1_score'] = float(row['F1 值'])
                        performance_data['memory_usage'] = float(row['内存使用 (MB)'])
                        performance_data['analysis_time'] = row['分析时间']

                        found = True
                
                if not found:
                    print(f"未找到模型 {target_model_name} 的性能数据")
        else:
            print(f"性能报告文件不存在: {performance_report_path}")
    except Exception as e:
        print(f"读取性能数据时出错: {e}")
    
    return render_template('performance.html', performance_data=performance_data, is_logged_in=is_logged_in(), username=session.get('username'))

# 模型训练过程数据
@app.route('/training')
@login_required
def training():
    # 从CSV文件读取真实训练数据
    results_path = os.path.join(root_dir, 'runs/classify/flower_detection', model_name, 'results.csv')
    
    training_data = {
        'epochs': 0,
        'epoch': [],
        'time': [],
        'loss': [],
        'accuracy': [],
        'accuracy_top5': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': [],
        'learning_rates': {}
    }
    
    try:
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            training_data['epochs'] = len(df)
            training_data['epoch'] = df['epoch'].tolist()
            training_data['time'] = df['time'].tolist()
            training_data['loss'] = df['train/loss'].tolist()
            training_data['accuracy'] = df['metrics/accuracy_top1'].tolist()
            training_data['accuracy_top5'] = df['metrics/accuracy_top5'].tolist()
            training_data['val_loss'] = df['val/loss'].tolist()
            training_data['val_accuracy'] = df['metrics/accuracy_top5'].tolist()
            # 读取所有学习率列
            training_data['learning_rate'] = df['lr/pg0'].tolist()
            for i in range(8):
                col_name = f'lr/pg{i}'
                if col_name in df.columns:
                    training_data['learning_rates'][f'pg{i}'] = df[col_name].tolist()
        else:
            print(f"训练结果文件不存在: {results_path}")
    except Exception as e:
        print(f"读取训练数据时出错: {e}")
    
    return render_template('training.html', training_data=training_data, is_logged_in=is_logged_in(), username=session.get('username'))


if __name__ == '__main__':
    app.run(debug=True, port=1104)
    