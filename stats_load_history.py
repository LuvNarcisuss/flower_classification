import json
import os
import pymysql
from datetime import datetime

# 数据库连接配置
db_config = {
    'host': 'localhost',
    'user': 'Narcisuss',
    'password': '688376',
    'database': 'flower_classification',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 获取数据库连接
def get_db_connection():
    try:
        conn = pymysql.connect(**db_config)
        return conn
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return None

# 加载历史记录
def load_history(user_id=None):
    """从MySQL数据库加载历史记录"""
    history = []
    conn = get_db_connection()
    if not conn:
        return history
    
    try:
        with conn.cursor() as cursor:
            # 根据用户ID查询历史记录
            if user_id:
                cursor.execute("""
                    SELECT id, timestamp, user_id, filename, class_name, confidence
                    FROM history
                    WHERE user_id = %s
                    ORDER BY timestamp DESC
                """, (user_id,))
            else:
                # 未登录时返回空记录
                return history
            
            results = cursor.fetchall()
            
            # 整理数据结构
            history_map = {}
            for row in results:
                # 使用时间戳作为唯一标识，因为合并表中每条记录都有自己的ID
                # 但为了保持原有结构，我们按时间戳分组
                timestamp = row['timestamp'].isoformat()
                if timestamp not in history_map:
                    history_map[timestamp] = {
                        'id': row['id'],
                        'timestamp': timestamp,
                        'user_id': row['user_id'],
                        'files': []
                    }
                
                # 查找文件是否已存在
                file_exists = False
                for file_entry in history_map[timestamp]['files']:
                    if file_entry['filename'] == row['filename']:
                        file_exists = True
                        # 添加预测结果
                        file_entry['predictions'].append({
                            'class': row['class_name'],
                            'confidence': row['confidence']
                        })
                        break
                
                if not file_exists:
                    # 创建新文件条目
                    file_entry = {
                        'filename': row['filename'],
                        'predictions': [{
                            'class': row['class_name'],
                            'confidence': row['confidence']
                        }]
                    }
                    history_map[timestamp]['files'].append(file_entry)
            
            # 转换为列表
            history = list(history_map.values())
    except Exception as e:
        print(f"加载历史记录时出错: {e}")
    finally:
        if conn:
            conn.close()
    
    return history

# 保存历史记录
def save_history(history_entry, user_id=None):
    """保存历史记录到MySQL数据库"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cursor:
            # 开始事务
            conn.begin()
            
            # 插入历史记录
            for file_entry in history_entry['files']:
                for pred in file_entry['predictions']:
                    cursor.execute(
                        "INSERT INTO history (timestamp, user_id, filename, class_name, confidence) VALUES (%s, %s, %s, %s, %s)",
                        (history_entry['timestamp'], user_id, file_entry['filename'], pred['class'], pred['confidence'])
                    )
            
            # 提交事务
            conn.commit()
            return True
    except Exception as e:
        print(f"保存历史记录时出错: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# 计算统计数据
def calculate(history):
    total_files = 0
    total_confidence = 0.0
    total_predictions = 0
    
    for entry in history:
        if isinstance(entry, dict) and 'files' in entry:
            for file_entry in entry['files']:
                if isinstance(file_entry, dict):
                    total_files += 1
                    if 'predictions' in file_entry:
                        for pred in file_entry['predictions']:
                            if isinstance(pred, dict) and 'confidence' in pred:
                                total_confidence += pred['confidence']
                                total_predictions += 1
    
    average_confidence = 0.0
    if total_predictions > 0:
        average_confidence = round(total_confidence / total_predictions, 2)
    
    return total_files, average_confidence

# 计算各类别的识别分布
def calculate_class_distribution(history):
    class_counts = {}
    
    for entry in history:
        if isinstance(entry, dict) and 'files' in entry:
            for file_entry in entry['files']:
                if isinstance(file_entry, dict) and 'predictions' in file_entry:
                    for pred in file_entry['predictions']:
                        if isinstance(pred, dict) and 'class' in pred:
                            class_name = pred['class']
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return class_counts

# 计算各类别的平均置信度
def calculate_confidence_by_class(history):
    class_confidence = {}
    class_counts = {}
    
    for entry in history:
        if isinstance(entry, dict) and 'files' in entry:
            for file_entry in entry['files']:
                if isinstance(file_entry, dict) and 'predictions' in file_entry:
                    for pred in file_entry['predictions']:
                        if isinstance(pred, dict) and 'class' in pred and 'confidence' in pred:
                            class_name = pred['class']
                            confidence = pred['confidence']
                            class_confidence[class_name] = class_confidence.get(class_name, 0) + confidence
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # 计算平均置信度
    class_avg_confidence = {}
    for class_name, total_confidence in class_confidence.items():
        if class_counts.get(class_name, 0) > 0:
            class_avg_confidence[class_name] = round(total_confidence / class_counts[class_name], 2)
    
    return class_avg_confidence

# 计算时间趋势
def calculate_temporal_trends(history):
    # 按日期分组
    daily_stats = {}
    
    for entry in history:
        if isinstance(entry, dict) and 'timestamp' in entry:
            try:
                # 提取日期部分
                date = entry['timestamp'].split('T')[0]
                if date not in daily_stats:
                    daily_stats[date] = {
                        'count': 0,
                        'total_confidence': 0.0,
                        'predictions': 0
                    }
                
                if 'files' in entry:
                    for file_entry in entry['files']:
                        if isinstance(file_entry, dict) and 'predictions' in file_entry:
                            daily_stats[date]['count'] += 1
                            for pred in file_entry['predictions']:
                                if isinstance(pred, dict) and 'confidence' in pred:
                                    daily_stats[date]['total_confidence'] += pred['confidence']
                                    daily_stats[date]['predictions'] += 1
            except Exception as e:
                print(f"处理时间戳时出错: {e}")
    
    # 计算每日平均置信度
    temporal_data = []
    for date, stats in sorted(daily_stats.items()):
        avg_confidence = 0.0
        if stats['predictions'] > 0:
            avg_confidence = round(stats['total_confidence'] / stats['predictions'], 2)
        temporal_data.append({
            'date': date,
            'count': stats['count'],
            'average_confidence': avg_confidence
        })
    
    return temporal_data

if __name__ == "__main__":
    history = load_history()
    total_files, average_confidence = calculate(history)
    print(f"总文件数: {total_files}")
    print(f"平均置信度: {average_confidence}")
    
    # 测试地图数据
    map_data = get_flower_map()
    print(f"花朵地图数据条数: {len(map_data)}")
