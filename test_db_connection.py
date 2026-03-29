import pymysql
import os

# 数据库连接配置
db_config = {
    'host': 'localhost',
    'user': 'Narcisuss',
    'password': '688376',
    'database': 'flower_classification',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def test_connection():
    """测试数据库连接"""
    try:
        conn = pymysql.connect(**db_config)
        print("数据库连接成功！")
        
        # 测试查询
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"查询结果: {result}")
        
        conn.close()
        return True
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return False

if __name__ == "__main__":
    test_connection()
