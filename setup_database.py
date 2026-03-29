import pymysql
import os

# 数据库连接配置
db_config = {
    'host': 'localhost',
    'user': 'Narcisuss',
    'password': '688376',
    'charset': 'utf8mb4'
}

def execute_sql_file(sql_file):
    """执行SQL文件"""
    try:
        # 连接到MySQL服务器
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        
        # 读取SQL文件
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # 执行SQL脚本
        for statement in sql_script.split(';'):
            statement = statement.strip()
            if statement:
                cursor.execute(statement)
        
        # 提交事务
        conn.commit()
        print("SQL文件执行成功！")
        
    except Exception as e:
        print(f"执行SQL文件时出错: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    sql_file = 'flower_classification.sql'
    execute_sql_file(sql_file)
