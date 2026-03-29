-- 创建数据库
CREATE DATABASE IF NOT EXISTS flower_classification;

-- 使用数据库
USE flower_classification;

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建历史记录表
CREATE TABLE IF NOT EXISTS history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    user_id INT DEFAULT NULL,
    filename VARCHAR(255) NOT NULL,
    class_name VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- 创建纠正记录表
CREATE TABLE IF NOT EXISTS corrections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    predicted_class VARCHAR(100) NOT NULL,
    correct_class VARCHAR(100) NOT NULL,
    user_id INT DEFAULT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

