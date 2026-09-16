# build_db.py
import sqlite3
import os

db_file = "your_school.db"
conn = sqlite3.connect(db_file)
cur = conn.cursor()

# 创建院校表，新增 school_level、year 字段，保留原id主键
cur.execute('''
CREATE TABLE IF NOT EXISTS graduate_school (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    school_name TEXT NOT NULL,
    major TEXT NOT NULL,
    degree_type TEXT NOT NULL,
    re_line INTEGER,
    recruit_num INTEGER,
    province TEXT,
    school_level TEXT,
    year INTEGER
)
''')

# 插入测试样例数据，补充新增字段的值
test_data = [
    ["昆明理工大学","控制科学与工程","学硕",295,18,"云南","双非",2025],
    ["云南大学","控制科学与工程","学硕",305,12,"云南","211",2025],
    ["贵州大学","控制科学与工程","学硕",290,14,"贵州","211",2025]
]
cur.executemany('''
INSERT INTO graduate_school
(school_name,major,degree_type,re_line,recruit_num,province,school_level,year)
VALUES (?,?,?,?,?,?,?,?)
''', test_data)

conn.commit()
conn.close()
print(f"{db_file} 创建完成，已写入测试数据（包含school_level、year字段）")
