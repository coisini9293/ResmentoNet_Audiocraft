"""
数据库模型定义（MySQL + PyMySQL）
"""

import pymysql
from typing import Dict, List, Optional, Union
import os
import json


class DatabaseManager:
    """数据库管理器（基于 PyMySQL）"""

    def __init__(self):
        # 连接配置（可用环境变量覆盖）
        self.host = os.getenv("DB_HOST", "127.0.0.1")
        self.port = int(os.getenv("DB_PORT", "3306"))
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "coisini9293")
        self.db_name = os.getenv("DB_NAME", "music_app")
        self._init_database()

    def _conn_server(self):
        return pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, autocommit=True, charset="utf8mb4")

    def _conn_db(self):
        return pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, database=self.db_name, autocommit=True, charset="utf8mb4")

    def _init_database(self):
        # 建库
        with self._conn_server() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;")
        # 建表（最小集：users/emotion_history/music_history/system_logs）
        with self._conn_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      username VARCHAR(64) NOT NULL UNIQUE,
                      password_hash VARCHAR(255) NOT NULL,
                      email VARCHAR(128) NULL,
                      is_admin TINYINT(1) NOT NULL DEFAULT 0,
                      is_active TINYINT(1) NOT NULL DEFAULT 1,
                      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      last_login DATETIME NULL
                    ) ENGINE=InnoDB;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS emotion_history (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      user_id INT NOT NULL,
                      image_path TEXT,
                      detected_emotion VARCHAR(16),
                      confidence_score FLOAT,
                      timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                      KEY idx_emotion_user (user_id, timestamp)
                    ) ENGINE=InnoDB;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS music_history (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      user_id INT NOT NULL,
                      emotion_history_id INT NULL,
                      music_path TEXT,
                      music_filename VARCHAR(255),
                      target_emotion VARCHAR(16),
                      prompt TEXT,
                      generation_time FLOAT,
                      timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                      FOREIGN KEY (emotion_history_id) REFERENCES emotion_history(id) ON DELETE SET NULL,
                      KEY idx_music_user (user_id, timestamp)
                    ) ENGINE=InnoDB;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS system_logs (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      user_id INT NULL,
                      action VARCHAR(64),
                      details TEXT,
                      status VARCHAR(16),
                      timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                      KEY idx_logs_user (user_id, timestamp)
                    ) ENGINE=InnoDB;
                    """
                )
                # 新增：sessions / events / feedbacks
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      user_id INT NOT NULL,
                      chain_type VARCHAR(32) NOT NULL DEFAULT 'web',
                      source VARCHAR(32) NOT NULL DEFAULT 'web',
                      started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                      KEY idx_sessions_user (user_id, started_at)
                    ) ENGINE=InnoDB;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      session_id INT NOT NULL,
                      user_id INT NULL,
                      agent VARCHAR(64) NULL,
                      event_type VARCHAR(64) NOT NULL,
                      payload_json TEXT,
                      ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                      KEY idx_events_session (session_id, ts)
                    ) ENGINE=InnoDB;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedbacks (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      generation_id INT NOT NULL,
                      user_id INT NOT NULL,
                      score INT NULL,
                      vote VARCHAR(8) NULL,
                      comment TEXT NULL,
                      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (generation_id) REFERENCES music_history(id) ON DELETE CASCADE,
                      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                      KEY idx_feedbacks_gen (generation_id, created_at)
                    ) ENGINE=InnoDB;
                    """
                )

    def add_user(self, username: str, password_hash: str, email: str = None) -> bool:
        """
        添加新用户
        
        Args:
            username: 用户名
            password_hash: 密码哈希
            email: 邮箱
            
        Returns:
            bool: 是否成功
        """
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (username, password_hash, email) VALUES (%s, %s, %s)",
                        (username, password_hash, email)
                    )
            return True
        except pymysql.err.IntegrityError:
            return False
        except Exception as e:
            print(f"添加用户失败: {e}")
            return False
    
    def verify_user(self, username: str, password_hash: str) -> Optional[int]:
        """
        验证用户登录
        
        Args:
            username: 用户名
            password_hash: 密码哈希
            
        Returns:
            Optional[int]: 用户ID，验证失败返回None
        """
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM users WHERE username=%s AND password_hash=%s AND is_active=1",
                        (username, password_hash)
                    )
                    row = cur.fetchone()
                    if row:
                        user_id = row[0]
                        cur.execute("UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=%s", (user_id,))
                        return user_id
            return None
        except Exception as e:
            print(f"验证用户失败: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """
        根据ID获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[Dict]: 用户信息字典
        """
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, username, email, is_admin, created_at, last_login FROM users WHERE id=%s AND is_active=1",
                        (user_id,)
                    )
                    r = cur.fetchone()
            if r:
                return {
                    'id': r[0], 'username': r[1], 'email': r[2], 'is_admin': r[3], 'created_at': r[4], 'last_login': r[5]
                }
            return None
        except Exception as e:
            print(f"获取用户信息失败: {e}")
            return None
    
    def add_emotion_detection(self, user_id: int, image_path: str, 
                            detected_emotion: str, confidence_score: float) -> int:
        """
        添加情绪检测记录
        
        Args:
            user_id: 用户ID
            image_path: 图片路径
            detected_emotion: 检测到的情绪
            confidence_score: 置信度分数
            
        Returns:
            int: 记录ID
        """
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO emotion_history (user_id, image_path, detected_emotion, confidence_score)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (user_id, image_path, detected_emotion, confidence_score)
                    )
                    cur.execute("SELECT LAST_INSERT_ID();")
                    rid = cur.fetchone()[0]
            return int(rid)
        except Exception as e:
            print(f"添加情绪检测记录失败: {e}")
            return -1
    
    def add_music_generation(self, user_id: int, emotion_history_id: int,
                           music_path: str, music_filename: str,
                           target_emotion: str, prompt: str,
                           generation_time: float) -> int:
        """
        添加音乐生成记录
        
        Args:
            user_id: 用户ID
            emotion_history_id: 情绪检测记录ID
            music_path: 音乐文件路径
            music_filename: 音乐文件名
            target_emotion: 目标情绪
            prompt: 生成提示词
            generation_time: 生成时间
            
        Returns:
            int: 记录ID
        """
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO music_history (user_id, emotion_history_id, music_path, music_filename, target_emotion, prompt, generation_time)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (user_id, emotion_history_id, music_path, music_filename, target_emotion, prompt, generation_time)
                    )
                    cur.execute("SELECT LAST_INSERT_ID();")
                    rid = cur.fetchone()[0]
            return int(rid)
        except Exception as e:
            print(f"添加音乐生成记录失败: {e}")
            return -1
    
    def add_system_log(self, user_id: int, action: str, details: str, status: str):
        """
        添加系统日志
        
        Args:
            user_id: 用户ID
            action: 操作类型
            details: 详细信息
            status: 状态（success/error）
        """
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO system_logs (user_id, action, details, status) VALUES (%s, %s, %s, %s)",
                        (user_id, action, details, status)
                    )
        except Exception as e:
            print(f"添加系统日志失败: {e}")
    
    def get_user_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        获取用户历史记录
        
        Args:
            user_id: 用户ID
            limit: 限制记录数量
            
        Returns:
            List[Dict]: 历史记录列表
        """
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    # 包含仅音乐（无情绪记录）与有情绪记录两类
                    cur.execute(
                        """
                        SELECT * FROM (
                          SELECT 
                            eh.id as emotion_id,
                            eh.image_path,
                            eh.detected_emotion,
                            eh.confidence_score,
                            eh.timestamp as emotion_time,
                            mh.id as music_id,
                            mh.music_filename,
                            mh.target_emotion,
                            mh.generation_time,
                            mh.timestamp as music_time
                          FROM emotion_history eh
                          LEFT JOIN music_history mh ON eh.id = mh.emotion_history_id
                          WHERE eh.user_id = %s
                          UNION ALL
                          SELECT 
                            NULL as emotion_id,
                            NULL as image_path,
                            NULL as detected_emotion,
                            NULL as confidence_score,
                            NULL as emotion_time,
                            mh.id as music_id,
                            mh.music_filename,
                            mh.target_emotion,
                            mh.generation_time,
                            mh.timestamp as music_time
                          FROM music_history mh
                          WHERE mh.user_id = %s AND mh.emotion_history_id IS NULL
                        ) t
                        ORDER BY COALESCE(music_time, emotion_time) DESC
                        LIMIT %s
                        """,
                        (user_id, user_id, int(limit))
                    )
                    results = cur.fetchall()
            history = []
            for row in results:
                history.append({
                    'emotion_id': row[0], 'image_path': row[1], 'detected_emotion': row[2],
                    'confidence_score': row[3], 'emotion_time': row[4], 'music_id': row[5],
                    'music_filename': row[6], 'target_emotion': row[7], 'generation_time': row[8], 'music_time': row[9]
                })
            return history
        except Exception as e:
            print(f"获取用户历史记录失败: {e}")
            return []

    # 会话/事件/反馈
    def start_session(self, user_id: int, chain_type: str = "web", source: str = "web") -> int:
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO sessions (user_id, chain_type, source) VALUES (%s, %s, %s)
                        """,
                        (user_id, chain_type, source)
                    )
                    cur.execute("SELECT LAST_INSERT_ID();")
                    rid = cur.fetchone()[0]
            return int(rid)
        except Exception as e:
            print(f"创建会话失败: {e}")
            return -1

    def add_event(self, session_id: int, user_id: Optional[int], agent: Optional[str], event_type: str, payload: Union[str, Dict]) -> int:
        try:
            payload_str = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO events (session_id, user_id, agent, event_type, payload_json)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (session_id, user_id, agent, event_type, payload_str)
                    )
                    cur.execute("SELECT LAST_INSERT_ID();")
                    rid = cur.fetchone()[0]
            return int(rid)
        except Exception as e:
            print(f"添加事件失败: {e}")
            return -1

    def add_feedback(self, generation_id: int, user_id: int, score: Optional[int], vote: Optional[str], comment: Optional[str]) -> int:
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO feedbacks (generation_id, user_id, score, vote, comment)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (generation_id, user_id, score, vote, comment)
                    )
                    cur.execute("SELECT LAST_INSERT_ID();")
                    rid = cur.fetchone()[0]
            return int(rid)
        except Exception as e:
            print(f"添加反馈失败: {e}")
            return -1
    
    def get_system_logs(self, user_id: int = None, limit: int = 50) -> List[Dict]:
        """
        获取系统日志
        
        Args:
            user_id: 用户ID（可选）
            limit: 限制记录数量
            
        Returns:
            List[Dict]: 日志记录列表
        """
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    if user_id:
                        cur.execute(
                            "SELECT id, user_id, action, details, status, timestamp FROM system_logs WHERE user_id=%s ORDER BY timestamp DESC LIMIT %s",
                            (user_id, int(limit))
                        )
                    else:
                        cur.execute(
                            "SELECT id, user_id, action, details, status, timestamp FROM system_logs ORDER BY timestamp DESC LIMIT %s",
                            (int(limit),)
                        )
                    results = cur.fetchall()
            logs = []
            for row in results:
                logs.append({
                    'id': row[0], 'user_id': row[1], 'action': row[2], 'details': row[3], 'status': row[4], 'timestamp': row[5]
                })
            return logs
        except Exception as e:
            print(f"获取系统日志失败: {e}")
            return []

    # 新增：系统日志筛选/分页
    def query_system_logs(self,
                          user_id: Optional[int] = None,
                          actions: Optional[List[str]] = None,
                          status: Optional[List[str]] = None,
                          start: Optional[str] = None,  # 'YYYY-MM-DD'
                          end: Optional[str] = None,
                          keyword: Optional[str] = None,
                          limit: int = 100,
                          offset: int = 0) -> List[Dict]:
        try:
            clauses = []
            params: List = []
            if user_id:
                clauses.append("user_id=%s")
                params.append(user_id)
            if actions:
                placeholders = ",".join(["%s"] * len(actions))
                clauses.append(f"action IN ({placeholders})")
                params.extend(actions)
            if status:
                placeholders = ",".join(["%s"] * len(status))
                clauses.append(f"status IN ({placeholders})")
                params.extend(status)
            if start:
                clauses.append("DATE(timestamp) >= %s")
                params.append(start)
            if end:
                clauses.append("DATE(timestamp) <= %s")
                params.append(end)
            if keyword:
                clauses.append("details LIKE %s")
                params.append(f"%{keyword}%")
            where = ("WHERE "+" AND ".join(clauses)) if clauses else ""
            sql = f"SELECT id, user_id, action, details, status, timestamp FROM system_logs {where} ORDER BY timestamp DESC LIMIT %s OFFSET %s"
            params.extend([int(limit), int(offset)])
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
            return [
                {'id': r[0], 'user_id': r[1], 'action': r[2], 'details': r[3], 'status': r[4], 'timestamp': r[5]}
                for r in rows
            ]
        except Exception as e:
            print(f"筛选系统日志失败: {e}")
            return []

    # 简易表浏览（只允许白名单表名）
    _BROWSE_TABLES = {"users","emotion_history","music_history","system_logs","sessions","events","feedbacks"}

    def fetch_table(self, table: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        table = table.strip()
        if table not in self._BROWSE_TABLES:
            return []
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM {table} ORDER BY 1 DESC LIMIT %s OFFSET %s", (int(limit), int(offset)))
                    cols = [c[0] for c in cur.description]
                    rows = cur.fetchall()
            return [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            print(f"读取表 {table} 失败: {e}")
            return []

    def count_table(self, table: str) -> int:
        table = table.strip()
        if table not in self._BROWSE_TABLES:
            return 0
        try:
            with self._conn_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(1) FROM {table}")
                    (cnt,) = cur.fetchone()
            return int(cnt)
        except Exception as e:
            print(f"统计表 {table} 失败: {e}")
            return 0

# 创建全局数据库管理器实例
db_manager = DatabaseManager() 