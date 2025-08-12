from __future__ import annotations

import os
from typing import Any, Dict, Optional

import sqlalchemy as sa
from sqlalchemy import text


def get_default_engine() -> sa.Engine: # 定义一个函数，用于创建默认的 MySQL 引擎
    """根据环境变量创建默认的 MySQL 引擎。
    需要设置：DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, 可选 DB_PORT。
    """
    host = os.getenv("DB_HOST", "127.0.0.1")
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    name = os.getenv("DB_NAME", "music_app")
    port = int(os.getenv("DB_PORT", "3306"))
    url = sa.engine.URL.create(
        "mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=port,
        database=name,
    )
    return sa.create_engine(url, pool_pre_ping=True, pool_recycle=3600)


class MySQLRepository: # 定义一个类，用于存储 MySQL 引擎
    def __init__(self, engine: sa.Engine): # 初始化方法，用于初始化 MySQL 引擎
        self.engine = engine

    # 偏好读取
    def get_user_preferences(self, user_id: int) -> Dict[str, Any]: # 定义一个函数，用于读取用户偏好
        with self.engine.begin() as conn:
            row = conn.execute(
                text("""
                SELECT fav_genres, fav_instruments, bpm_min, bpm_max, negative_prompts
                FROM user_preferences WHERE user_id=:uid
                """),
                {"uid": user_id},
            ).mappings().first()
        return dict(row) if row else {}

    # 会话
    def create_session(self, user_id: int, source: str = "web") -> Optional[int]: # 定义一个函数，用于创建会话
        with self.engine.begin() as conn:
            res = conn.execute(
                text("""
                INSERT INTO sessions (user_id, started_at, source)
                VALUES (:uid, NOW(), :src)
                """),
                {"uid": user_id, "src": source},
            )
            sid = res.lastrowid
        return sid

    # 情绪记录
    def save_emotion(self, session_id: int, emotion: Dict[str, Any]) -> None: # 定义一个函数，用于保存情绪记录
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO emotions (session_id, emotion, confidence, extra, created_at)
                VALUES (:sid, :e, :c, :x, NOW())
                """),
                {
                    "sid": session_id,
                    "e": emotion.get("emotion"),
                    "c": float(emotion.get("confidence", 0.0)),
                    "x": sa.text("JSON_OBJECT()") if not emotion.get("extra") else str(emotion.get("extra")),
                },
            )

    # 生成记录
    def save_generation( # 定义一个函数，用于保存生成记录
        self,
        session_id: int,
        prompt_initial: str,
        prompt_personalized: str,
        model: str,
        params_json: Dict[str, Any],
        wav_path: Optional[str],
        video_path: Optional[str],
        duration: int,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO generations (session_id, prompt_initial, prompt_personalized, model, params_json, wav_path, video_path, duration_sec, created_at)
                VALUES (:sid, :pi, :pp, :m, :pj, :wp, :vp, :dur, NOW())
                """),
                {
                    "sid": session_id,
                    "pi": prompt_initial,
                    "pp": prompt_personalized,
                    "m": model,
                    "pj": str(params_json),
                    "wp": wav_path,
                    "vp": video_path,
                    "dur": duration,
                },
            )


