## MySQL 使用说明（本项目）

本文解释：MySQL 在本项目里如何“实现存储”、具体存哪些内容、以什么形式保存、以及如何查看与修改。

### 1. 连接与配置

- 代码位置：`tools/mysql_repo.py`
- 连接使用 SQLAlchemy，连接参数来自环境变量：
  - `DB_HOST`（默认 127.0.0.1）
  - `DB_PORT`（默认 3306）
  - `DB_USER`（默认 root）
  - `DB_PASSWORD`（默认空）
  - `DB_NAME`（默认 music_app）

示例（PowerShell 永久设置）：

```powershell
setx DB_HOST 127.0.0.1
setx DB_PORT 3306
setx DB_USER user
setx DB_PASSWORD pass
setx DB_NAME music_app
```

### 2. 存储是如何实现的（代码层）

- 入口：多智能体完成每一步后会调用 `MySQLRepository` 的方法写库（若环境变量未配置/连接失败，会跳过落库）。
- 代码要点：
  - `get_default_engine()`：按环境变量创建 Engine
  - `create_session(user_id, source)`：创建一次交互会话
  - `save_emotion(session_id, emotion)`：写入面部情绪识别结果
  - `save_generation(...)`：写入音乐生成记录（提示词、参数、文件路径）

你可以在 `agents/multi_agent.py` 中看到落库调用的实际位置（生成完成后尝试写入 DB）。

### 3. 存储的内容、位置与形式

推荐的最小表集合如下（可直接使用下述 DDL 初始化）。大型文件（音频/视频）存在磁盘目录，仅将相对路径与元数据写入 MySQL。

```sql
-- 建库与字符集
CREATE DATABASE IF NOT EXISTS music_app CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE music_app;

-- 用户偏好（可选，供个性化提示词使用）
CREATE TABLE IF NOT EXISTS user_preferences (
  user_id INT PRIMARY KEY,
  fav_genres JSON NULL,
  fav_instruments JSON NULL,
  bpm_min INT NULL,
  bpm_max INT NULL,
  negative_prompts TEXT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 会话（一次从情绪识别到生成的完整流程）
CREATE TABLE IF NOT EXISTS sessions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  source VARCHAR(32) DEFAULT 'web'
) ENGINE=InnoDB;

-- 情绪识别结果
CREATE TABLE IF NOT EXISTS emotions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT NOT NULL,
  emotion ENUM('sad','disgust','angry','fear','neutral') NOT NULL,
  confidence FLOAT NOT NULL,
  extra JSON NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_emotions_session_id (session_id)
) ENGINE=InnoDB;

-- 生成记录
CREATE TABLE IF NOT EXISTS generations (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT NOT NULL,
  prompt_initial TEXT NOT NULL,
  prompt_personalized TEXT NOT NULL,
  model VARCHAR(128) NOT NULL,
  params_json JSON NULL,
  duration_sec INT NOT NULL,
  wav_path TEXT NULL,
  video_path TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_generations_session_id (session_id)
) ENGINE=InnoDB;

-- 用户反馈（可选）
CREATE TABLE IF NOT EXISTS feedbacks (
  id INT AUTO_INCREMENT PRIMARY KEY,
  generation_id INT NOT NULL,
  user_id INT NULL,
  score TINYINT NULL,
  vote ENUM('up','down') NULL,
  comment TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_feedbacks_generation_id (generation_id)
) ENGINE=InnoDB;
```

文件存放建议：

- 将音/视频输出保存在 `generated_music/` 或 `data/{user_id}/{session_id}/` 目录下；
- MySQL 只保存相对路径（如 `generated_music/2025-01-01/abc.wav`）与必要元信息（时长、模型、参数）。

### 4. 如何查看与修改（多种方式）

1) 图形客户端：

   - MySQL Workbench、DBeaver、TablePlus。连接你的 MySQL，选择 `music_app` 数据库，直接浏览/编辑表数据。
2) 命令行：

```bash
mysql -h 127.0.0.1 -u user -p
SHOW DATABASES;        -- 查看数据库
USE music_app;         -- 切换数据库
SHOW TABLES;           -- 查看表
DESCRIBE generations;  -- 查看表结构
SELECT * FROM generations ORDER BY id DESC LIMIT 20;   -- 查最近生成记录
UPDATE user_preferences SET bpm_min=80,bpm_max=120 WHERE user_id=1;  -- 修改偏好
DELETE FROM emotions WHERE id=123;  -- 删除一条情绪记录
```

3) 代码方式（SQLAlchemy 原生执行示例）：

```python
from tools.mysql_repo import get_default_engine
from sqlalchemy import text

engine = get_default_engine()
with engine.begin() as conn:
    rows = conn.execute(text("SELECT id, model, wav_path FROM generations ORDER BY id DESC LIMIT 5")).mappings().all()
    print(rows)
    conn.execute(text("UPDATE user_preferences SET bpm_min=:a, bpm_max=:b WHERE user_id=:u"), {"a": 80, "b": 120, "u": 1})
```

### 5. 初始化与权限

初次使用建议：

```sql
CREATE DATABASE music_app CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
CREATE USER 'user'@'%' IDENTIFIED BY 'pass';
GRANT ALL PRIVILEGES ON music_app.* TO 'user'@'%';
FLUSH PRIVILEGES;
```

然后执行本文件第 3 节的建表 DDL。

### 6. 备份与恢复（简要）

- 备份：`mysqldump -h 127.0.0.1 -u user -p music_app > backup.sql`
- 恢复：`mysql -h 127.0.0.1 -u user -p music_app < backup.sql`

### 7. 注意事项与最佳实践

- 字符集统一 `utf8mb4` 以支持全字符；
- 大文件不要入库，只存路径；
- 为高频查询加索引（如 `session_id`）；
- 生产建议使用迁移工具（Alembic）管理 DDL 版本；
- 数据敏感时谨慎收集并设置清理周期（可在 DB 增加保留期字段或后台清理任务）。

如需要把更多数据（如素材、事件日志）落库，可在现有表的基础上按需扩展（例如增加 `assets`、`events` 表）。




### 目标与范围

- 为你的多智能体应用（面部情绪识别 → 提示词初稿 → 个性化提示词 → 文生音乐；或“跳过情绪识别直生音乐”）设计一套覆盖全链路的数据模型。
- 支持用户登录与权限（管理员可见系统日志）、用户偏好、素材与产物路径、参数可追溯、评分反馈、事件日志与审计、分页查询。

### 必须存储的内容（按模块）

- 用户与权限

  - 账户信息（用户名、密码哈希、邮箱）
  - 角色（是否管理员 is_admin）
  - 用户画像与偏好（喜欢的风格/乐器、BPM 范围、负面提示词等）
- 会话与流程执行

  - 一次“生成流程”的起止（谁、何时、来源、链路类型：emotion_chain/direct_chain）
- 素材与产物

  - 用户上传的图片/视频/旋律文件路径与元数据（MIME、md5）
  - 生成的 wav、视频波形路径（物理文件落磁盘/对象存储，DB 存相对路径+元数据）
- 情绪识别结果

  - 情绪枚举（angry/disgust/fear/sad/neutral…），置信度、模型版本、类别数（4/7）、回退标记、原图路径等
- LLM 与提示词

  - 初稿提示词（Agent3）、个性化提示词（Agent4）
  - 使用的 LLM 模型、温度等（可纳入 params_json 或单独表）
- 生成记录

  - 使用的 MusicGen 模型、解码器、关键参数（duration、top_k、top_p、temperature、cfg_coef）
  - 生成音频/视频的路径、耗时、状态
- 用户反馈

  - 评分、投票、评论，用于后续个性化与评估
- 系统事件/日志（管理员可见）

  - 智能体事件、异常、链路耗时、输入输出摘要（尽量脱敏）

### 建表设计（MySQL 8+，utf8mb4）

可以直接用以下 DDL 初始化（必要字段尽量规范化，复杂可变信息放 JSON）：

```sql
CREATE DATABASE IF NOT EXISTS music_app CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE music_app;

-- 用户
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(64) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  email VARCHAR(128) NULL,
  is_admin TINYINT(1) NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 用户画像/偏好（个性化用）
CREATE TABLE user_preferences (
  user_id INT PRIMARY KEY,
  fav_genres JSON NULL,
  fav_instruments JSON NULL,
  bpm_min INT NULL,
  bpm_max INT NULL,
  negative_prompts TEXT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_pref_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 会话（一次完整交互：情绪识别→生成 或 直接生成）
CREATE TABLE sessions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  chain_type ENUM('emotion_chain','direct_chain') NOT NULL,
  source VARCHAR(32) DEFAULT 'web',
  started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_sess_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  KEY idx_sessions_user (user_id, started_at)
) ENGINE=InnoDB;

-- 素材与产物（文件落磁盘，DB 存相对路径+元数据）
CREATE TABLE assets (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT NOT NULL,
  type ENUM('image','video','melody','wav','video_waveform') NOT NULL,
  path TEXT NOT NULL,
  mime VARCHAR(64) NULL,
  md5 CHAR(32) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_asset_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  KEY idx_assets_session (session_id, type, created_at)
) ENGINE=InnoDB;

-- 情绪识别（可多条，如多图/多次识别）
CREATE TABLE emotions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT NOT NULL,
  emotion ENUM('angry','disgust','fear','sad','happy','surprise','neutral') NOT NULL,
  confidence FLOAT NOT NULL,
  classes INT NOT NULL DEFAULT 4,                   -- 4类/7类等
  model_name VARCHAR(128) NULL,                     -- ResEmoteNet / ResNet18...
  fallback TINYINT(1) NOT NULL DEFAULT 0,           -- 是否回退模型
  extra JSON NULL,                                  -- 其他元数据（输入路径、索引等）
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_emotion_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  KEY idx_emotions_session (session_id, created_at)
) ENGINE=InnoDB;

-- 生成记录（一次文本到音乐）
CREATE TABLE generations (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT NOT NULL,
  model VARCHAR(128) NOT NULL,                      -- musicgen 模型名
  decoder ENUM('Default','MultiBand_Diffusion') NOT NULL DEFAULT 'Default',
  duration_sec INT NOT NULL,
  params_json JSON NULL,                            -- top_k/top_p/temperature/cfg_coef/LLM信息等
  prompt_initial TEXT NULL,
  prompt_personalized TEXT NULL,
  wav_path TEXT NULL,
  video_path TEXT NULL,
  status ENUM('success','failed','partial') DEFAULT 'success',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_gen_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  KEY idx_generations_session (session_id, created_at)
) ENGINE=InnoDB;

-- 反馈（多对一 generation）
CREATE TABLE feedbacks (
  id INT AUTO_INCREMENT PRIMARY KEY,
  generation_id INT NOT NULL,
  user_id INT NOT NULL,
  score TINYINT NULL,                               -- 0-5
  vote ENUM('up','down') NULL,
  comment TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_fb_gen FOREIGN KEY (generation_id) REFERENCES generations(id) ON DELETE CASCADE,
  CONSTRAINT fk_fb_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  KEY idx_feedbacks_user (user_id, created_at)
) ENGINE=InnoDB;

-- 系统事件/日志（管理员可见）
CREATE TABLE events (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT NULL,
  user_id INT NULL,
  agent VARCHAR(64) NULL,                           -- 哪个节点/工具
  event_type VARCHAR(64) NOT NULL,                  -- start/end/error/info
  payload_json JSON NULL,                           -- 耗时、错误栈、输入输出摘要(脱敏)
  ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_events_user (user_id, ts),
  KEY idx_events_session (session_id, ts)
) ENGINE=InnoDB;
```

### 关系与查询要点

- `users 1—* sessions 1—* emotions/generations/assets/ events`
- 典型查询
  - 用户历史（分页）：
    - emotions + generations 通过 `session_id` 聚合；前端按“Emotion only / Music only / Emotion + Music”分类展示。
  - 生成详情：
    - `generations` + `assets`(wav/video) + `params_json` 参数审计。
  - 个性化：
    - `user_preferences` 供 Agent4 读取，融合偏好为最终 Prompt。

### 存储策略

- 大文件不入库。目录建议：
  - `data/{user_id}/{session_id}/uploads/*`（图片/视频/旋律）
  - `data/{user_id}/{session_id}/outputs/{timestamp}.wav/.mp4`
- 数据库只存相对路径与基本元数据（mime、md5），便于迁移与 CDN/对象存储切换。

### 分页与索引

- 高频查询字段（`session_id/user_id/created_at`）已加组合索引。
- 历史记录在接口层做分页（limit/offset）或游标分页（ID/时间倒序）。

### 安全与合规

- 账号：仅存 `password_hash`（如 SHA256/BCrypt），不存明文。
- 事件日志：payload 脱敏，避免存入敏感文件路径/Key。
- 数据保留策略：原始素材与生成音频的保留期（如 30/90 天），定期清理 job。
- 用户删除：级联删除其 sessions/emotions/generations/assets/feedbacks（已用 FK CASCADE）。

### 与后端流程的映射

- 情绪链（自动生成）

  1) 创建 `sessions`（chain_type=emotion_chain）
  2) `assets` 写入上传素材
  3) `emotions` 写入识别结果
  4) 生成时合并参数→`generations` 写入（prompt_initial/prompt_personalized、model/decoder/params_json）
  5) 产物路径写入 `generations.wav_path/video_path`；如有波形视频，`assets` 也可同步写入
  6) 用户评分→`feedbacks`
- 直生链（跳过情绪）

  1) 创建 `sessions`（chain_type=direct_chain）
  2) 直接落 `generations`（可有上传旋律 `assets`）
  3) 评分与日志同上

### 文生音乐参数审计（建议纳入 params_json）

- LLM：model、temperature（Agent3/Agent4）
- MusicGen：duration、top_k、top_p、temperature、cfg_coef、model、decoder
- 推理侧：GPU/显存、耗时
- 便于复现与 A/B 测试

### 建设建议

- 迁移与版本：使用 Alembic 管理 DDL 变更。
- 连接参数：环境变量 `DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME`，连接池 `pool_pre_ping=true`。
- 备份恢复：mysqldump 定期备份；日志按天归档。

### 示例：写入一条完整链路（伪代码）

```python
# 创建会话
sid = repo.create_session(user_id, source="web", chain_type="emotion_chain")

# 上传文件
repo.save_asset(sid, type="image", path="data/uid/sid/uploads/xxx.jpg", mime="image/jpeg", md5=...)

# 情绪识别
repo.save_emotion(sid, emotion="angry", confidence=0.93, classes=4,
                  model_name="ResEmoteNet", extra={"path":"...", "classes":4})

# 文生音乐生成
params = {"topk":250,"topp":0.0,"temperature":1.0,"cfg_coef":3.0,"decoder":"Default","llm":{"model":"moonshot-v1-8k"}}
repo.save_generation(sid, prompt_initial="...", prompt_personalized="...", model="facebook/musicgen-stereo-melody",
                     decoder="Default", params_json=params, wav_path="data/.../out.wav", video_path="data/.../out.mp4",
                     duration=15)
```

### 小结

- 表：users / user_preferences / sessions / assets / emotions / generations / feedbacks / events
- 文件：全部落本地/对象存储，DB 存路径
- 审计：params_json 全量记录，支持复现与 A/B
- 历史记录：分页汇总“情绪/音乐/两者”三种类型
- 安全：哈希密码、日志脱敏、文件保留期、管理员可见系统日志

如需，我可把这份文档落到仓库（如 `docs/mysql_schema.md`），并补齐 `repo` 层的 CRUD 方法与分页查询示例。



