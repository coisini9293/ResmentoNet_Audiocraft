## 情绪音乐助手（Emotion Music Assistant）

一个面向情绪调节与创意音乐生成的多智能体应用：
- 通过面部图像/视频自动识别情绪（angry / disgust / fear / sad / neutral），或跳过识别直接文生音乐
- 基于 LLM 生成“安抚/舒缓导向”的音乐描述，并融合用户偏好个性化优化
- 调用 Audiocraft MusicGen 生成可下载的音频/可选波形视频
- 全链路写入 MySQL，提供历史记录、系统日志、打分与运维面板

运行形态：
- Web 应用：Streamlit 富界面（登录、情绪检测、直接生成、历史、日志、下载与评分）
- 编程接口：Python API 一行调用多智能体编排
- 命令行：简易端到端验证脚本


## 功能总览

- 情绪检测与预热
  - 图片/视频/摄像头输入；视频默认取第一帧
  - ResEmoteNet（优先）或本地 ResNet18 回退；显示情绪与置信度
  - 支持“预热面部模型”避免首帧等待

- 文本到音乐（MusicGen）
  - 模型选择：`facebook/musicgen-*` 与 `*-stereo-*` 系列
  - 采样参数：`duration/topk/topp/temperature/cfg_coef`，可切换解码器 `Default/MultiBand_Diffusion`
  - 生成 WAV 与（可选）波形视频，支持一键下载

- 多智能体编排（agents）
  - Agent1：情绪识别（工具函数）
  - Agent3：基于情绪的“舒缓导向”初稿提示词（Moonshot OpenAI 兼容）
  - Agent5：合并用户自定义描述与初稿，保持舒缓导向
  - Agent4：融合数据库用户偏好，输出最终提示词与参数建议
  - Agent2：调用 MusicGen 生成音频/视频

- 个性化与反馈
  - 用户偏好：风格/乐器/BPM 范围/负面提示词
  - 历史记录：含“仅情绪/仅音乐/情绪+音乐”多类型聚合
  - 评分与投票：星级与 like/dislike，写入反馈表

- 运维与可观测性
  - GPU 显存条（自动刷新），生成时暂停刷新提升稳定性
  - 系统日志筛选：按操作、状态、日期、关键词，分页查看
  - 数据表浏览白名单：`users/emotion_history/music_history/system_logs/sessions/events/feedbacks`


## 目录结构（关键）

```
Alex项目/
  agents/
    multi_agent.py            # 多智能体编排与对外 API（run_multi_agent/submit/GenerationParams）
  tools/
    face_tool.py              # 面部情绪识别（ResEmoteNet，回退 ResNet18）
    musicgen_tool.py          # 封装 Audiocraft MusicGen 的 predict_full
    mysql_repo.py             # 可选：SQLAlchemy 版仓库（与多智能体集成）
  web_app/
    main.py                   # Streamlit Web 主应用（登录/检测/生成/历史/日志）
    run_app.py                # 一键安装依赖并启动（支持本地/局域网/Cloudflare Tunnel）
    agents/                   # 示例情绪/音乐推荐 Agent（LangChain）
    auth.py                   # 登录注册/会话状态
    database/models.py        # 基于 PyMySQL 的 DB 管理器（自动建库建表）
    requirements.txt          # Web 应用依赖
  audiocraft-main/            # MusicGen 相关代码与 demos（predict_full 入口）
  ResEmoteNet/                # 情绪识别模型仓库与说明
  tests/test_multi_agent.py   # 端到端验证脚本（非 pytest）
  web_uploads/                # Web 上传素材目录（运行时生成）
```


## 快速开始（Windows 友好）

### 环境准备
- Python 3.10（推荐）
- GPU 可选；若使用 CUDA，请保证 `torch/torchvision/torchaudio` 版本与 CUDA 构建一致（如 cu121）

### 安装依赖
建议分模块安装（可在同一 venv 内）：

1) Web 应用
```
cd web_app
pip install -r requirements.txt
```

2) ResEmoteNet（用于人脸情绪识别，按需）
```
cd ResEmoteNet
pip install -r requirements.txt
```

3) Audiocraft（随仓库提供，首次运行会按需下载权重/依赖）
- 若生成阶段出现依赖缺失，请根据报错补装 `torch/torchaudio/transformers` 等

### 必备环境变量
将以下变量设置为系统“用户变量”（PowerShell 示例）：

```
setx OPENAI_API_KEY "你的Moonshot或OpenAI兼容Key"
setx OPENAI_BASE_URL "https://api.moonshot.cn/v1"

# MySQL（如无需落库可跳过，Web 应用默认会本地建库建表）
setx DB_HOST 127.0.0.1
setx DB_PORT 3306
setx DB_USER root
setx DB_PASSWORD 你的密码
setx DB_NAME music_app

# 面部模型权重（强烈建议设置，避免找不到默认权重）
setx FACE_CKPT "E:\\path\\to\\fer2013_model.pth"

# Windows 中文路径兼容（ASCII 安全目录，默认已内置）
setx ASCII_TMP "C:\\Windows\\Temp"
setx RESULT_DIR "C:\\Windows\\Temp\\emotion_music_result"
```

可选缓存目录：`HF_HOME/TRANSFORMERS_CACHE/TORCH_HOME`（若磁盘/权限受限，建议设置到 ASCII 路径）。

### 启动 Web 应用

方法一（推荐，一键安装并启动）：
```
cd web_app
python run_app.py             # 本机访问  http://127.0.0.1:8501
# 或
python run_app.py --public    # 局域网访问（0.0.0.0）
# 或
python run_app.py --tunnel    # 通过 Cloudflare Tunnel 暴露临时公网 URL（需已安装 cloudflared）
```

方法二（直接启动）：
```
cd web_app
streamlit run main.py
```

### 命令行端到端验证
```
python tests/test_multi_agent.py --media_path E:\\path\\to\\face.jpg --duration 10
```


## 使用指南（Web 页面）

- 登录/注册：首次使用可注册账号；登录后侧边栏展示用户信息；管理员标记显示“系统日志”入口
- 情绪检测：
  - 上传图片/视频或使用摄像头，点击“开始检测”获得情绪/置信度
  - 支持“初始化面部模型”预热
  - 可设置 MusicGen 参数与个性化偏好（风格/乐器/BPM/负面提示）
- 生成音乐：
  - 从情绪检测结果或手动选择情绪出发；也可直接粘贴完整英文 Prompt
  - 生成完成后展示“初稿/个性化”提示词，提供 WAV/视频播放与下载
  - 简易规则打分：基于音频长度/BPM/动态范围与情绪-风格一致性给出 0~1 分
- 直接生成：
  - 跳过情绪识别，直接输入音乐描述进行生成
- 历史记录：
  - 聚合“情绪+音乐/仅音乐/仅情绪”，支持分页
- 系统日志（管理员）：
  - 按操作/状态/日期/关键词筛选查看，分页展示；可浏览白名单表的数据明细


## 编程接口（Python API）

### 多智能体统一入口
```python
from agents.multi_agent import run_multi_agent, GenerationParams

res = run_multi_agent(
    media_path=None,              # 可为图片/视频路径，None 表示跳过情绪识别
    user_id=1,                    # 可选，用于个性化与落库
    gen_params=GenerationParams(
        duration=15,
        topk=250, topp=0.0, temperature=1.0, cfg_coef=3.0,
        model="facebook/musicgen-stereo-large",
        decoder="Default",       # 或 "MultiBand_Diffusion"
    ),
    override_emotion=None,        # 可直接指定情绪（覆盖检测）
    user_prefs_override={         # Web 同款个性化字段（可选）
        "fav_genres": ["lofi","ambient"],
        "fav_instruments": ["piano","warm pads"],
        "bpm_min": 60, "bpm_max": 90,
        "negative_prompts": "vocals, harsh"
    },
    manual_prompt=None,           # 直接提供完整英文描述（可选）
)
```

返回值结构（示例）：
```json
{
  "emotion": {"emotion": "sad", "confidence": 0.82, "extra": {"classes": 4}},
  "prompt_initial": "...",
  "prompt_final": "...",
  "params": {
    "duration": 15, "topk": 250, "topp": 0.0, "temperature": 1.0,
    "cfg_coef": 3.0, "model": "facebook/musicgen-stereo-large",
    "model_path": "", "decoder": "Default"
  },
  "outputs": {"wav": "C:/.../audio_*.wav", "video": "C:/.../video_*.mp4"}
}
```

`GenerationParams` 字段：
- `duration`：秒数（5~120）
- `topk`/`topp`/`temperature`/`cfg_coef`：MusicGen 采样与 CFG 参数
- `model`/`model_path`：模型名与本地路径（通常只需要模型名）
- `decoder`：`Default` 或 `MultiBand_Diffusion`

### 工具层 API
- 面部情绪：`tools.face_tool.detect_emotion(media_path) -> {emotion, confidence, extra}`
- 音乐生成：`tools.musicgen_tool.generate_music_files(...) -> {wav, video, wav_mbd, video_mbd}`


## 数据库存储（MySQL）

web 端默认使用 `web_app/database/models.py`（PyMySQL）自动建库建表：
- `users`：用户信息与权限
- `emotion_history`：情绪识别记录
- `music_history`：音乐生成记录
- `system_logs`：系统操作日志
- `sessions`/`events`/`feedbacks`：会话、事件、用户反馈

多智能体侧也提供 `tools/mysql_repo.py`（SQLAlchemy 版本）用于可选的落库与偏好读取。

初始化方式：首次运行自动创建库表；生产建议根据 `mysql部分.md` 与 `多智能体部分.md` 的 DDL 建表并制定索引/权限策略。


## 常见问题（FAQ）

- Windows 中文路径/编码错误
  - 已在应用与工具层强制使用 ASCII 安全临时/结果目录；仍建议设置 `ASCII_TMP` 与 `RESULT_DIR` 到英文路径

- 找不到面部模型权重
  - 设置 `FACE_CKPT` 指向你的权重文件，如 `fer2013_model.pth`
  - 若失败将回退使用本地 ResNet18 方案（需要项目内 `Face` 目录权重）

- LLM 连接失败
  - 设置 `OPENAI_API_KEY`，并将 `OPENAI_BASE_URL` 设为 `https://api.moonshot.cn/v1`

- CUDA/torchaudio DLL 报错
  - 确保 `torch/torchvision/torchaudio` 版本与 CUDA 构建一致；同源安装

- MySQL 连接/权限问题
  - 确认环境变量、账号权限与端口；web 端默认会建库建表


## 路线图与改进建议

- 将 MusicGen 依赖从 demo 抽象为纯后端库函数，去除 UI 依赖
- 视频输入多帧/投票情绪识别，输出时间段与主导情绪
- 用户画像/偏好完善：自动从反馈中学习更新
- 生成服务进程化（队列/RPC），提升并发与稳定性
- 加入单元测试与 CI（当前 `tests/test_multi_agent.py` 为手动验证脚本）


## 许可证与致谢

- 本仓库整合与复用：ResEmoteNet（MIT）、Audiocraft（Meta）等开源项目，版权与许可遵循对应子项目
- 本项目仅供学习与研究使用


## 致开发者的提示

- 重要环境变量：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`FACE_CKPT`、`DB_*`、`ASCII_TMP`、`RESULT_DIR`
- 可观测性：生成完成后会写入规则分数事件；管理员可在“系统日志”检索
- 如需二次开发：从 `agents/multi_agent.py` 与 `web_app/main.py` 入手，遵循已有模块拆分


