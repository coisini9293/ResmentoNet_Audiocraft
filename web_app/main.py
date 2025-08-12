"""
情绪音乐助手 - Streamlit主应用
streamlit run web_app/main.py
"""

import streamlit as st
import sys
import os
import time
from PIL import Image
from pathlib import Path
import tempfile

# 强制 UTF-8 与 ASCII 安全的临时/结果目录，避免 Windows 中文路径导致的 ascii 错误
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
_ASCII_TMP = os.environ.get("ASCII_TMP") or (r"C:\\Windows\\Temp")
for _k in ("TMP", "TEMP", "TMPDIR"):
    os.environ.setdefault(_k, _ASCII_TMP)
if not os.environ.get("RESULT_DIR"):
    try:
        os.makedirs(os.path.join(_ASCII_TMP, "emotion_music_result"), exist_ok=True)
        os.environ["RESULT_DIR"] = os.path.join(_ASCII_TMP, "emotion_music_result")
    except Exception:
        os.environ.setdefault("RESULT_DIR", "result")

# 添加项目根路径，便于导入多智能体与工具
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入自定义模块 / 多智能体
from auth import auth_manager, login_page, register_page, show_user_info, require_login
from database.models import db_manager
from agents.multi_agent import run_multi_agent, GenerationParams, submit
from tools.face_tool import detect_emotion as face_detect
import json
import torch
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None

# 取消中英文切换，统一使用中文

# 简易打分（规则特征 + 情绪-风格一致性）
def compute_rule_score_full(prompt_text: str, target_emotion: str, wav_path: str) -> dict:
    """返回 {metrics:{duration_sec,bpm,crest_db}, alignment:{emotion,reco_genres,hit_score}, rule_score}。失败返回空字典。"""
    try:
        import numpy as np
        import librosa
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        duration_sec = float(librosa.get_duration(y=y, sr=sr))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
        peak = float(np.max(np.abs(y)) + 1e-9)
        rms = float(np.sqrt(np.mean(y**2)) + 1e-9)
        import math
        crest_db = float(20.0 * math.log10(peak / rms))

        def clamp01(x):
            return max(0.0, min(1.0, x))

        score_len = clamp01((duration_sec - 10.0) / (60.0 - 10.0))
        score_bpm = clamp01(1.0 - abs(bpm - 120.0) / 50.0)
        score_crest = clamp01((crest_db - 6.0) / (18.0 - 6.0))
        audio_score = 0.4 * score_len + 0.3 * score_bpm + 0.3 * score_crest

        # 情绪 -> 推荐风格
        emo2genres = {
            "happy": ["pop", "edm", "dance", "funk", "house"],
            "sad": ["lofi", "ambient", "piano", "ballad", "acoustic"],
            "angry": ["rock", "metal", "drum and bass", "industrial"],
            "fear": ["ambient", "cinematic", "dark", "drone"],
            "disgust": ["industrial", "experimental", "noise"],
            "neutral": ["classical", "jazz", "chill", "lofi"],
        }
        emo = (target_emotion or "").lower()
        reco = emo2genres.get(emo, [])
        text_lower = (prompt_text or "").lower()
        hits = sum(1 for g in reco if g in text_lower)
        hit_score = clamp01(hits / max(1, len(reco))) if reco else 0.5

        rule_score = round(float(0.5 * audio_score + 0.5 * hit_score), 3)
        return {
            "metrics": {
                "duration_sec": round(duration_sec, 2),
                "bpm": round(bpm, 1),
                "crest_db": round(crest_db, 2),
            },
            "alignment": {"emotion": emo, "reco_genres": reco, "hit_score": round(hit_score, 3)},
            "rule_score": rule_score,
        }
    except Exception:
        return {}

# 页面配置
st.set_page_config(
    page_title="🎭 情绪音乐助手",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="auto"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .music-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EmotionMusicApp:
    """情绪音乐助手应用"""
    
    def __init__(self):
        """初始化应用"""
        # 上传目录
        self.upload_dir = (PROJECT_ROOT / "web_uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def save_upload(self, uploaded_file) -> str:
        suffix = Path(uploaded_file.name).suffix
        dst = self.upload_dir / f"upload_{int(time.time()*1000)}{suffix}"
        with open(dst, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(dst)

def main():
    """主函数"""
    # 初始化应用
    app = EmotionMusicApp()
    
    # 侧边栏用户信息
    show_user_info()
    
    # 页面路由
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    # 若已登录，强制进入主页面
    from auth import auth_manager
    if auth_manager.is_logged_in() and st.session_state.get('page') == 'login':
        st.session_state['page'] = 'main'
    
    if st.session_state['page'] == 'login':
        login_page()
    elif st.session_state['page'] == 'register':
        register_page()
    elif st.session_state['page'] == 'main':
        main_page(app)
    elif st.session_state['page'] == 'history':
        history_page()
    elif st.session_state['page'] == 'logs':
        logs_page()

def main_page(app):
    """主页面"""
    st.markdown('<h1 class="main-header">🎭 情绪音乐助手</h1>', unsafe_allow_html=True)
    
    # 检查登录状态
    require_login()
    
    # 侧边栏导航
    st.sidebar.title("导航")
    # GPU 显存信息（自动刷新）
    try:
        box = st.sidebar.empty()
        prog_box = st.sidebar.empty()
        def render_vram():
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used = total - free
                def fmt(b):
                    return f"{b/1024/1024/1024:.2f} GB"
                box.caption(f"GPU 显存：已用 {fmt(used)} / 总计 {fmt(total)}，可用 {fmt(free)}")
                try:
                    percent = int((used / max(1, total)) * 100)
                    prog_box.progress(percent, text=f"VRAM 使用率 {percent}%")
                except Exception:
                    pass
            else:
                box.caption("GPU 不可用，使用 CPU 模式")
                prog_box.empty()
        render_vram()
        # 非生成阶段时，每3秒自动刷新一次，提升“实时性”
        if st_autorefresh and not st.session_state.get('is_generating'):
            st_autorefresh(interval=3000, key='auto_vram_refresh')
    except Exception:
        pass
    # 仅管理员可见“系统日志”
    is_admin = st.session_state.get('is_admin', False)
    pages = ["情绪检测", "直接生成", "历史记录"] + (["系统日志"] if is_admin else [])
    page = st.sidebar.selectbox(
        "选择页面",
        pages
    )
    
    if page == "情绪检测":
        emotion_detection_page(app)
    elif page == "直接生成":
        direct_generate_page()
    elif page == "历史记录":
        history_page()
    elif page == "系统日志":
        logs_page()

def emotion_detection_page(app):
    """情绪检测页面"""
    st.header("📸 情绪检测与音乐生成（多智能体）")

    # 初始化面部情绪识别模型（预热）
    col_init1, col_init2 = st.columns([1, 5])
    with col_init1:
        if st.button("初始化面部模型", help="预先加载面部情绪识别模型，避免首次检测等待"):
            try:
                # 延迟导入，避免页面首次加载时就初始化
                from tools.face_tool import _load_rese_model  # type: ignore
                _load_rese_model()
                st.session_state["face_model_warmed"] = True
                st.success("✅ 面部模型已初始化")
            except Exception as e:
                st.error(f"❌ 初始化失败: {e}")

    # 参数设置（参考 musicgen_app.py）
    st.subheader("参数设置（MusicGen）")
    colp1, colp2 = st.columns(2)
    with colp1:
        model = st.selectbox(
            "模型",
            [
                "facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
                "facebook/musicgen-large", "facebook/musicgen-melody-large",
                "facebook/musicgen-stereo-small", "facebook/musicgen-stereo-medium",
                "facebook/musicgen-stereo-melody", "facebook/musicgen-stereo-large",
                "facebook/musicgen-stereo-melody-large",
            ],
            index=8,  # 默认 stereo-large
        )
        decoder = st.radio("解码器", ["Default", "MultiBand_Diffusion"], index=0, horizontal=True)
        duration = st.slider("时长（秒）", min_value=5, max_value=120, value=15, step=5)
    with colp2:
        topk = st.number_input("Top-k", min_value=0, max_value=1000, value=250, step=10)
        topp = st.number_input("Top-p", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f")
        temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1, format="%.2f")
        cfg_coef = st.number_input("Classifier Free Guidance", min_value=0.0, max_value=10.0, value=3.0, step=0.1, format="%.1f")
    
    # 读取已保存状态
    saved_media_path = st.session_state.get("emotion_media_path")
    saved_emotion_result = st.session_state.get("emotion_result")

    # 文件上传
    uploaded_file = st.file_uploader(
        "上传图片或视频进行情绪检测（视频将取第一帧）",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'mp4', 'mov', 'avi', 'mkv'],
        help="支持常见图片/视频格式"
    )
    use_camera = st.checkbox("使用摄像头拍照")
    camera_image_path = None
    if use_camera:
        camera_image = st.camera_input("请拍摄照片")
        if camera_image is not None:
            camera_image_path = app.save_upload(camera_image)
    
    # 处理新上传/摄像头
    if uploaded_file is not None:
        saved_media_path = app.save_upload(uploaded_file)
        st.session_state["emotion_media_path"] = saved_media_path
    if camera_image_path is not None:
        saved_media_path = camera_image_path
        st.session_state["emotion_media_path"] = saved_media_path

    # 预览
    if saved_media_path:
        if Path(saved_media_path).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:
            image = Image.open(saved_media_path)
            st.image(image, caption="输入素材", use_column_width=False, width=480)
        else:
            st.video(saved_media_path)

    # 检测按钮（常驻）
    detect_disabled = not bool(saved_media_path)
    if st.button("🔍 开始检测", type="primary", disabled=detect_disabled):
        with st.spinner("正在检测情绪..."):
            try:
                emotion_result = face_detect(saved_media_path)
                st.session_state["emotion_result"] = emotion_result
                db_manager.add_system_log(user_id=st.session_state['user_id'], action="emotion_detect", details=json.dumps(emotion_result, ensure_ascii=False), status="success")
                st.success("✅ 情绪检测完成！")
                saved_emotion_result = emotion_result
            except Exception as e:
                st.session_state.pop("emotion_result", None)
                db_manager.add_system_log(user_id=st.session_state['user_id'], action="emotion_detect", details=json.dumps({"error": str(e)}, ensure_ascii=False), status="error")
                st.error(f"❌ 情绪检测失败: {e}")

    # 情绪结果卡片（常驻）
    st.subheader("🎭 检测结果")
    if saved_emotion_result:
        st.markdown(f"- **情绪**: {saved_emotion_result.get('emotion','-')}  ")
        st.markdown(f"- **置信度**: {saved_emotion_result.get('confidence',0.0):.2f}")
    else:
        st.info("请进行情绪检测")

    # 个性化偏好与生成（常驻）
    st.subheader("🎚️ 个性化偏好（可选）")
    pref_genres = st.multiselect("喜欢的音乐风格 (可多选)", ["pop","rock","r&b","lofi","edm","classical","jazz","hiphop"], default=[])
    custom_genre = st.text_input("自定义风格（可选，逗号分隔）", value="")
    pref_instruments = st.multiselect("喜欢的乐器 (可多选)", ["piano","guitar","violin","synth","bass","drums","flute"], default=[])
    bpm_min, bpm_max = st.slider("BPM 范围", min_value=60, max_value=180, value=(80, 130))
    negative_prompts = st.text_input("不想要的元素（负面提示）", value="vocals")
    manual_prompt = st.text_area("自定义完整 Prompt（若已检测到情绪则可留空）", height=120, key="manual_prompt_emotion")
    manual_emotion = st.selectbox("或直接选择情绪（可跳过检测）", ["", "sad", "disgust", "angry", "fear", "happy", "neutral"], key="manual_emotion_emotion") or None

    if st.button("🎵 生成音乐", type="secondary"):
        with st.spinner("正在生成音乐..."):
            try:
                st.session_state['is_generating'] = True
                # 会话（sessions）
                user_id_for_session = st.session_state['user_id']
                if 'current_session_id' not in st.session_state:
                    try:
                        sid = db_manager.start_session(user_id_for_session, chain_type="web", source="web")
                    except Exception:
                        sid = -1
                    if isinstance(sid, int) and sid > 0:
                        st.session_state['current_session_id'] = sid
                params = GenerationParams(
                    duration=duration,
                    topk=int(topk), topp=float(topp), temperature=float(temperature), cfg_coef=float(cfg_coef),
                    model=model, decoder=decoder,
                )
                # 合并自定义风格
                all_genres = pref_genres + ([g.strip() for g in custom_genre.split(',') if g.strip()] if custom_genre else [])
                user_prefs_override = {
                    "fav_genres": pref_genres,
                    "fav_instruments": pref_instruments,
                    "bpm_min": bpm_min,
                    "bpm_max": bpm_max,
                    "negative_prompts": negative_prompts,
                }
                # 情绪优先级：已检测 > 手动选择 > 文本 prompt
                override_emotion_eff = None
                manual_prompt_eff = None
                if saved_emotion_result and saved_emotion_result.get("emotion"):
                    override_emotion_eff = saved_emotion_result.get("emotion")
                elif manual_emotion:
                    override_emotion_eff = manual_emotion
                else:
                    manual_prompt_eff = manual_prompt

                # 粗略进度
                prog = st.progress(10, text="准备参数")
                res = run_multi_agent(
                    media_path=st.session_state.get("emotion_media_path"),
                    user_id=st.session_state['user_id'],
                    gen_params=params,
                    override_emotion=override_emotion_eff,
                    user_prefs_override={**user_prefs_override, "fav_genres": all_genres},
                    manual_prompt=manual_prompt_eff,
                )
                prog.progress(80, text="生成完成，整理结果…")
                st.success("✅ 音乐生成完成！")

                # 展示提示词
                with st.expander("初稿提示词 (Agent3)"):
                    st.write(res.get("prompt_initial", ""))
                with st.expander("个性化提示词 (Agent4)"):
                    st.write(res.get("prompt_final", ""))

                # 展示音频/视频
                outs = res.get("outputs", {})
                if outs.get("wav"):
                    st.audio(outs["wav"])
                    try:
                        with open(outs["wav"], "rb") as f:
                            st.download_button("下载音频", data=f, file_name=Path(outs["wav"]).name, mime="audio/wav")
                    except Exception:
                        pass
                    st.code(outs["wav"], language="text")
                if outs.get("video"):
                    st.video(outs["video"])
                    try:
                        with open(outs["video"], "rb") as f:
                            st.download_button("下载视频", data=f, file_name=Path(outs["video"]).name, mime="video/mp4")
                    except Exception:
                        pass
                    st.code(outs["video"], language="text")

                # 写入数据库（若存在）
                user_id = st.session_state['user_id']
                emotion_record_id = -1
                if saved_emotion_result:
                    emotion_record_id = db_manager.add_emotion_detection(
                        user_id=user_id,
                        image_path=Path(st.session_state.get("emotion_media_path") or "").name,
                        detected_emotion=saved_emotion_result.get('emotion',''),
                        confidence_score=float(saved_emotion_result.get('confidence',0.0))
                    )
                gen_record_id = db_manager.add_music_generation(
                    user_id=user_id,
                    emotion_history_id=emotion_record_id if isinstance(emotion_record_id, int) and emotion_record_id>0 else None,
                    music_path=outs.get("wav") or "",
                    music_filename=Path(outs.get("wav") or "result.wav").name,
                    target_emotion=(saved_emotion_result.get('emotion') if saved_emotion_result else (manual_emotion or '')),
                    prompt=res.get("prompt_final", ""),
                    generation_time=duration
                )
                st.session_state['last_generation_id'] = gen_record_id if isinstance(gen_record_id, int) and gen_record_id > 0 else None
                db_manager.add_system_log(
                    user_id=user_id,
                    action="music_generate",
                    details=json.dumps({"wav": outs.get('wav'), "video": outs.get('video')}, ensure_ascii=False),
                    status="success"
                )

                # 简易规则打分并落 events（含情绪-风格一致性）
                if outs.get("wav"):
                    target_em = (saved_emotion_result.get('emotion') if saved_emotion_result else (manual_emotion or ''))
                    metrics = compute_rule_score_full(res.get("prompt_final", ""), target_em, outs["wav"])
                    if metrics:
                        try:
                            session_id = st.session_state.get('current_session_id')
                            if isinstance(session_id, int) and session_id > 0:
                                db_manager.add_event(session_id=session_id, user_id=user_id, agent="scorer", event_type="rule_score", payload=metrics)
                        except Exception:
                            pass

                prog.progress(100, text="完成")

                # 反馈区
                st.markdown("---")
                st.subheader("评价这次生成")
                colf1, colf2 = st.columns([2, 1])
                with colf1:
                    score = st.slider("星级评分", 0, 5, 5)
                    comment = st.text_area("文本评价（可选）", height=100)
                with colf2:
                    vote = st.radio("投票", ["like", "dislike"], index=0, horizontal=True)
                    if st.button("提交评价"):
                        gid = st.session_state.get('last_generation_id')
                        if isinstance(gid, int) and gid > 0:
                            fb_id = db_manager.add_feedback(generation_id=int(gid), user_id=user_id, score=int(score), vote=vote, comment=comment or None)
                            db_manager.add_system_log(user_id=user_id, action="feedback_submit", details=json.dumps({"generation_id": int(gid), "score": int(score), "vote": vote, "comment": comment or None}, ensure_ascii=False), status="success" if (isinstance(fb_id, int) and fb_id>0) else "error")
                            st.success("感谢你的反馈！")
                        else:
                            st.warning("未找到本次生成的记录，无法提交评价。")
            except Exception as e:
                st.error(f"❌ 音乐生成失败: {e}")
                db_manager.add_system_log(
                    user_id=st.session_state['user_id'],
                    action="music_generate",
                    details=f"error={e}",
                    status="error"
                )
            finally:
                st.session_state['is_generating'] = False

def direct_generate_page():
    """直接文生音乐页面"""
    st.header("📝 文本生成音乐（跳过情绪识别）")
    require_login()
    # 参数设置（参考 musicgen_app.py）
    st.subheader("参数设置（MusicGen）")
    colp1, colp2 = st.columns(2)
    with colp1:
        model = st.selectbox(
            "模型",
            [
                "facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
                "facebook/musicgen-large", "facebook/musicgen-melody-large",
                "facebook/musicgen-stereo-small", "facebook/musicgen-stereo-medium",
                "facebook/musicgen-stereo-melody", "facebook/musicgen-stereo-large",
                "facebook/musicgen-stereo-melody-large",
            ],
            index=8,
            key="model_direct",
        )
        decoder = st.radio("解码器", ["Default", "MultiBand_Diffusion"], index=0, horizontal=True, key="decoder_direct")
        duration_only = st.slider("时长（秒）", min_value=5, max_value=120, value=15, step=5, key="duration_only")
    with colp2:
        topk = st.number_input("Top-k", min_value=0, max_value=1000, value=250, step=10, key="topk_direct")
        topp = st.number_input("Top-p", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f", key="topp_direct")
        temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1, format="%.2f", key="temp_direct")
        cfg_coef = st.number_input("Classifier Free Guidance", min_value=0.0, max_value=10.0, value=3.0, step=0.1, format="%.1f", key="cfg_direct")

    manual_prompt_only = st.text_area("音乐描述（英文更稳定，建议包含风格/节奏/乐器/结构/混音/用途/负面提示）", height=160)
    if st.button("🎵 直接生成音乐", type="primary"):
        with st.spinner("正在生成音乐..."):
            try:
                st.session_state['is_generating'] = True
                # 会话
                user_id_for_session = st.session_state['user_id']
                if 'current_session_id' not in st.session_state:
                    try:
                        sid = db_manager.start_session(user_id_for_session, chain_type="web", source="web")
                    except Exception:
                        sid = -1
                    if isinstance(sid, int) and sid > 0:
                        st.session_state['current_session_id'] = sid
                params = GenerationParams(
                    duration=duration_only,
                    topk=int(topk), topp=float(topp), temperature=float(temperature), cfg_coef=float(cfg_coef),
                    model=model, decoder=decoder,
                )
                prog = st.progress(10, text="准备参数")
                res = run_multi_agent(
                    media_path=None,
                    user_id=st.session_state['user_id'],
                    gen_params=params,
                    override_emotion=None,
                    user_prefs_override=None,
                    manual_prompt=manual_prompt_only,
                )
                prog.progress(80, text="生成完成，整理结果…")
                st.success("✅ 音乐生成完成！")
                with st.expander("最终提示词"):
                    st.write(res.get("prompt_final", ""))
                outs = res.get("outputs", {})
                if outs.get("wav"):
                    st.audio(outs["wav"])
                    try:
                        with open(outs["wav"], "rb") as f:
                            st.download_button("下载音频", data=f, file_name=Path(outs["wav"]).name, mime="audio/wav")
                    except Exception:
                        pass
                    st.code(outs["wav"], language="text")
                if outs.get("video"):
                    st.video(outs["video"])
                    try:
                        with open(outs["video"], "rb") as f:
                            st.download_button("下载视频", data=f, file_name=Path(outs["video"]).name, mime="video/mp4")
                    except Exception:
                        pass
                    st.code(outs["video"], language="text")

                # 记录到数据库（仅音乐）
                try:
                    user_id = st.session_state['user_id']
                    gen_record_id = db_manager.add_music_generation(
                        user_id=user_id,
                        emotion_history_id=None,
                        music_path=outs.get("wav") or "",
                        music_filename=Path(outs.get("wav") or "result.wav").name,
                        target_emotion='',
                        prompt=res.get("prompt_final", ""),
                        generation_time=duration_only
                    )
                    st.session_state['last_generation_id'] = gen_record_id if isinstance(gen_record_id, int) and gen_record_id > 0 else None
                    db_manager.add_system_log(user_id=user_id, action="music_generate", details=json.dumps({"wav": outs.get('wav'), "video": outs.get('video')}, ensure_ascii=False), status="success")
                    # 简易规则打分（无目标情绪时按 neutral）
                    if outs.get("wav"):
                        metrics = compute_rule_score_full(res.get("prompt_final", ""), "neutral", outs["wav"])
                        if metrics:
                            session_id = st.session_state.get('current_session_id')
                            if isinstance(session_id, int) and session_id > 0:
                                db_manager.add_event(session_id=session_id, user_id=user_id, agent="scorer", event_type="rule_score", payload=metrics)
                except Exception:
                    pass

                prog.progress(100, text="完成")

                # 反馈区
                st.markdown("---")
                st.subheader("评价这次生成")
                colf1, colf2 = st.columns([2, 1])
                with colf1:
                    score = st.slider("星级评分", 0, 5, 5, key="direct_score")
                    comment = st.text_area("文本评价（可选）", height=100, key="direct_comment")
                with colf2:
                    vote = st.radio("投票", ["like", "dislike"], index=0, horizontal=True, key="direct_vote")
                    if st.button("提交评价", key="submit_direct_feedback"):
                        gid = st.session_state.get('last_generation_id')
                        if isinstance(gid, int) and gid > 0:
                            fb_id = db_manager.add_feedback(generation_id=int(gid), user_id=st.session_state['user_id'], score=int(score), vote=vote, comment=comment or None)
                            db_manager.add_system_log(user_id=st.session_state['user_id'], action="feedback_submit", details=json.dumps({"generation_id": int(gid), "score": int(score), "vote": vote, "comment": comment or None}, ensure_ascii=False), status="success" if (isinstance(fb_id, int) and fb_id>0) else "error")
                            st.success("感谢你的反馈！")
                        else:
                            st.warning("未找到本次生成的记录，无法提交评价。")
            except Exception as e:
                st.error(f"❌ 生成失败: {e}")
            finally:
                st.session_state['is_generating'] = False

def history_page():
    """历史记录页面"""
    st.header("📚 历史记录")
    # 检查登录状态
    require_login()
    
    user_id = st.session_state['user_id']
    
    # 分页控件
    st.subheader("记录列表（分页）")
    page_size = st.selectbox("每页条数", [10, 20, 50, 100], index=1)
    page_num = st.number_input("页码", min_value=1, step=1, value=1)

    # 获取历史记录（尽可能多获取，再做前端分页；若你的 db 支持分页，可替换为服务端分页）
    try:
        history_all = db_manager.get_user_history(user_id, limit=10000)
    except TypeError:
        history_all = db_manager.get_user_history(user_id, limit=1000)

    total = len(history_all) if history_all else 0
    if total == 0:
        st.info("暂无历史记录")
        return

    start = (page_num - 1) * page_size
    end = start + page_size
    if start >= total:
        st.warning("页码超出范围，已重置到第 1 页")
        page_num = 1
        start, end = 0, page_size

    current_page = history_all[start:end]

    st.caption(f"共 {total} 条，当前第 {page_num} 页")

    for record in current_page:
        # 记录类型判定
        did_emotion = bool(record.get('emotion_id'))
        did_music = bool(record.get('music_id'))
        if did_emotion and did_music:
            kind = 'Emotion + Music'
        elif did_music:
            kind = 'Music only'
        elif did_emotion:
            kind = 'Emotion only'
        else:
            kind = 'Unknown'

        title_time = record.get('emotion_time') or record.get('music_time') or ''
        with st.expander(f"[{kind}] {title_time}"):
            col1, col2 = st.columns(2)
            with col1:
                if did_emotion:
                    st.write(f"**检测情绪**: {record.get('detected_emotion','-')}")
                    cs = record.get('confidence_score')
                    if cs is not None:
                        st.write(f"**置信度**: {cs:.2f}")
                    st.write(f"**检测时间**: {record.get('emotion_time','-')}")
                else:
                    st.write("未进行情绪识别")
            with col2:
                if did_music:
                    st.write(f"**生成音乐**: {record.get('music_filename','-')}")
                    st.write(f"**目标情绪**: {record.get('target_emotion','-')}")
                    st.write(f"**生成时间**: {record.get('music_time','-')}")
                else:
                    st.write("未生成音乐")

    # 页脚导航
    max_page = (total + page_size - 1) // page_size
    st.caption(f"页码 {page_num}/{max_page}")

def logs_page():
    """系统日志页面"""
    st.header("📋 系统日志")
    
    # 检查登录状态
    require_login()
    
    # 管理员可浏览所有日志；普通用户仅看自己的
    is_admin = st.session_state.get('is_admin', False)
    filter_user = None if is_admin else st.session_state['user_id']

    # 筛选控件
    st.subheader("筛选")
    cols = st.columns(3)
    with cols[0]:
        actions = st.multiselect("操作类型", ["user_login","user_register","emotion_detect","music_generate","error","admin_view","admin_export"]) or None
        status = st.multiselect("状态", ["success","error"]) or None
    with cols[1]:
        start = st.date_input("开始日期", value=None)
        end = st.date_input("结束日期", value=None)
    with cols[2]:
        keyword = st.text_input("关键词").strip() or None
        page_size = st.selectbox("每页条数", [20,50,100,200], index=1)
        page_num = st.number_input("页码", min_value=1, step=1, value=1)

    # 查询
    start_str = start.isoformat() if start else None
    end_str = end.isoformat() if end else None
    offset = (page_num - 1) * page_size
    logs = db_manager.query_system_logs(
        user_id=filter_user,
        actions=actions,
        status=status,
        start=start_str,
        end=end_str,
        keyword=keyword,
        limit=int(page_size),
        offset=int(offset)
    )

    if not logs:
        st.info("暂无日志或筛选无结果")
        return

    # 展示
    for log in logs:
        status_color = "🟢" if log['status'] == 'success' else "🔴"
        with st.expander(f"{status_color} [{log['action']}] {log['timestamp']}"):
            st.write(f"用户: {log['user_id']}")
            st.write(f"状态: {log['status']}")
            st.write("详情:")
            st.code(str(log['details']), language="json")
    st.caption(f"本页 {len(logs)} 条")

    st.markdown("---")
    st.subheader("数据库浏览（白名单表）")
    table = st.selectbox("选择表", ["users","emotion_history","music_history","system_logs","sessions","events","feedbacks"])
    t_page_size = st.selectbox("每页条数(表)", [20,50,100,200], index=1, key="t_page_size")
    t_page_num = st.number_input("页码(表)", min_value=1, step=1, value=1, key="t_page_num")
    t_offset = (t_page_num - 1) * t_page_size
    records = db_manager.fetch_table(table, limit=int(t_page_size), offset=int(t_offset))
    if records:
        import pandas as pd
        st.dataframe(pd.DataFrame(records))
        st.caption(f"本页 {len(records)} 条")
    else:
        st.info("该页无数据或无权限")

if __name__ == "__main__":
    main() 