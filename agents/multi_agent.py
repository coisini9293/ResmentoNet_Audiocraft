import os # 导入os模块，用于操作文件和目录
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable
from pathlib import Path
import sys as _sys

# 确保项目根目录在 sys.path，避免 "No module named 'tools'"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

# 加载 .env（支持在项目根或当前目录放置 .env）
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

from langchain_openai import ChatOpenAI # 导入ChatOpenAI类，用于与OpenAI API交互
from langchain_core.prompts import ChatPromptTemplate # 导入ChatPromptTemplate类，用于创建提示模板
from langchain_core.output_parsers import StrOutputParser # 导入StrOutputParser类，用于解析输出

from tools.mysql_repo import MySQLRepository, get_default_engine # 导入MySQLRepository类和get_default_engine函数
from tools.face_tool import detect_emotion # 导入detect_emotion函数
from tools.musicgen_tool import generate_music_files # 导入generate_music_files函数


@dataclass # 定义一个数据类，用于存储生成参数
class GenerationParams: # 定义一个类，用于存储生成参数
    duration: int = 60
    topk: int = 250
    topp: float = 0.0
    temperature: float = 1.0
    cfg_coef: float = 3.0
    model: str = "facebook/musicgen-stereo-large"
    model_path: str = ""
    decoder: str = "Default"  # or "MultiBand_Diffusion"


def _make_llm() -> ChatOpenAI: # 定义一个函数，用于创建LLM模型
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 环境变量未设置")
    base_url = os.getenv("OPENAI_BASE_URL")  # 如使用 Moonshot，建议设置为 https://api.moonshot.cn/v1
    return ChatOpenAI(model="moonshot-v1-8k", openai_api_key=api_key, openai_api_base=base_url, temperature=0.3)

def _build_calming_recipe(emotion: str) -> Dict[str, Any]:
    """根据检测到的情绪，构造安抚/情绪调节向的风格建议与约束。"""
    e = (emotion or "").lower()
    # 默认的舒缓基线
    recipe = {
        "goal": "soothing and calming; reduce stress and negative arousal",
        "genres": ["lofi", "ambient", "neo-classical", "soft acoustic"],
        "bpm_min": 60,
        "bpm_max": 85,
        "instruments": ["piano", "warm pads", "soft strings", "clean guitar", "gentle drums"],
        "arrangement": "intro - gentle build - soft outro",
        "mix": "warm, smooth highs, controlled low-end, soft transient, low distortion",
        "avoid": ["harsh", "aggressive", "distorted", "screaming", "industrial noise", "heavy metal", "hard rock"],
    }
    if e in {"angry", "anger"}:
        recipe.update({
            "genres": ["lofi", "chillhop", "ambient", "soft jazz"],
            "bpm_min": 60,
            "bpm_max": 80,
            "instruments": ["piano", "warm pads", "soft sax", "brush drums"],
            "avoid": recipe["avoid"] + ["fast tempo", "sharp snares", "distorted guitars"],
        })
    elif e in {"fear"}:
        recipe.update({
            "genres": ["ambient", "neo-classical", "cinematic soft"],
            "bpm_min": 55,
            "bpm_max": 75,
            "instruments": ["piano", "strings", "pads", "choir ahh"],
            "avoid": recipe["avoid"] + ["suspense risers", "horror stingers", "dissonant clusters"],
        })
    elif e in {"disgust"}:
        recipe.update({
            "genres": ["soft jazz", "ambient", "bossa nova slow"],
            "bpm_min": 60,
            "bpm_max": 85,
            "instruments": ["piano", "upright bass", "brush drums", "pads"],
            "avoid": recipe["avoid"] + ["industrial", "glitch noise"],
        })
    elif e in {"sad"}:
        recipe.update({
            "genres": ["lofi", "acoustic folk", "neo-classical"],
            "bpm_min": 65,
            "bpm_max": 90,
            "instruments": ["piano", "acoustic guitar", "soft strings", "pads"],
            "mix": "warm and intimate, gentle reverb, avoid excessive darkness",
            "avoid": recipe["avoid"] + ["overly depressing tone", "excessive dissonance"],
        })
    elif e in {"neutral"}:
        recipe.update({"genres": ["lofi", "ambient", "soft chill"], "bpm_min": 65, "bpm_max": 95})
    elif e in {"happy"}:
        # 也保持舒缓，而非更兴奋
        recipe.update({"genres": ["soft pop", "chillhop", "lofi"], "bpm_min": 80, "bpm_max": 105})
    return recipe


# agent3 用于生成初稿（安抚负面情绪导向）
def agent3_draft_prompt(emotion_result: Dict[str, Any]) -> str:
    llm = _make_llm()
    emotion = emotion_result.get("emotion", "unknown")
    recipe = _build_calming_recipe(emotion)
    tmpl = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a music prompt designer for a text-to-music model. Always craft prompts that are soothing and calming to help regulate negative emotions (anger, fear, disgust, sadness). Keep output concise English under 350 chars.",
        ),
        (
            "human",
            (
                "Detected emotion: {emotion}\nConfidence: {confidence}\nDetails: {extra}\n\n"
                "Goal: calming/soothing. Use this recipe: genres={genres}; bpm_range={bpm_min}-{bpm_max}; instruments={instruments}; arrangement={arrangement}; mix={mix}; avoid={avoid}.\n"
                "Produce a concrete, single-line music description including mood, genre, bpm range, key instruments, 3-part arrangement, and mixing hints. Avoid the listed elements."
            ),
        ),
    ])
    chain = tmpl | llm | StrOutputParser()
    return chain.invoke({
        "emotion": emotion,
        "confidence": emotion_result.get("confidence", 0.0),
        "extra": emotion_result.get("extra", {}),
        **recipe,
    }).strip()


def agent5_merge_user_prompt(base_prompt: str, user_prompt: str) -> str:
    """将网页自定义 prompt 与基线 calming 提示词合并，保持舒缓导向。"""
    llm = _make_llm()
    tmpl = ChatPromptTemplate.from_messages([
        (
            "system",
            "You merge a user's custom music description with a calming baseline prompt. Keep the result soothing and concise (<= 400 chars). Prefer gentle variants if conflicts arise.",
        ),
        (
            "human",
            (
                "Base (calming): {base}\nUser prompt: {user}\n\n"
                "Return a single-line final prompt that keeps calming intent while reflecting user's specifics (style, instruments, structure)."
            ),
        ),
    ])
    chain = tmpl | llm | StrOutputParser()
    return chain.invoke({"base": base_prompt, "user": user_prompt}).strip()

# agent4 用于生成个性化完整提示词
def agent4_personalize_prompt(user_id: Optional[int], prompt_initial: str, repo: Optional[MySQLRepository], user_prefs_override: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """Return (personalized_prompt, suggested_params_dict)."""
    user_prefs: Dict[str, Any] = {}
    if user_id is not None and repo is not None:
        try:
            user_prefs = repo.get_user_preferences(user_id) or {}
        except Exception:
            user_prefs = {}
    # 覆盖/补充来自页面的用户偏好
    if user_prefs_override:
        try:
            # 合并覆盖（浅合并）
            merged = dict(user_prefs)
            merged.update({k: v for k, v in user_prefs_override.items() if v not in (None, "", [])})
            user_prefs = merged
        except Exception:
            pass

    llm = _make_llm()
    tmpl = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a music prompt optimizer. Merge user preferences into the base prompt while preserving a soothing/calming orientation. If any preference conflicts with calmness (e.g., aggressive/harsh/distorted), down-weight or replace them with gentle alternatives. Output <= 450 chars.",
        ),
        (
            "human",
            (
                "Base prompt:\n{base}\n\nUser preferences (JSON):\n{prefs}\n\n"
                "Refine to keep calming intent. Reflect user's genres/instruments/bpm bounds/negative prompts as long as they don't break calmness."
                " Respond in two parts separated by a line '---PARAMS---'. First the final prompt (single line). Then a compact JSON with keys: duration, topk, topp, temperature, cfg_coef."
            ),
        ),
    ])
    output = (tmpl | llm | StrOutputParser()).invoke({
        "base": prompt_initial,
        "prefs": user_prefs,
    })
    if "---PARAMS---" in output:
        prompt_final, params_json = output.split("---PARAMS---", 1)
    else:
        prompt_final, params_json = output, "{}"
    prompt_final = prompt_final.strip().replace("\n", " ")

    # 尝试解析参数
    import json
    try:
        params = json.loads(params_json)
        if not isinstance(params, dict):
            params = {}
    except Exception:
        params = {}
    return prompt_final, params

# 主函数，用于执行多代理流程
def run_multi_agent(
    media_path: Optional[str],
    user_id: Optional[int] = None,
    gen_params: Optional[GenerationParams] = None,
    melody: Optional[tuple] = None,
    override_emotion: Optional[str] = None,
    user_prefs_override: Optional[Dict[str, Any]] = None,
    manual_prompt: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Orchestrate Agent1->Agent3->Agent4->Agent2 and return result dict.
    """
    gen_params = gen_params or GenerationParams()
    if progress_cb is None:
        progress_cb = lambda pct, text='': None  # no-op

    # 数据库仓库
    repo: Optional[MySQLRepository]
    try:
        engine = get_default_engine()
        repo = MySQLRepository(engine)
    except Exception:
        repo = None

    # Agent1: 情绪识别 / 手动覆盖
    progress_cb(5, "init")
    emotion_res: Dict[str, Any] = {}
    if override_emotion:
        emotion_res = {"emotion": override_emotion, "confidence": 1.0}
    elif media_path:
        progress_cb(15, "detect_emotion")
        emotion_res = detect_emotion(media_path)
    else:
        emotion_res = {"emotion": "neutral", "confidence": 0.0}

    # Agent3: 初稿（构建 calming 基线），并在有用户自定义时用 Agent5 合并
    progress_cb(30, "agent3_prompt")
    prompt_initial = agent3_draft_prompt(emotion_res)
    if manual_prompt and manual_prompt.strip():
        try:
            prompt_initial = agent5_merge_user_prompt(prompt_initial, manual_prompt.strip())
        except Exception:
            prompt_initial = f"{prompt_initial} | {manual_prompt.strip()}"

    # Agent4: 个性化
    progress_cb(45, "agent4_personalize")
    prompt_final, suggested = agent4_personalize_prompt(user_id, prompt_initial, repo, user_prefs_override=user_prefs_override)

    # 合并参数建议
    p = GenerationParams(**{**gen_params.__dict__, **{k: suggested.get(k, v) for k, v in gen_params.__dict__.items()}})

    # Agent2: 生成音乐
    progress_cb(60, "agent2_generate")
    out = generate_music_files(
        model=p.model,
        model_path=p.model_path,
        decoder=p.decoder,
        text=prompt_final,
        melody=melody,
        duration=p.duration,
        topk=p.topk,
        topp=p.topp,
        temperature=p.temperature,
        cfg_coef=p.cfg_coef,
    )
    progress_cb(95, "finalize")

    result = {
        "emotion": emotion_res,
        "prompt_initial": prompt_initial,
        "prompt_final": prompt_final,
        "params": p.__dict__,
        "outputs": out,
    }

    # 可选：落库（若可用）
    if repo is not None:
        try:
            session_id = None
            if user_id is not None:
                session_id = repo.create_session(user_id=user_id, source="web")
            if session_id is not None:
                repo.save_emotion(session_id=session_id, emotion=emotion_res)
                repo.save_generation(
                    session_id=session_id,
                    prompt_initial=prompt_initial,
                    prompt_personalized=prompt_final,
                    model=p.model,
                    params_json={"topk": p.topk, "topp": p.topp, "temperature": p.temperature, "cfg_coef": p.cfg_coef, "decoder": p.decoder},
                    wav_path=out.get("wav"),
                    video_path=out.get("video"),
                    duration=p.duration,
                )
        except Exception:
            pass

    progress_cb(100, "done")
    return result


def submit(request: Dict[str, Any]) -> Dict[str, Any]:
    """统一提交入口。
    request 支持字段：
      - media_path: Optional[str]
      - user_id: Optional[int]
      - gen_params: Optional[Dict]
      - override_emotion: Optional[str]
      - user_prefs: Optional[Dict]
      - manual_prompt: Optional[str]
    """
    gp_dict = request.get("gen_params") or {}
    params = GenerationParams(**{k: gp_dict.get(k, v) for k, v in GenerationParams().__dict__.items()})
    return run_multi_agent(
        media_path=request.get("media_path"),
        user_id=request.get("user_id"),
        gen_params=params,
        melody=None,
        override_emotion=request.get("override_emotion"),
        user_prefs_override=request.get("user_prefs"),
        manual_prompt=request.get("manual_prompt"),
    )



if __name__ == "__main__":
    # 提供一个简易 CLI：可选择仅做情绪识别，或跑完整多智能体流程
    import argparse
    from tools.face_tool import detect_emotion as _detect

    parser = argparse.ArgumentParser(description="Run multi-agent pipeline or face-only detection")
    parser.add_argument("--media_path", type=str, default=None, help="图片/视频路径；留空则尝试弹出文件选择对话框")
    parser.add_argument("--user_id", type=int, default=None, help="可选用户ID（用于个性化与落库）")
    parser.add_argument("--duration", type=int, default=15, help="生成音频时长（秒）")
    parser.add_argument("--face_only", action="store_true", help="仅做面部情绪识别，不调用LLM与音乐生成")
    args = parser.parse_args()

    media_path = args.media_path
    if not media_path:
        # 尝试弹出系统文件选择器（Windows 可用）；失败则回退到输入
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            media_path = filedialog.askopenfilename(title="选择图片或视频",
                                                    filetypes=[("Media", ".jpg .jpeg .png .bmp .webp .mp4 .mov .avi .mkv"), ("All", "*.*")])
        except Exception:
            media_path = None
        if not media_path:
            media_path = input("请输入图片/视频路径（留空则仅使用中立情绪）：").strip() or None

    if args.face_only:
        if media_path:
            res = _detect(media_path)
            print("Emotion:", res)
        else:
            print("未提供媒体文件，无法执行情绪识别。")
    else:
        params = GenerationParams(duration=args.duration)
        res = run_multi_agent(media_path=media_path, user_id=args.user_id, gen_params=params)
        print("Emotion:", res.get("emotion"))
        print("Prompt initial:", res.get("prompt_initial"))
        print("Prompt final:", res.get("prompt_final"))
        print("Outputs:", res.get("outputs"))

