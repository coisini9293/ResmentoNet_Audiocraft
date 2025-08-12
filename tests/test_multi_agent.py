import argparse
import os
import sys
from pathlib import Path


# 保证项目根在 import 路径中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.multi_agent import (
    run_multi_agent,
    GenerationParams,
    agent3_draft_prompt,
    agent4_personalize_prompt,
)
from tools.face_tool import detect_emotion
from tools.mysql_repo import MySQLRepository, get_default_engine


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def has_mysql() -> bool:
    try:
        _ = get_default_engine()
        return True
    except Exception:
        return False


def test_face(media_path: str | None):
    print("\n[1/4] Face detection")
    if not media_path:
        print("- skip (no media_path provided)")
        return {"emotion": "neutral", "confidence": 0.0}
    try:
        res = detect_emotion(media_path)
        print("- result:", res)
        return res
    except Exception as e:
        print("- error:", e)
        return {"emotion": "neutral", "confidence": 0.0, "extra": {"error": str(e)}}


def test_llm_prompts(emotion_res: dict, skip_llm: bool):
    print("\n[2/4] LLM prompts (draft + personalize)")
    if skip_llm or not has_openai_key():
        print("- skip LLM (no OPENAI_API_KEY or --skip_llm)")
        return "", ("", {})
    try:
        draft = agent3_draft_prompt(emotion_res)
        print("- draft:", draft)
        repo = None
        if has_mysql():
            try:
                repo = MySQLRepository(get_default_engine())
            except Exception:
                repo = None
        final_prompt, params = agent4_personalize_prompt(user_id=None, prompt_initial=draft, repo=repo)
        print("- final:", final_prompt)
        print("- params:", params)
        return draft, (final_prompt, params)
    except Exception as e:
        print("- error:", e)
        return "", ("", {})


def test_music(media_path: str | None, final_prompt: str, duration: int, skip_music: bool):
    print("\n[3/4] Music generation (multi-agent end-to-end)")
    if skip_music:
        print("- skip music generation (--skip_music)")
        return {}
    try:
        # 若 final_prompt 为空，run_multi_agent 会自行生成
        params = GenerationParams(duration=duration)
        res = run_multi_agent(media_path=media_path, user_id=None, gen_params=params)
        print("- outputs:", res.get("outputs"))
        return res
    except Exception as e:
        print("- error:", e)
        return {}


def main():
    parser = argparse.ArgumentParser(description="Test multi-agent pipeline")
    parser.add_argument("--media_path", type=str, default=None, help="optional image/video path for emotion detection")
    parser.add_argument("--duration", type=int, default=10, help="music duration seconds (short for test)")
    parser.add_argument("--skip_llm", action="store_true", help="skip LLM draft/personalize")
    parser.add_argument("--skip_music", action="store_true", help="skip music generation")
    args = parser.parse_args()

    media_path = args.media_path
    if media_path is not None and not Path(media_path).exists():
        print(f"[warn] media_path not exists: {media_path}")
        media_path = None

    # 1) Face detection
    emotion_res = test_face(media_path)

    # 2) LLM prompts
    _, (final_prompt, _params) = test_llm_prompts(emotion_res, skip_llm=args.skip_llm)

    # 3) Music generation (end-to-end)
    test_music(media_path, final_prompt, args.duration, skip_music=args.skip_music)

    print("\n[4/4] Done.")


if __name__ == "__main__":
    main()








