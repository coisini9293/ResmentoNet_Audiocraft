from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import sys
import importlib
import os
import time
import shutil

# 先设置 UTF-8 与 ASCII 安全目录（在导入 demo 模块之前，确保其模块级 RESULT_DIR 使用 ASCII 路径）
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
_ASCII_TMP = os.environ.get('ASCII_TMP') or r"C:\\Windows\\Temp"
for _k in ('TMP', 'TEMP', 'TMPDIR'):
    os.environ.setdefault(_k, _ASCII_TMP)
os.environ.setdefault('GRADIO_TEMP_DIR', os.path.join(_ASCII_TMP, 'gradio_tmp'))
os.environ.setdefault('HF_HOME', os.path.join(_ASCII_TMP, 'hf_home'))
os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(_ASCII_TMP, 'hf_cache'))
os.environ.setdefault('TORCH_HOME', os.path.join(_ASCII_TMP, 'torch_home'))

# 结果目录：若未设置或包含非 ASCII 字符，则使用 ASCII 安全目录
def _is_ascii_path(p: str) -> bool:
    try:
        p.encode('ascii')
        return True
    except Exception:
        return False

if not os.environ.get('RESULT_DIR') or not _is_ascii_path(os.environ.get('RESULT_DIR', '')):
    _default_result = os.path.join(_ASCII_TMP, 'emotion_music_result')
    try:
        os.makedirs(_default_result, exist_ok=True)
        os.environ['RESULT_DIR'] = _default_result
    except Exception:
        # 兜底为当前目录下 result
        os.environ.setdefault('RESULT_DIR', 'result')

"""导入 audiocraft demo 模块并打上补丁，避免 Windows 进程池问题。"""
try:
    mgen_mod = importlib.import_module(
        "audiocraft_main.audiocraft_main.demos.musicgen_app"
    )  # type: ignore[attr-defined]
except Exception:
    demo_dir = (Path(__file__).resolve().parents[1] / "audiocraft-main" / "audiocraft-main" / "demos")
    sys.path.insert(0, str(demo_dir))
    mgen_mod = importlib.import_module("musicgen_app")  # type: ignore[attr-defined]

predict_full = getattr(mgen_mod, "predict_full")

# 替换模块内的进程池为同步执行，避免 "child process terminated" 错误
class _DummyFuture:
    def __init__(self, value):
        self._value = value
    def result(self, timeout: Optional[float] = None):
        return self._value

class _DummyExecutor:
    def submit(self, fn, *args, **kwargs):
        return _DummyFuture(fn(*args, **kwargs))

try:
    mgen_mod.pool = _DummyExecutor()  # type: ignore[attr-defined]
except Exception:
    pass

# 定义一个函数，用于生成音乐文件
def generate_music_files(
    model: str,
    model_path: str,
    decoder: str,
    text: str,
    melody: Optional[Tuple[int, Any]],
    duration: int,
    topk: int,
    topp: float,
    temperature: float,
    cfg_coef: float,
) -> Dict[str, Optional[str]]:
    # 设定 ASCII 安全的临时与结果目录，避免 Windows 中文路径引发编码问题
    ascii_tmp = os.environ.get('ASCII_TMP') or (r"C:\\Windows\\Temp")
    for k in ('TMP', 'TEMP', 'TMPDIR'):
        os.environ.setdefault(k, ascii_tmp)
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

    # 调用predict_full函数，生成音乐文件
    # 在 ASCII 目录中执行生成，避免第三方在 cwd 上做 ASCII 编码
    prev_cwd = os.getcwd()
    try:
        os.chdir(os.environ.get('RESULT_DIR', _ASCII_TMP))
        video, wav, video_mbd, wav_mbd = predict_full(
            model, model_path, decoder, text, melody, duration, topk, topp, temperature, cfg_coef
        )
    finally:
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass
    # 统一标准化路径至 ASCII 安全的 result 目录，并复制输出为 ASCII 文件名
    result_dir = Path(os.environ.get("RESULT_DIR", "result")).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    def norm(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        return str(Path(p).resolve())

    def copy_ascii(src: Optional[str], suffix: str) -> Optional[str]:
        if not src:
            return None
        try:
            ts = int(time.time() * 1000)
            dst = result_dir / f"{suffix}_{ts}{Path(src).suffix or ''}"
            shutil.copy2(src, dst)
            return str(dst.resolve())
        except Exception:
            return norm(src)

    return {
        "wav": copy_ascii(wav, "audio"),
        "video": copy_ascii(video, "video"),
        "wav_mbd": copy_ascii(wav_mbd, "audio_mbd"),
        "video_mbd": copy_ascii(video_mbd, "video_mbd"),
    }


