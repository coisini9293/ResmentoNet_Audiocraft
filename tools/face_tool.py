from __future__ import annotations

import os
import sys
from importlib.machinery import SourceFileLoader
from typing import Any, Dict, Optional

import torch
import torchvision.transforms as T
from PIL import Image


_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_MODEL = None  # ResEmoteNet 实例
_TFM = None    # 预处理

# FER2013 7类 -> 4类映射所用的索引（按 FER 默认顺序）
_FER7_TO_4_KEEP = [0, 1, 2, 4]  # angry, disgust, fear, sad
_FER4_LABELS = ["angry", "disgust", "fear", "sad"]


def _build_transform() -> T.Compose:
    global _TFM
    if _TFM is None:
        size = int(os.getenv("FACE_IMG_SIZE", "224"))
        _TFM = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return _TFM


def _find_resemotenet_class() -> Any:
    """从本地 ResEmoteNet 仓库加载模型类 ResEmoteNet。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_dir = os.getenv("RESE_REPO", os.path.join(project_root, "ResEmoteNet"))
    # 常见实现文件名
    for fname in ["ResEmoteNet.py", "resemotenet.py", "model.py"]:
        f = os.path.join(repo_dir, "approach", fname)
        if os.path.exists(f):
            mod = SourceFileLoader("resemote_dyn_mod", f).load_module()
            if hasattr(mod, "ResEmoteNet"):
                return getattr(mod, "ResEmoteNet")
    raise ImportError("未找到 ResEmoteNet 模型实现，请检查 RESE_REPO/approach 下的文件。")


def _load_rese_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    # 权重路径：默认 Face/models/fer2013_model.pth，可通过 FACE_CKPT 覆盖
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_ckpt = os.path.join(project_root, "Face", "models", "fer2013_model.pth")
    ckpt = os.getenv("FACE_CKPT", default_ckpt)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"找不到权重文件: {ckpt}")
    Model = _find_resemotenet_class()
    try:
        model = Model(num_classes=7)
    except TypeError:
        model = Model()
    model.to(_DEVICE)
    state = torch.load(ckpt, map_location=_DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    model.load_state_dict(new_state, strict=False)
    model.eval()
    _MODEL = model
    return _MODEL


def _read_image_from_path(path: str) -> Optional[Image.Image]: # 定义一个函数，用于读取图片
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)
    except Exception:
        pass
    return None


@torch.inference_mode()
def detect_emotion(media_path: str) -> Dict[str, Any]: # 定义一个函数，用于检测情感
    img = _read_image_from_path(media_path)
    if img is None:
        return {"emotion": "neutral", "confidence": 0.0, "extra": {"error": "unreadable", "path": media_path}}

    # 使用 ResEmoteNet 推理（7类），再折叠到4类
    try:
        model = _load_rese_model()
        x = _build_transform()(img).unsqueeze(0).to(_DEVICE)
        logits = model(x)
        # 判定类别数
        num_classes = logits.shape[1] if logits.dim() == 2 else int(logits.numel())
        probs = torch.softmax(logits, dim=1).squeeze(0)
        if probs.numel() == 4:
            # 直接四类
            conf, idx4 = torch.max(probs, dim=0)
            idx4 = int(idx4.item())
            emotion = _FER4_LABELS[idx4]
            return {"emotion": emotion, "confidence": float(conf.item()), "extra": {"path": media_path, "classes": 4}}
        else:
            # 假设为 FER7 顺序，折叠到四类
            keep = torch.tensor(_FER7_TO_4_KEEP, device=probs.device, dtype=torch.long)
            sub = probs.index_select(0, keep)
            sub = sub / (sub.sum() + 1e-12)
            conf, idx_sub = torch.max(sub, dim=0)
            idx4 = int(idx_sub.item())
            emotion = _FER4_LABELS[idx4]
            return {"emotion": emotion, "confidence": float(conf.item()), "extra": {"path": media_path, "classes": int(probs.numel())}}
    except Exception:
        # 回退旧模型（ResNet18 本地权重），以确保功能不至于中断
        try:
            # 旧实现：导入本地 ResNet18 权重的简易推理
            face_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Face"))
            if face_dir not in sys.path:
                sys.path.insert(0, face_dir)
            from models.resnet18_no import ResNet18  # type: ignore
            model = ResNet18().to(_DEVICE)
            # 简单查找旧权重
            ckpt = os.path.join(face_dir, "RAF_checkpoints_ResNet18_1_200", "best", "best_model_70200.pth")
            state = torch.load(ckpt, map_location=_DEVICE)
            model.load_state_dict(state, strict=False)
            model.eval()
            x = T.Compose([
                T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])(img).unsqueeze(0).to(_DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            if probs.numel() == 4:
                conf, idx = torch.max(probs, dim=0)
                idx4 = int(idx.item())
                emotion = _FER4_LABELS[idx4]
            else:
                keep = torch.tensor(_FER7_TO_4_KEEP, device=probs.device, dtype=torch.long)
                sub = probs.index_select(0, keep)
                sub = sub / (sub.sum() + 1e-12)
                conf, idx = torch.max(sub, dim=0)
                idx4 = int(idx.item())
                emotion = _FER4_LABELS[idx4]
            return {"emotion": emotion, "confidence": float(conf.item()), "extra": {"path": media_path, "fallback": True}}
        except Exception:
            return {"emotion": "neutral", "confidence": 0.0, "extra": {"error": "inference_failed", "path": media_path}}


