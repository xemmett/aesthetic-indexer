from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class YoloEntityStats:
    max: float
    mean: float
    frames_present: int
    frame_ratio: float


def _normalize_label(s: str) -> str:
    return (s or "").strip()


def detect_objects_yolo(
    frame_paths: list[Path],
    *,
    model: Path = Path("models/yolov8n.pt"),
    conf: float = 0.25,
    iou: float = 0.45,
    device: Optional[str] = None,
) -> dict[str, dict[str, float]]:
    """
    Run YOLOv8 on a set of frames and aggregate per-entity stats.

    Returns:
      {
        "person": {"max": 0.98, "mean": 0.72, "frames_present": 4, "frame_ratio": 0.80},
        ...
      }

    Notes:
    - mean is computed over per-frame maxima for frames where the entity is present (>= conf).
    - frames_present counts frames where entity appears with confidence >= conf.
    - frame_ratio = frames_present / total_frames
    """
    if not frame_paths:
        return {}

    # Lazy import so the package remains importable without ultralytics installed.
    from ultralytics import YOLO  # type: ignore

    
    # Initialize model
    if not model.exists():
        raise FileNotFoundError(f"Model not found at {model}")
    
    yolo = YOLO(str(model))

    total_frames = len(frame_paths)

    # Per entity: list of per-frame max confidences for frames where present.
    per_entity_frame_max: dict[str, list[float]] = {}
    per_entity_global_max: dict[str, float] = {}
    per_entity_frames_present: dict[str, int] = {}

    # Run frame-by-frame for determinism and lower peak VRAM (clips are small).
    for p in frame_paths:
        res = yolo.predict(source=str(p), conf=conf, iou=iou, device=device, verbose=False)
        if not res:
            continue

        r0 = res[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.cls is None or boxes.conf is None:
            continue

        # For this frame, track per-entity max confidence.
        frame_max: dict[str, float] = {}
        for cls_id, c in zip(boxes.cls.tolist(), boxes.conf.tolist(), strict=False):
            try:
                name = r0.names.get(int(cls_id))
            except Exception:
                name = None
            label = _normalize_label(str(name or cls_id))
            if not label:
                continue
            c = float(c)
            if c > frame_max.get(label, 0.0):
                frame_max[label] = c

        # Update aggregates
        for label, c in frame_max.items():
            per_entity_frame_max.setdefault(label, []).append(c)
            per_entity_frames_present[label] = per_entity_frames_present.get(label, 0) + 1
            per_entity_global_max[label] = max(per_entity_global_max.get(label, 0.0), c)

    out: dict[str, dict[str, float]] = {}
    for label in sorted(per_entity_global_max.keys()):
        frames_present = int(per_entity_frames_present.get(label, 0))
        vals = per_entity_frame_max.get(label, [])
        if not vals:
            continue
        out[label] = {
            "max": float(per_entity_global_max[label]),
            "mean": float(np.mean(vals)) if vals else 0.0,
            "frames_present": int(frames_present),
            "frame_ratio": float(frames_present / max(1, total_frames)),
        }
    return out


def _places365_categories_path() -> Path:
    return Path(__file__).with_name("places365_categories.txt")


def _load_places365_categories() -> list[str]:
    """
    Places365 has 365 fixed categories. To keep this repo lightweight, we treat the label
    file as optional:
    - If `places365_categories.txt` exists next to this module, we use it.
    - Otherwise we fall back to generic `class_{i}` names.
    """
    p = _places365_categories_path()
    if not p.exists():
        return [f"class_{i}" for i in range(365)]
    cats: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cats.append(s)
    # If it's malformed, still keep running with fallback labels.
    if len(cats) != 365:
        return [f"class_{i}" for i in range(365)]
    return cats


def classify_scenes_places365(
    frame_paths: list[Path],
    *,
    weights_path: Path,
    top_k: int = 3,
    device: Optional[str] = None,
) -> dict[str, float]:
    """
    Places365 scene classification with weighted aggregation.

    - Collect top-k per frame.
    - Aggregate weighted vote: sum(probability * frame_ratio) across frames.

    Returns dict of scene -> weighted_confidence (0..1-ish).
    """
    if not frame_paths:
        return {}

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import models, transforms  # type: ignore

    cats = _load_places365_categories()

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ResNet18 Places365 expects 365 classes.
    model = models.resnet18(num_classes=365)
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state_dict: dict[str, Any]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError("Unexpected Places365 checkpoint format.")

    # Strip possible 'module.' prefix
    cleaned = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    model.load_state_dict(cleaned, strict=False)
    model = model.to(dev)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    total = len(frame_paths)
    frame_ratio = 1.0 / max(1, total)
    agg: dict[int, float] = {}

    with torch.no_grad():
        for p in frame_paths:
            try:
                with Image.open(p) as im:
                    x = tfm(im.convert("RGB")).unsqueeze(0).to(dev)
            except Exception:
                continue
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            k = int(max(1, top_k))
            topv, topi = torch.topk(probs, k=k)
            for v, i in zip(topv.tolist(), topi.tolist(), strict=False):
                agg[int(i)] = float(agg.get(int(i), 0.0) + float(v) * frame_ratio)

    out: dict[str, float] = {}
    for idx, score in sorted(agg.items(), key=lambda kv: kv[1], reverse=True):
        if 0 <= idx < len(cats):
            out[cats[idx]] = float(score)
    return out


