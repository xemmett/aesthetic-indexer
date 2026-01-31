from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioTagResult:
    rms: float
    spectral_centroid: float
    silence_ratio: float
    tags: dict[str, float]  # tag -> confidence (heuristic)


def analyze_audio(
    wav_path: Path,
    *,
    silence_db_threshold: float = 35.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> AudioTagResult:
    """
    Minimal, deterministic audio analysis:
    - RMS energy
    - Spectral centroid
    - Silence ratio (first-class)

    Tags are heuristics (optional / best-effort). Raw signals are primary output.
    """
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if y.size == 0:
        # Treat empty audio as silence.
        return AudioTagResult(rms=0.0, spectral_centroid=0.0, silence_ratio=1.0, tags={"silence": 1.0})

    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms = float(np.mean(rms_frames))

    centroid_frames = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    centroid = float(np.mean(centroid_frames))

    # Silence ratio: percent of frames below threshold in dB relative to peak.
    # Convert RMS to dBFS-like scale via librosa amplitude_to_db.
    rms_db = librosa.amplitude_to_db(rms_frames, ref=np.max)
    silent = (rms_db <= -silence_db_threshold).astype(np.float32)
    silence_ratio = float(np.mean(silent))

    tags: dict[str, float] = {}
    if silence_ratio >= 0.60:
        tags["silence"] = min(1.0, (silence_ratio - 0.60) / 0.40)

    # Very rough heuristics for a few aesthetic buckets; these can be replaced later.
    # Mechanical: sustained energy + higher centroid
    if rms > 0.02 and centroid > 2000:
        tags["mechanical"] = float(min(1.0, (rms / 0.1) * (centroid / 6000)))

    # Crowd/chanting: mid centroid + moderate rms + low silence
    if silence_ratio < 0.2 and 800 < centroid < 2500 and 0.01 < rms < 0.08:
        tags["crowd"] = 0.5
        tags["chanting"] = 0.3

    return AudioTagResult(rms=rms, spectral_centroid=centroid, silence_ratio=silence_ratio, tags=tags)


