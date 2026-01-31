from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress

from src.audio.audio_tagger import analyze_audio
from src.utils.ffmpeg import extract_audio_wav, ffprobe_info, extract_frames_jpg
from src.utils.io import read_json, write_json
from src.utils.paths import data_dir


app = typer.Typer(add_completion=False, help="Compute query-primitives (signals) for each clip.")
console = Console()


def _iter_clip_meta(clips_root: Path) -> list[Path]:
    return sorted(clips_root.rglob("*.json"))


def _brightness_entropy(gray: np.ndarray, bins: int = 64) -> float:
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).astype(np.float64)
    p = hist / (np.sum(hist) + 1e-12)
    p = p[p > 0]
    ent = -float(np.sum(p * np.log2(p)))
    return ent


def _motion_score_from_frames(frames: list[np.ndarray]) -> float:
    """
    Simple optical-flow magnitude across consecutive frames.
    Returns average magnitude.
    """
    if len(frames) < 2:
        return 0.0
    mags: list[float] = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for f in frames[1:]:
        cur = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(float(np.mean(mag)))
        prev = cur
    return float(np.mean(mags)) if mags else 0.0


@app.command()
def run(
    clips_root: Path = typer.Option(None, help="Root dir for clips; defaults to DATA_DIR/clips"),
    out_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/signals"),
    overwrite: bool = typer.Option(False, help="Overwrite existing signals JSON."),
):
    run_job(clips_root=clips_root, out_dir=out_dir, overwrite=overwrite)


def run_job(
    *,
    clips_root: Path | None = None,
    out_dir: Path | None = None,
    overwrite: bool = False,
):
    """
    Writes one JSON per clip_id in data/signals/{clip_id}.json containing:
      motion_score, noise_level, silence_ratio, brightness_entropy
    """
    clips_root = clips_root or (data_dir() / "clips")
    out_dir = out_dir or (data_dir() / "signals")
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = _iter_clip_meta(clips_root)
    console.print(f"[bold]Compute signals[/bold] clips={len(metas)} -> {out_dir}")

    tmp = data_dir() / "signals_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("Computing", total=len(metas))
        for meta_path in metas:
            progress.advance(task)
            meta = read_json(meta_path)
            clip_id = meta.get("clip_id") or meta_path.stem
            clip_mp4 = Path(meta.get("filepath") or meta_path.with_suffix(".mp4"))

            out_json = out_dir / f"{clip_id}.json"
            if out_json.exists() and not overwrite:
                continue
            if not clip_mp4.exists():
                continue

            # Motion + brightness entropy from 5 evenly spaced frames
            info = ffprobe_info(clip_mp4)
            duration = info.duration_seconds
            if duration <= 0:
                continue
            ts = np.linspace(0.0, max(0.0, duration - 0.001), num=5).tolist()
            frame_dir = tmp / clip_id
            try:
                jpgs = extract_frames_jpg(clip_mp4, frame_dir, timestamps=ts, scale_width=256)
            except Exception:
                # Frame extraction failed - skip this clip
                continue
            
            frames: list[np.ndarray] = []
            ents: list[float] = []
            for j in jpgs:
                if not j.exists():
                    continue
                img = cv2.imread(str(j))
                if img is None:
                    continue
                frames.append(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ents.append(_brightness_entropy(gray))

            # Need at least 2 frames for motion score, at least 1 for brightness
            if len(frames) < 1:
                # No frames extracted - skip this clip
                continue
                
            motion_score = _motion_score_from_frames(frames) if len(frames) >= 2 else 0.0
            brightness_entropy = float(np.mean(ents)) if ents else 0.0

            # Audio: silence ratio + a noise proxy
            wav = tmp / f"{clip_id}.wav"
            try:
                wav_path = extract_audio_wav(clip_mp4, wav, sample_rate=16000, mono=True)
                if wav_path is None:
                    # Video has no audio stream - use silent audio values
                    noise_level = 0.0
                    silence_ratio = 1.0
                else:
                    a = analyze_audio(wav_path)
                    # noise_level proxy: rms * (1 - silence_ratio)
                    noise_level = float(a.rms * (1.0 - a.silence_ratio))
                    silence_ratio = float(a.silence_ratio)
            finally:
                if wav.exists():
                    try:
                        wav.unlink()
                    except Exception:
                        pass

            write_json(
                out_json,
                {
                    "clip_id": clip_id,
                    "motion_score": motion_score,
                    "noise_level": noise_level,
                    "silence_ratio": silence_ratio,
                    "brightness_entropy": brightness_entropy,
                },
            )

            # Cleanup temp frame dir
            if frame_dir.exists():
                for p in frame_dir.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                try:
                    frame_dir.rmdir()
                except Exception:
                    pass


if __name__ == "__main__":
    app()


