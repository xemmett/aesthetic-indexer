from __future__ import annotations

import csv
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import typer
from rich.console import Console
from rich.progress import Progress

from src.utils.deduplication import compute_phash, phash_distance
from src.utils.ffmpeg import FFmpegError, extract_clip_copy, extract_frames_jpg
from src.utils.io import write_json
from src.utils.paths import data_dir
from src.utils.video_discovery import discover_raw_videos


app = typer.Typer(add_completion=False, help="Extract 1–7s clips from shot CSVs using deterministic policy.")
console = Console()


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".mpeg", ".mpg", ".m4v", ".webm"}


@dataclass(frozen=True)
class Shot:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _load_shots(csv_path: Path) -> list[Shot]:
    shots: list[Shot] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                s = float(row["start_time"])
                e = float(row["end_time"])
            except Exception:
                continue
            if e > s:
                shots.append(Shot(start=s, end=e))
    return shots


def _windows_for_shot(
    shot: Shot,
    *,
    min_seconds: float,
    max_seconds: float,
    overlap: float,
) -> list[Shot]:
    d = shot.duration
    if d < min_seconds:
        return []
    if d <= max_seconds:
        return [shot]

    step = max_seconds * (1.0 - overlap)
    if step <= 0:
        step = max_seconds

    out: list[Shot] = []
    t = shot.start
    while (t + max_seconds) <= shot.end:
        out.append(Shot(start=t, end=t + max_seconds))
        t += step

    # Tail rule: if remaining tail >= min_seconds, add final window ending at shot.end.
    if out:
        last_end = out[-1].end
        if (shot.end - last_end) >= min_seconds:
            final_start = max(shot.start, shot.end - max_seconds)
            out.append(Shot(start=final_start, end=shot.end))
    else:
        # Extremely short max_seconds? fallback:
        out.append(Shot(start=shot.end - max_seconds, end=shot.end))
    return out


def _is_static_slide_by_phash(first_jpg: Path, last_jpg: Path, max_distance: int) -> bool:
    try:
        h1 = compute_phash(first_jpg).hex_hash
        h2 = compute_phash(last_jpg).hex_hash
        return phash_distance(h1, h2) <= max_distance
    except Exception:
        return False


@app.command()
def run(
    raw_videos_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/raw_videos"),
    shots_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/shots"),
    out_clips_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/clips"),
    overwrite: bool = typer.Option(False, help="Overwrite existing clips."),
    static_phash_distance: int = typer.Option(1, help="If first/last frame pHash distance <= this, discard as static."),
):
    run_job(
        raw_videos_dir=raw_videos_dir,
        shots_dir=shots_dir,
        out_clips_dir=out_clips_dir,
        overwrite=overwrite,
        static_phash_distance=static_phash_distance,
    )


def run_job(
    *,
    raw_videos_dir: Path | None = None,
    shots_dir: Path | None = None,
    out_clips_dir: Path | None = None,
    overwrite: bool = False,
    static_phash_distance: int = 1,
):
    """
    Deterministic extraction policy (see CLIP_EXTRACTION_POLICY.md).
    Outputs:
      data/clips/{source}/{video_id}/{clip_id}.mp4
      data/clips/{source}/{video_id}/{clip_id}.json
    """
    raw_videos_dir = raw_videos_dir or (data_dir() / "raw_videos")
    shots_dir = shots_dir or (data_dir() / "shots")
    out_clips_dir = out_clips_dir or (data_dir() / "clips")

    out_clips_dir.mkdir(parents=True, exist_ok=True)

    min_seconds = _env_float("CLIP_MIN_SECONDS", 1.0)
    max_seconds = _env_float("CLIP_MAX_SECONDS", 7.0)
    overlap = _env_float("CLIP_WINDOW_OVERLAP", 0.25)
    max_per_video = _env_int("CLIP_MAX_PER_VIDEO", 500)
    phash_max_dist = _env_int("PHASH_MAX_DISTANCE", 4)

    videos = discover_raw_videos(raw_videos_dir)
    console.print(f"[bold]Clip extraction[/bold] videos={len(videos)} -> {out_clips_dir}")

    with Progress() as progress:
        task = progress.add_task("Extracting", total=len(videos))
        for v in videos:
            progress.advance(task)
            shots_csv = shots_dir / f"{v.source}__{v.video_id}.csv"
            if not shots_csv.exists():
                continue

            shots = _load_shots(shots_csv)
            windows: list[Shot] = []
            for s in shots:
                windows.extend(
                    _windows_for_shot(
                        s, min_seconds=min_seconds, max_seconds=max_seconds, overlap=overlap
                    )
                )

            windows.sort(key=lambda x: (x.start, x.end))
            if len(windows) > max_per_video:
                windows = windows[:max_per_video]

            video_out_dir = out_clips_dir / v.source / v.video_id
            video_out_dir.mkdir(parents=True, exist_ok=True)

            # In-video dedup set (pHash of first frame) to prevent sludge.
            seen_phashes: list[str] = []

            for w in windows:
                clip_id = str(uuid.uuid4())
                clip_path = video_out_dir / f"{clip_id}.mp4"
                meta_path = video_out_dir / f"{clip_id}.json"

                if (clip_path.exists() and meta_path.exists()) and not overwrite:
                    continue

                try:
                    extract_clip_copy(v.filepath, w.start, w.end, clip_path)
                except FFmpegError:
                    # Fallback to re-encode if stream-copy fails.
                    import subprocess

                    cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{w.start:.6f}",
                        "-to",
                        f"{w.end:.6f}",
                        "-i",
                        str(v.filepath),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "20",
                        "-c:a",
                        "aac",
                        str(clip_path),
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode != 0:
                        continue

                duration = w.end - w.start

                # Extract two frames for static-slide check + phash dedup.
                tmp_dir = video_out_dir / ".tmp_frames" / clip_id
                try:
                    frames = extract_frames_jpg(
                        clip_path,
                        tmp_dir,
                        timestamps=[0.0, max(0.0, duration - 0.001)],
                        scale_width=320,
                    )
                    if len(frames) == 2 and _is_static_slide_by_phash(
                        frames[0], frames[1], static_phash_distance
                    ):
                        clip_path.unlink(missing_ok=True)
                        continue

                    first_phash = compute_phash(frames[0]).hex_hash if frames else None
                except Exception:
                    first_phash = None
                finally:
                    # Cleanup tmp frames
                    if tmp_dir.exists():
                        for p in tmp_dir.glob("*"):
                            try:
                                p.unlink()
                            except Exception:
                                pass
                        try:
                            tmp_dir.rmdir()
                        except Exception:
                            pass

                # In-video near-dup check.
                if first_phash:
                    is_dup = any(phash_distance(first_phash, h) <= phash_max_dist for h in seen_phashes)
                    if is_dup:
                        clip_path.unlink(missing_ok=True)
                        continue
                    seen_phashes.append(first_phash)

                write_json(
                    meta_path,
                    {
                        "clip_id": clip_id,
                        "source": v.source,
                        "video_id": v.video_id,
                        "filepath": str(clip_path),
                        "start_time": w.start,
                        "end_time": w.end,
                        "duration": duration,
                        "year": v.year,
                        "perceptual_hash": first_phash,
                    },
                )


if __name__ == "__main__":
    app()


