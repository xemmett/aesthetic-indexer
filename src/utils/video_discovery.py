from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".mpeg", ".mpg", ".m4v", ".webm"}


@dataclass(frozen=True)
class DiscoveredVideo:
    source: str  # 'archive' | 'youtube' | 'local'
    video_id: str
    filepath: Path
    year: Optional[int]
    meta: dict


def _safe_int_year(v: object) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(str(v)[:4])
    except Exception:
        return None


def discover_archive_videos(raw_archive_dir: Path) -> list[DiscoveredVideo]:
    """
    Expected layout (from scraper):
      raw_videos/archive/{collection}/{identifier}/{video_file}
      raw_videos/archive/{collection}/{identifier}/metadata.json
    """
    out: list[DiscoveredVideo] = []
    if not raw_archive_dir.exists():
        return out

    for video in raw_archive_dir.rglob("*"):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTS:
            continue

        # identifier is parent folder
        identifier = video.parent.name
        meta_path = video.parent / "metadata.json"
        meta: dict = {}
        year = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                year = _safe_int_year(meta.get("year"))
            except Exception:
                meta = {}

        out.append(
            DiscoveredVideo(
                source="archive",
                video_id=identifier,
                filepath=video,
                year=year,
                meta=meta,
            )
        )
    return out


def discover_youtube_videos(raw_youtube_dir: Path) -> list[DiscoveredVideo]:
    """
    Expected layout (from scraper):
      raw_videos/youtube/{keyword}/{id}/{id}.mp4
      raw_videos/youtube/{keyword}/{id}/{id}.info.json
    """
    out: list[DiscoveredVideo] = []
    if not raw_youtube_dir.exists():
        return out

    for video in raw_youtube_dir.rglob("*"):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTS:
            continue

        video_id = video.parent.name
        info_json = video.parent / f"{video_id}.info.json"
        meta: dict = {}
        year = None
        if info_json.exists():
            try:
                meta = json.loads(info_json.read_text(encoding="utf-8"))
                # yt-dlp: upload_date like YYYYMMDD
                upload_date = meta.get("upload_date")
                year = _safe_int_year(upload_date)
            except Exception:
                meta = {}

        out.append(
            DiscoveredVideo(
                source="youtube",
                video_id=video_id,
                filepath=video,
                year=year,
                meta=meta,
            )
        )
    return out


def _stable_video_id_from_path(path: Path) -> str:
    # Stable across runs/machines as long as the resolved absolute path stays the same.
    p = str(path.resolve()).lower().encode("utf-8", errors="ignore")
    return hashlib.sha1(p).hexdigest()[:16]


def _guess_year_from_path(path: Path) -> Optional[int]:
    # Very light heuristic: find a 4-digit year in the filename.
    import re

    m = re.search(r"(18\d{2}|19\d{2}|20\d{2})", path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def discover_local_videos(root_dir: Path) -> list[DiscoveredVideo]:
    """
    Unstructured mode: recursively finds video files under root_dir.
    """
    out: list[DiscoveredVideo] = []
    if not root_dir.exists():
        return out

    for video in root_dir.rglob("*"):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTS:
            continue
        vid = _stable_video_id_from_path(video)
        out.append(
            DiscoveredVideo(
                source="local",
                video_id=vid,
                filepath=video,
                year=_guess_year_from_path(video),
                meta={"input_root": str(root_dir.resolve()), "original_path": str(video.resolve())},
            )
        )
    return out


def discover_raw_videos(raw_videos_dir: Path) -> list[DiscoveredVideo]:
    # If this looks like our structured raw_videos dir, use structured discovery.
    # Otherwise, treat it as an arbitrary local folder (unstructured mode).
    if (raw_videos_dir / "archive").exists() or (raw_videos_dir / "youtube").exists():
        return discover_archive_videos(raw_videos_dir / "archive") + discover_youtube_videos(
            raw_videos_dir / "youtube"
        )
    return discover_local_videos(raw_videos_dir)


