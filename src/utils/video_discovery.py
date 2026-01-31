from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".mpeg", ".mpg", ".m4v", ".webm"}


@dataclass(frozen=True)
class DiscoveredVideo:
    source: str  # 'archive' | 'youtube'
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


def discover_raw_videos(raw_videos_dir: Path) -> list[DiscoveredVideo]:
    return discover_archive_videos(raw_videos_dir / "archive") + discover_youtube_videos(
        raw_videos_dir / "youtube"
    )


