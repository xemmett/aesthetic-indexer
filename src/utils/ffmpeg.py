from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


class FFmpegError(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n\nSTDERR:\n{proc.stderr}"
        )
    return proc


@dataclass(frozen=True)
class VideoInfo:
    duration_seconds: float
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]


def ffprobe_info(video_path: Path) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration:stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    proc = _run(cmd)
    data = json.loads(proc.stdout)

    duration = float(data.get("format", {}).get("duration") or 0.0)
    streams = data.get("streams") or []
    stream0 = streams[0] if streams else {}

    width = stream0.get("width")
    height = stream0.get("height")
    r_frame_rate = stream0.get("r_frame_rate")
    fps = None
    if r_frame_rate and isinstance(r_frame_rate, str) and "/" in r_frame_rate:
        num, den = r_frame_rate.split("/", 1)
        try:
            fps = float(num) / float(den)
        except Exception:
            fps = None

    return VideoInfo(duration_seconds=duration, width=width, height=height, fps=fps)


def extract_clip_copy(
    input_video: Path,
    start_time: float,
    end_time: float,
    output_clip: Path,
) -> None:
    output_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_time:.6f}",
        "-to",
        f"{end_time:.6f}",
        "-i",
        str(input_video),
        "-c",
        "copy",
        str(output_clip),
    ]
    _run(cmd)


def extract_frames_jpg(
    input_video: Path,
    output_dir: Path,
    *,
    timestamps: Optional[Iterable[float]] = None,
    fps: Optional[float] = None,
    scale_width: Optional[int] = None,
) -> list[Path]:
    """
    Extract frames as JPEGs.
    Provide either timestamps (seconds) or fps. If both are None, defaults to fps=1.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "%04d.jpg"

    vf_parts: list[str] = []
    if scale_width:
        vf_parts.append(f"scale={scale_width}:-1")

    if timestamps is not None:
        # Extract all frames in a single ffmpeg pass using select filter
        timestamp_list = sorted(set(timestamps))  # Sort and deduplicate
        
        if not timestamp_list:
            return []
        
        # Build select filter: eq(t,0.5)+eq(t,1.0)+eq(t,2.0)...
        select_expr = "+".join([f"eq(t,{ts:.6f})" for ts in timestamp_list])
        
        # Build video filter chain
        filter_parts = [f"select='{select_expr}'"]
        if scale_width:
            filter_parts.append(f"scale={scale_width}:-1")
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            ",".join(filter_parts),
            "-vsync",
            "0",  # Don't duplicate frames
            "-q:v",
            "2",
            str(pattern),
        ]
        
        try:
            _run(cmd)
            # Frames are extracted in video order (sorted timestamp order)
            # Output files are numbered 0001.jpg, 0002.jpg, etc.
            extracted = sorted(output_dir.glob("*.jpg"))
            if not extracted:
                # Batch extraction succeeded but no frames - try fallback
                raise FFmpegError("Batch extraction returned no frames")
            return extracted
        except FFmpegError as e:
            # If batch extraction fails, fall back to individual extractions
            # (for compatibility with edge cases or problematic videos)
            outputs: list[Path] = []
            fallback_vf_parts: list[str] = []
            if scale_width:
                fallback_vf_parts.append(f"scale={scale_width}:-1")
            
            # Try individual frame extraction as fallback
            for i, ts in enumerate(timestamps, start=1):
                out = output_dir / f"{i:04d}.jpg"
                try:
                    cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{ts:.6f}",
                        "-i",
                        str(input_video),
                        "-frames:v",
                        "1",
                    ]
                    if fallback_vf_parts:
                        cmd += ["-vf", ",".join(fallback_vf_parts)]
                    cmd += [str(out)]
                    _run(cmd)
                    if out.exists():
                        outputs.append(out)
                except FFmpegError:
                    continue
            
            if not outputs:
                # All fallback attempts failed - raise original error with context
                raise FFmpegError(
                    f"Frame extraction failed for {input_video.name}: "
                    f"batch extraction failed ({str(e)}), "
                    f"fallback individual extraction also failed for all {len(timestamps)} timestamps"
                )
            return outputs

    if fps is None:
        fps = 1.0

    vf_parts.insert(0, f"fps={fps}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_video),
        "-q:v",
        "2",
        "-vf",
        ",".join(vf_parts),
        str(pattern),
    ]
    _run(cmd)
    return sorted(output_dir.glob("*.jpg"))


def has_audio_stream(video_path: Path) -> bool:
    """Check if video file has an audio stream."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        return len(streams) > 0
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return False


def extract_audio_wav(
    input_video: Path,
    output_wav: Path,
    *,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Optional[Path]:
    """
    Extract audio from video to WAV file.
    Returns None if video has no audio stream.
    """
    # Check if video has audio stream first
    if not has_audio_stream(input_video):
        return None
    
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ar",
        str(sample_rate),
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += [str(output_wav)]
    
    try:
        _run(cmd)
        return output_wav
    except FFmpegError:
        # If extraction fails, return None (video might not have audio)
        return None


