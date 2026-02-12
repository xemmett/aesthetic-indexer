from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress

from src.utils.ffmpeg import ffprobe_info, extract_frames_jpg
from src.utils.paths import data_dir


app = typer.Typer(add_completion=False, help="Extract 3–5 evenly spaced frames per clip (JPG) for CLIP inference.")
console = Console()


def _iter_clips(clips_root: Path) -> list[Path]:
    return sorted(clips_root.rglob("*.mp4"))


@app.command()
def run(
    clips_root: Path = typer.Option(None, help="Root dir for clips; defaults to DATA_DIR/clips"),
    frames_root: Path = typer.Option(None, help="Root dir for frames; defaults to DATA_DIR/frames"),
    frames_per_clip: int = typer.Option(5, help="Number of frames per clip (3–5 recommended)."),
    scale_width: int = typer.Option(512, help="Resize frames to this width for faster CLIP inference."),
    overwrite: bool = typer.Option(False, help="Overwrite existing frames."),
):
    run_job(
        clips_root=clips_root,
        frames_root=frames_root,
        frames_per_clip=frames_per_clip,
        scale_width=scale_width,
        overwrite=overwrite,
    )


def run_job(
    *,
    clips_root: Path | None = None,
    frames_root: Path | None = None,
    frames_per_clip: int = 5,
    scale_width: int = 512,
    overwrite: bool = False,
):
    clips_root = clips_root or (data_dir() / "clips")
    frames_root = frames_root or (data_dir() / "frames")
    frames_root.mkdir(parents=True, exist_ok=True)

    clips = _iter_clips(clips_root)
    console.print(f"[bold]Frame extraction[/bold] clips={len(clips)} -> {frames_root}")

    failed_clips: list[tuple[str, str]] = []  # (clip_id, error_message)
    skipped_clips: list[str] = []  # clip_ids with invalid duration

    with Progress() as progress:
        task = progress.add_task("Extracting", total=len(clips))
        for clip_path in clips:
            progress.advance(task)
            clip_id = clip_path.stem
            out_dir = frames_root / clip_id

            if out_dir.exists() and not overwrite:
                # If it has at least 1 jpg, treat as done.
                if any(out_dir.glob("*.jpg")):
                    continue

            # Check if file exists and is readable
            if not clip_path.exists():
                failed_clips.append((clip_id, "Clip file does not exist"))
                continue

            try:
                info = ffprobe_info(clip_path)
                duration = info.duration_seconds
                if duration <= 0:
                    skipped_clips.append(clip_id)
                    console.print(f"[yellow]Skipping[/yellow] {clip_id}: invalid duration ({duration})")
                    continue
            except Exception as e:
                failed_clips.append((clip_id, f"ffprobe failed: {str(e)}"))
                continue

            n = int(max(1, frames_per_clip))
            # Evenly spaced timestamps excluding exact endpoints.
            # Clamp to valid range for very short clips.
            start = min(0.05, max(0.0, duration / 10.0))
            end = max(start, duration - 0.05)
            if end <= start:
                ts = [0.0]
            else:
                ts = np.linspace(start, end, num=n).tolist()

            # Clean if overwrite
            if overwrite and out_dir.exists():
                for p in out_dir.glob("*.jpg"):
                    try:
                        p.unlink()
                    except Exception:
                        pass

            try:
                extracted = extract_frames_jpg(
                    clip_path,
                    out_dir,
                    timestamps=ts,
                    scale_width=scale_width,
                )
                if not extracted:
                    failed_clips.append((clip_id, "No frames extracted (ffmpeg returned empty)"))
            except Exception as e:
                failed_clips.append((clip_id, f"Frame extraction failed: {str(e)}"))
                # Clean up empty directory on failure
                if out_dir.exists() and not any(out_dir.glob("*.jpg")):
                    try:
                        out_dir.rmdir()
                    except Exception:
                        pass

    # Summary
    successful = len(clips) - len(failed_clips) - len(skipped_clips)
    console.print(f"\n[bold]Frame extraction summary:[/bold]")
    console.print(f"  [green]Successful:[/green] {successful}/{len(clips)}")
    if skipped_clips:
        console.print(f"  [yellow]Skipped (invalid duration):[/yellow] {len(skipped_clips)}")
    if failed_clips:
        console.print(f"  [red]Failed:[/red] {len(failed_clips)}")
        console.print(f"\n[dim]Failed clips (showing first 20):[/dim]")
        for clip_id, error in failed_clips[:20]:
            console.print(f"  [red]{clip_id}[/red]: {error}")
        if len(failed_clips) > 20:
            console.print(f"  ... and {len(failed_clips) - 20} more")


if __name__ == "__main__":
    app()


