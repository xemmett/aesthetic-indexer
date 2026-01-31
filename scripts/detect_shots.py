from __future__ import annotations

import csv
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress
from scenedetect import open_video
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager

from src.utils.paths import data_dir
from src.utils.video_discovery import discover_raw_videos


app = typer.Typer(add_completion=False, help="Detect shots using PySceneDetect (content-aware).")
console = Console()


def _shot_csv_path(shots_dir: Path, source: str, video_id: str) -> Path:
    # Avoid collisions between sources.
    return shots_dir / f"{source}__{video_id}.csv"


@app.command()
def run(
    raw_videos_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/raw_videos"),
    shots_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/shots"),
    threshold: float = typer.Option(27.0, help="ContentDetector threshold (lower=more cuts)."),
    min_scene_len_frames: int = typer.Option(15, help="Minimum scene length in frames."),
    overwrite: bool = typer.Option(False, help="Overwrite existing shot CSVs."),
):
    run_job(
        raw_videos_dir=raw_videos_dir,
        shots_dir=shots_dir,
        threshold=threshold,
        min_scene_len_frames=min_scene_len_frames,
        overwrite=overwrite,
    )


def run_job(
    *,
    raw_videos_dir: Path | None = None,
    shots_dir: Path | None = None,
    threshold: float = 27.0,
    min_scene_len_frames: int = 15,
    overwrite: bool = False,
):
    """
    Outputs CSV files in data/shots/ as:
      source__video_id.csv
    with (start_time, end_time) in seconds.
    """
    raw_videos_dir = raw_videos_dir or (data_dir() / "raw_videos")
    shots_dir = shots_dir or (data_dir() / "shots")
    shots_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_raw_videos(raw_videos_dir)
    console.print(f"[bold]Shot detection[/bold] videos={len(videos)} -> {shots_dir}")

    with Progress() as progress:
        task = progress.add_task("Detecting", total=len(videos))
        for v in videos:
            progress.advance(task)
            out_csv = _shot_csv_path(shots_dir, v.source, v.video_id)
            if out_csv.exists() and not overwrite:
                continue

            try:
                video = open_video(str(v.filepath))
                scene_manager = SceneManager()
                scene_manager.add_detector(
                    ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
                )
                scene_manager.detect_scenes(video, show_progress=False)
                scenes = scene_manager.get_scene_list()

                # Write start/end in seconds.
                with out_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["start_time", "end_time"])
                    for start, end in scenes:
                        w.writerow([start.get_seconds(), end.get_seconds()])
            except Exception as e:
                console.print(f"[red]Failed[/red] {v.source}/{v.video_id}: {e}")


if __name__ == "__main__":
    app()


