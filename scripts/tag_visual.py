from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress

from src.utils.io import write_json
from src.utils.paths import data_dir, prompts_file
from src.vision.clip_tagger import load_prompt_bank, tag_frames_with_clip


app = typer.Typer(add_completion=False, help="Run CLIP visual tagging for extracted clips (raw similarity scores).")
console = Console()


def _iter_clips(clips_root: Path) -> list[Path]:
    return sorted(clips_root.rglob("*.mp4"))


@app.command()
def run(
    clips_root: Path = typer.Option(None, help="Root dir for clips; defaults to DATA_DIR/clips"),
    frames_root: Path = typer.Option(None, help="Root dir for frames; defaults to DATA_DIR/frames"),
    prompts_path: Path = typer.Option(None, help="Prompt bank file; defaults to PROMPTS_FILE"),
):
    run_job(clips_root=clips_root, frames_root=frames_root, prompts_path=prompts_path)


def run_job(
    *,
    clips_root: Path | None = None,
    frames_root: Path | None = None,
    prompts_path: Path | None = None,
):
    clips_root = clips_root or (data_dir() / "clips")
    frames_root = frames_root or (data_dir() / "frames")
    prompts_path = prompts_path or prompts_file()

    prompts = load_prompt_bank(prompts_path)
    clips = _iter_clips(clips_root)
    console.print(f"[bold]Visual tagging[/bold] clips={len(clips)} prompts={len(prompts)}")

    out_tags_dir = data_dir() / "tags" / "visual"
    out_emb_dir = data_dir() / "embeddings"
    out_tags_dir.mkdir(parents=True, exist_ok=True)
    out_emb_dir.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("Tagging", total=len(clips))
        for clip_path in clips:
            progress.advance(task)
            clip_id = clip_path.stem

            out_json = out_tags_dir / f"{clip_id}.json"
            out_npy = out_emb_dir / f"{clip_id}.npy"
            if out_json.exists() and out_npy.exists():
                continue

            clip_frames_dir = frames_root / clip_id
            frame_paths = sorted(clip_frames_dir.glob("*.jpg"))
            if not frame_paths:
                # Not an error; frame extraction may not have run yet.
                continue

            res = tag_frames_with_clip(frame_paths=frame_paths, prompts=prompts)
            np.save(out_npy, res.embedding)
            write_json(
                out_json,
                {
                    "clip_id": clip_id,
                    "tags": res.tags,  # raw similarity scores
                },
            )


if __name__ == "__main__":
    app()


