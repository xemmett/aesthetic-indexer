from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from src.audio.audio_tagger import AudioTagger, AudioTagResult
from src.utils.ffmpeg import extract_audio_wav
from src.utils.io import write_json
from src.utils.paths import data_dir


app = typer.Typer(add_completion=False, help="Run audio tagging for extracted clips.")
console = Console()


def _iter_clips(clips_root: Path) -> list[Path]:
    return sorted(clips_root.rglob("*.mp4"))


@app.command()
def run(
    clips_root: Path = typer.Option(None, help="Root dir for clips; defaults to DATA_DIR/clips"),
):
    run_job(clips_root=clips_root)


def run_job(
    *,
    clips_root: Path | None = None,
):
    clips_root = clips_root or (data_dir() / "clips")
    clips = _iter_clips(clips_root)
    console.print(f"[bold]Audio tagging[/bold] clips={len(clips)}")

    out_dir = data_dir() / "tags" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_audio_dir = data_dir() / "audio_tmp"
    tmp_audio_dir.mkdir(parents=True, exist_ok=True)

    # Create reusable audio tagger instance
    tagger = AudioTagger()

    with Progress() as progress:
        task = progress.add_task("Tagging", total=len(clips))
        for clip_path in clips:
            progress.advance(task)
            clip_id = clip_path.stem
            out_json = out_dir / f"{clip_id}.json"
            if out_json.exists():
                continue

            wav = tmp_audio_dir / f"{clip_id}.wav"
            try:
                wav_path = extract_audio_wav(clip_path, wav, sample_rate=16000, mono=True)
                if wav_path is None:
                    # Video has no audio stream - create silent audio result
                    res = AudioTagResult(rms=0.0, spectral_centroid=0.0, silence_ratio=1.0, tags={"silence": 1.0})
                else:
                    res = tagger.analyze(wav_path)
                
                write_json(
                    out_json,
                    {
                        "clip_id": clip_id,
                        "rms": res.rms,
                        "spectral_centroid": res.spectral_centroid,
                        "silence_ratio": res.silence_ratio,  # first-class
                        "tags": res.tags,
                    },
                )
            finally:
                # best-effort cleanup
                if wav.exists():
                    try:
                        wav.unlink()
                    except Exception:
                        pass


if __name__ == "__main__":
    app()


