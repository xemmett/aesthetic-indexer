from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from src.utils.io import write_json
from src.utils.paths import data_dir


app = typer.Typer(add_completion=False, help="Scrape YouTube via yt-dlp from keyword list.")
console = Console()


@app.command()
def run(
    keywords_file: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="Text file of keywords, one per line."),
    limit_per_keyword: int = typer.Option(10, help="Max videos per keyword (best-effort)."),
    max_filesize_mb: int = typer.Option(None, help="yt-dlp --max-filesize in MB (skips larger downloads)."),
    out_dir: Optional[Path] = typer.Option(None, help="Output dir; defaults to DATA_DIR/raw_videos/youtube"),
):
    run_job(
        keywords_file=keywords_file,
        limit_per_keyword=limit_per_keyword,
        max_filesize_mb=max_filesize_mb,
        out_dir=out_dir,
    )


def run_job(
    *,
    keywords_file: Path,
    limit_per_keyword: int = 10,
    max_filesize_mb: Optional[int] = None,
    out_dir: Optional[Path] = None,
):
    """
    Downloads full videos and writes metadata JSON alongside.
    Idempotent: skips if output folder already contains a video file.
    """
    base = out_dir or (data_dir() / "raw_videos" / "youtube")
    base.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]YouTube scrape[/bold] -> {base}")

    keywords = [k.strip() for k in keywords_file.read_text(encoding="utf-8").splitlines() if k.strip() and not k.strip().startswith("#")]
    if not keywords:
        raise typer.BadParameter("keywords_file has no keywords.")

    for kw in keywords:
        kw_dir = base / kw.replace(" ", "_")
        kw_dir.mkdir(parents=True, exist_ok=True)

        # yt-dlp output template: keep per-video folder by id
        # Note: We use subprocess to avoid locking to a Python API surface.
        # Requires yt-dlp on PATH (installed via pip provides entrypoint).
        # We request best video+audio in mp4 container when possible.
        import subprocess

        cmd = [
            "yt-dlp",
            f"ytsearch{limit_per_keyword}:{kw}",
            "-P",
            str(kw_dir),
            "-o",
            "%(id)s/%(id)s.%(ext)s",
            "--write-info-json",
            "--no-playlist",
            "--merge-output-format",
            "mp4",
            "-f",
            "bv*+ba/best",
        ]
        if max_filesize_mb is not None:
            cmd += ["--max-filesize", f"{int(max_filesize_mb)}M"]
        console.print(f"[cyan]Keyword[/cyan]: {kw}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            console.print(f"[red]yt-dlp failed[/red] for '{kw}':\n{proc.stderr}")
            continue

        # Also write a small run manifest for reproducibility.
        write_json(
            kw_dir / "scrape_manifest.json",
            {
                "source": "youtube",
                "keyword": kw,
                "limit_per_keyword": limit_per_keyword,
                "cmd": cmd,
            },
        )


if __name__ == "__main__":
    app()


