from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from src.utils.paths import data_dir


app = typer.Typer(add_completion=False, help="Resumable pipeline orchestrator (skip completed stages by default).")
console = Console()


def _marker(stage: str) -> Path:
    return data_dir() / ".pipeline" / f"{stage}.completed"


def _is_done(stage: str) -> bool:
    return _marker(stage).exists()


def _mark_done(stage: str) -> None:
    p = _marker(stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ok\n", encoding="utf-8")

def _stage_scrape_archive(
    *,
    skip_if_done: bool = True,
    limit: int = 50,
    max_filesize_mb: float | None = None,
) -> None:
    stage = "scrape_archive"
    if skip_if_done and _is_done(stage):
        return
    from scripts.scrape_archive import run_job as _run

    _run(limit=limit, max_filesize_mb=max_filesize_mb)
    _mark_done(stage)

def _stage_scrape_archive_keywords(
    *,
    keywords_file: Path,
    skip_if_done: bool = True,
    limit: int = 50,
    max_filesize_mb: float | None = None,
) -> None:
    stage = "scrape_archive_keywords"
    if skip_if_done and _is_done(stage):
        return
    from scripts.scrape_archive import run_keywords_job as _run

    _run(keywords_file=keywords_file, limit=limit, max_filesize_mb=max_filesize_mb)
    _mark_done(stage)


def _stage_scrape_youtube(*, keywords_file: Path, skip_if_done: bool = True) -> None:
    stage = "scrape_youtube"
    if skip_if_done and _is_done(stage):
        return
    from scripts.scrape_youtube import run_job as _run

    _run(keywords_file=keywords_file)
    _mark_done(stage)


def _stage_detect_shots(*, skip_if_done: bool = True) -> None:
    stage = "detect_shots"
    if skip_if_done and _is_done(stage):
        return
    from scripts.detect_shots import run_job as _run

    _run()
    _mark_done(stage)


def _stage_extract_clips(*, skip_if_done: bool = True) -> None:
    stage = "extract_clips"
    if skip_if_done and _is_done(stage):
        return
    from scripts.extract_clips import run_job as _run

    _run()
    _mark_done(stage)


def _stage_extract_frames(*, skip_if_done: bool = True) -> None:
    stage = "extract_frames"
    if skip_if_done and _is_done(stage):
        return
    from scripts.extract_frames import run_job as _run

    _run()
    _mark_done(stage)


def _stage_tag_visual(*, skip_if_done: bool = True) -> None:
    stage = "tag_visual"
    if skip_if_done and _is_done(stage):
        return
    from scripts.tag_visual import run_job as _run

    _run()
    _mark_done(stage)


def _stage_tag_audio(*, skip_if_done: bool = True) -> None:
    stage = "tag_audio"
    if skip_if_done and _is_done(stage):
        return
    from scripts.tag_audio import run_job as _run

    _run()
    _mark_done(stage)


def _stage_compute_signals(*, skip_if_done: bool = True) -> None:
    stage = "compute_signals"
    if skip_if_done and _is_done(stage):
        return
    from scripts.compute_signals import run_job as _run

    _run()
    _mark_done(stage)


def _stage_ingest_db(*, skip_if_done: bool = True) -> None:
    stage = "ingest_db"
    if skip_if_done and _is_done(stage):
        return
    from scripts.ingest_db import run_job as _run

    _run()
    _mark_done(stage)


@app.command()
def scrape_archive(
    skip_if_done: bool = typer.Option(True),
    limit: int = typer.Option(50, "--archive-limit", min=1, help="Max items per collection."),
    max_filesize_mb: float = typer.Option(
        None,
        "--archive-max-filesize-mb",
        help="Skip downloads larger than this many MB (IA metadata).",
    ),
):
    _stage_scrape_archive(skip_if_done=skip_if_done, limit=limit, max_filesize_mb=max_filesize_mb)

@app.command()
def scrape_archive_keywords(
    keywords_file: Path = typer.Option(
        ...,
        "--keywords-file",
        "--archive-keywords-file",
        help="Text file of keywords, one per line.",
    ),
    skip_if_done: bool = typer.Option(True),
    limit: int = typer.Option(50, "--archive-limit", min=1, help="Max total items across keywords."),
    max_filesize_mb: float = typer.Option(
        None,
        "--archive-max-filesize-mb",
        help="Skip downloads larger than this many MB (IA metadata).",
    ),
):
    _stage_scrape_archive_keywords(
        keywords_file=keywords_file,
        skip_if_done=skip_if_done,
        limit=limit,
        max_filesize_mb=max_filesize_mb,
    )


@app.command()
def scrape_youtube(keywords_file: Path = typer.Option(None), skip_if_done: bool = typer.Option(True)):
    if keywords_file is None:
        raise typer.BadParameter("keywords_file is required for scrape-youtube.")
    _stage_scrape_youtube(keywords_file=keywords_file, skip_if_done=skip_if_done)


@app.command()
def detect_shots(skip_if_done: bool = typer.Option(True)):
    _stage_detect_shots(skip_if_done=skip_if_done)


@app.command()
def extract_clips(skip_if_done: bool = typer.Option(True)):
    _stage_extract_clips(skip_if_done=skip_if_done)


@app.command()
def extract_frames(skip_if_done: bool = typer.Option(True)):
    _stage_extract_frames(skip_if_done=skip_if_done)


@app.command()
def tag_visual(skip_if_done: bool = typer.Option(True)):
    _stage_tag_visual(skip_if_done=skip_if_done)


@app.command()
def tag_audio(skip_if_done: bool = typer.Option(True)):
    _stage_tag_audio(skip_if_done=skip_if_done)


@app.command()
def compute_signals(skip_if_done: bool = typer.Option(True)):
    _stage_compute_signals(skip_if_done=skip_if_done)


@app.command()
def ingest_db(skip_if_done: bool = typer.Option(True)):
    _stage_ingest_db(skip_if_done=skip_if_done)


@app.command()
def run_all(
    keywords_file: Path = typer.Option(None, help="Only needed if running scrape-youtube."),
    archive_keywords_file: Path = typer.Option(None, help="If set, scrapes IA by keywords instead of default collections."),
    archive_limit: int = typer.Option(50, "--archive-limit", min=1, help="Max IA items (per collection or total across keywords)."),
    archive_max_filesize_mb: float = typer.Option(
        None,
        "--archive-max-filesize-mb",
        help="Skip IA downloads larger than this many MB (IA metadata).",
    ),
):
    """
    Runs all stages (archive scrape + optional youtube scrape if keywords_file provided).
    Safe to re-run; stages are skip-completed by default.
    """
    if archive_keywords_file is not None:
        _stage_scrape_archive_keywords(
            keywords_file=archive_keywords_file,
            limit=archive_limit,
            max_filesize_mb=archive_max_filesize_mb,
        )
    else:
        _stage_scrape_archive(limit=archive_limit, max_filesize_mb=archive_max_filesize_mb)
    if keywords_file is not None:
        _stage_scrape_youtube(keywords_file=keywords_file)
    _stage_detect_shots()
    _stage_extract_clips()
    _stage_extract_frames()
    _stage_tag_visual()
    _stage_tag_audio()
    _stage_compute_signals()
    _stage_ingest_db()


if __name__ == "__main__":
    app()


