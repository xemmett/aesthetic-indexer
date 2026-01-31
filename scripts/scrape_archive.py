from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import internetarchive as ia
import typer
from rich.console import Console
from rich.progress import Progress

from src.utils.io import write_json
from src.utils.paths import data_dir


app = typer.Typer(add_completion=False, help="Scrape Internet Archive collections (original video files only).")
console = Console()


DEFAULT_COLLECTIONS = (
    "war"
)
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".mpeg", ".mpg", ".m4v"}


def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip())
    return s[:180] if len(s) > 180 else s


@dataclass(frozen=True)
class ArchiveItem:
    identifier: str
    title: str
    year: Optional[int]


def _read_keywords_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    kws: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        kws.append(s)
    return kws


def _build_keyword_query(keywords: list[str]) -> str:
    """
    Build a simple IA advanced-search query from keywords.
    We keep this conservative and IA-compatible.
    """

    def _esc(s: str) -> str:
        return s.replace('"', '\\"')

    # IMPORTANT:
    # Field-scoped queries like title:("...") are often too restrictive on IA and can
    # produce empty result sets. Prefer broad queries without field prefixes.
    clauses: list[str] = []
    for kw in keywords:
        k = _esc(kw)
        clauses.append(f'("{k}")')
    return "(" + " OR ".join(clauses) + ") AND mediatype:(movies)"


def _iter_items_for_query(query: str, rows: int = 200) -> Iterable[ArchiveItem]:
    for r in ia.search_items(query, fields=["identifier", "title", "year"], params={"rows": rows}):
        identifier = (r or {}).get("identifier") or ""
        if not identifier:
            continue
        title = (r or {}).get("title") or identifier
        year_raw = (r or {}).get("year")
        year = None
        try:
            if year_raw is not None:
                year = int(str(year_raw)[:4])
        except Exception:
            year = None
        yield ArchiveItem(identifier=identifier, title=title, year=year)


def _iter_items(collection: str, rows: int = 200) -> Iterable[ArchiveItem]:
    query = f"collection:{collection} AND mediatype:(movies)"
    # internetarchive>=5 uses ia.search_items(query, fields=..., params=...)
    for r in ia.search_items(query, fields=["identifier", "title", "year"], params={"rows": rows}):
        identifier = (r or {}).get("identifier") or ""
        if not identifier:
            continue
        title = (r or {}).get("title") or identifier
        year_raw = (r or {}).get("year")
        year = None
        try:
            if year_raw is not None:
                year = int(str(year_raw)[:4])
        except Exception:
            year = None
        yield ArchiveItem(identifier=identifier, title=title, year=year)


def _choose_original_video_file(item: ia.Item) -> Optional[dict]:
    """
    Internet Archive provides multiple derivatives. We want the most original-ish file.
    Heuristic: prefer files with 'source' == 'original' and a video extension.
    """
    files = item.files or []
    candidates = []
    for f in files:
        name = f.get("name") or ""
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXTS:
            continue
        source = (f.get("source") or "").lower()
        if source == "original":
            candidates.append((0, f))
        else:
            candidates.append((1, f))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], -(int(t[1].get("size") or 0))))
    return candidates[0][1]

def _parse_size_bytes(v: object) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(str(v).strip())
        except Exception:
            return None


def _format_mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.1f}MB"


def _download_items(
    *,
    items: list[ArchiveItem],
    target_root: Path,
    metadata_base: dict,
    max_filesize_mb: Optional[float] = None,
) -> None:
    with Progress() as progress:
        task = progress.add_task("Downloading (items)", total=len(items))
        for it in items:
            try:
                item = ia.get_item(it.identifier)
                f = _choose_original_video_file(item)
                if not f:
                    progress.advance(task)
                    continue
                name = f.get("name") or f"{it.identifier}.mp4"
                size_bytes = _parse_size_bytes(f.get("size"))
                if max_filesize_mb is not None and size_bytes is not None:
                    if size_bytes > int(max_filesize_mb * 1024 * 1024):
                        console.print(
                            f"Skipping {it.identifier} :: {name} (size {_format_mb(size_bytes)} > {max_filesize_mb}MB)"
                        )
                        progress.advance(task)
                        continue
                safe_name = _sanitize_filename(name)
                target_dir = target_root / it.identifier
                target_dir.mkdir(parents=True, exist_ok=True)

                video_path = target_dir / safe_name
                meta_path = target_dir / "metadata.json"

                if not meta_path.exists():
                    write_json(
                        meta_path,
                        {
                            **metadata_base,
                            "identifier": it.identifier,
                            "title": it.title,
                            "year": it.year,
                            "file": f,
                        },
                    )

                if video_path.exists():
                    progress.advance(task)
                    continue

                console.print(f"Downloading {it.identifier} :: {name}")

                # Download exact file to the target_dir without creating nested directories.
                # Using File.download avoids the extra {identifier}/ subdir created by Item.download.
                ia_file = item.get_file(name)
                ia_file.download(destdir=str(target_dir), ignore_existing=True, verbose=True)

                progress.advance(task)
            except Exception as e:
                console.print(f"[red]Failed[/red] {it.identifier}: {e}")
                progress.advance(task)


@app.command()
def run(
    collections: list[str] = typer.Option(list(DEFAULT_COLLECTIONS), help="IA collections to scrape"),
    max_filesize_mb: float = typer.Option(None, help="Skip downloads larger than this many MB (based on IA metadata)."),
    limit: int = typer.Option(50, help="Max items per collection to attempt"),
    rows: int = typer.Option(200, help="Search page size for IA"),
    out_dir: Path = typer.Option(None, help="Output dir; defaults to DATA_DIR/raw_videos/archive"),
):
    run_job(
        collections=collections,
        max_filesize_mb=max_filesize_mb,
        limit=limit,
        rows=rows,
        out_dir=out_dir,
    )


def run_job(
    *,
    collections: list[str] | None = None,
    max_filesize_mb: float | None = None,
    limit: int = 50,
    rows: int = 200,
    out_dir: Path | None = None,
):
    """
    Downloads one original video per item (best-effort) and stores metadata JSON alongside.
    Idempotent: skips downloads if the target file already exists.
    """
    if collections is None:
        collections = list(DEFAULT_COLLECTIONS)
    base = out_dir or (data_dir() / "raw_videos" / "archive")
    base.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Internet Archive scrape[/bold] -> {base}")

    for collection in collections:
        console.print(f"\n[bold]Collection[/bold]: {collection}")
        items = list(_iter_items(collection, rows=rows))[:limit]
        if not items:
            console.print("No items found.")
            continue

        _download_items(
            items=items,
            target_root=base / collection,
            metadata_base={"source": "archive", "collection": collection},
            max_filesize_mb=max_filesize_mb,
        )


@app.command("keywords")
def run_keywords(
    keyword: list[str] = typer.Option([], "--keyword", "-k", help="Keyword (repeatable)."),
    keywords_file: Path = typer.Option(None, "--keywords-file", help="Text file of keywords, one per line."),
    max_filesize_mb: float = typer.Option(None, help="Skip downloads larger than this many MB (based on IA metadata)."),
    limit: int = typer.Option(50, help="Max items to attempt"),
    rows: int = typer.Option(200, help="Search page size for IA"),
    out_dir: Path = typer.Option(None, help="Output dir; defaults to DATA_DIR/raw_videos/archive"),
):
    run_keywords_job(
        keyword=keyword,
        keywords_file=keywords_file,
        max_filesize_mb=max_filesize_mb,
        limit=limit,
        rows=rows,
        out_dir=out_dir,
    )


def run_keywords_job(
    *,
    keyword: list[str] | None = None,
    keywords_file: Path | None = None,
    max_filesize_mb: float | None = None,
    limit: int = 50,
    rows: int = 200,
    out_dir: Path | None = None,
):
    """
    Keyword-based Internet Archive search (instead of collection filtering).
    Downloads original-ish movie files and stores metadata alongside.
    """
    keywords = list(keyword or [])
    if keywords_file:
        if not keywords_file.exists():
            raise typer.BadParameter(f"keywords_file not found: {keywords_file}")
        keywords.extend(_read_keywords_file(keywords_file))
    keywords = [k.strip() for k in keywords if k.strip()]
    if not keywords:
        raise typer.BadParameter("Provide at least one --keyword or --keywords-file.")

    base = out_dir or (data_dir() / "raw_videos" / "archive")
    base.mkdir(parents=True, exist_ok=True)

    # Strategy: search per keyword (broad query) and union results.
    # This avoids a single giant OR query and makes it obvious which keyword is yielding hits.
    console.print(f"[bold]Internet Archive keyword search[/bold] -> {base}")

    seen: set[str] = set()
    items: list[ArchiveItem] = []
    for kw in keywords:
        q = f'("{kw}") AND mediatype:(movies)'
        console.print(f"[bold]Keyword[/bold]: {kw}")
        found_this_kw = 0
        for it in _iter_items_for_query(q, rows=rows):
            if it.identifier in seen:
                continue
            seen.add(it.identifier)
            items.append(it)
            found_this_kw += 1
            if len(items) >= limit:
                break
        console.print(f"  found {found_this_kw}")
        if len(items) >= limit:
            break

    if not items:
        console.print("No items found.")
        return

    _download_items(
        items=items,
        target_root=base / "keyword_search",
        metadata_base={"source": "archive", "keywords": keywords},
        max_filesize_mb=max_filesize_mb,
    )


if __name__ == "__main__":
    app()


