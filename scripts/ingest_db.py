from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress
from sqlalchemy import select

from src.db.models import Clip, ClipEmbedding, ClipSignals, ClipTag
from src.db.session import session_scope
from src.utils.deduplication import phash_distance
from src.utils.io import read_json
from src.utils.paths import data_dir


app = typer.Typer(add_completion=False, help="Ingest clips + embeddings + tags + signals into PostgreSQL/pgvector.")
console = Console()


def _iter_clip_meta(clips_root: Path) -> list[Path]:
    return sorted(clips_root.rglob("*.json"))

def _to_uuid(value: object) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


@app.command()
def run(
    clips_root: Path = typer.Option(None, help="Root dir for clips; defaults to DATA_DIR/clips"),
    visual_tags_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/tags/visual"),
    audio_tags_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/tags/audio"),
    signals_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/signals"),
    embeddings_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/embeddings"),
    phash_max_distance: int = typer.Option(None, help="Override env PHASH_MAX_DISTANCE for DB dedup check."),
):
    run_job(
        clips_root=clips_root,
        visual_tags_dir=visual_tags_dir,
        audio_tags_dir=audio_tags_dir,
        signals_dir=signals_dir,
        embeddings_dir=embeddings_dir,
        phash_max_distance=phash_max_distance,
    )


def run_job(
    *,
    clips_root: Path | None = None,
    visual_tags_dir: Path | None = None,
    audio_tags_dir: Path | None = None,
    signals_dir: Path | None = None,
    embeddings_dir: Path | None = None,
    phash_max_distance: int | None = None,
):
    """
    - Inserts into clips, clip_embeddings, clip_tags, clip_signals.
    - Stores raw CLIP similarity scores (FLOAT).
    - Enforces silence_ratio NOT NULL via ClipSignals.
    - Best-effort dedup: if a candidate clip has pHash very close to an existing one, skip.
    """
    clips_root = clips_root or (data_dir() / "clips")
    visual_tags_dir = visual_tags_dir or (data_dir() / "tags" / "visual")
    audio_tags_dir = audio_tags_dir or (data_dir() / "tags" / "audio")
    signals_dir = signals_dir or (data_dir() / "signals")
    embeddings_dir = embeddings_dir or (data_dir() / "embeddings")

    metas = _iter_clip_meta(clips_root)
    console.print(f"[bold]DB ingest[/bold] clips={len(metas)}")

    import os

    max_dist = phash_max_distance
    if max_dist is None:
        try:
            max_dist = int(os.getenv("PHASH_MAX_DISTANCE") or "4")
        except Exception:
            max_dist = 4

    with session_scope() as session:
        with Progress() as progress:
            task = progress.add_task("Ingesting", total=len(metas))
            for meta_path in metas:
                progress.advance(task)
                meta = read_json(meta_path)

                clip_id_raw = meta.get("clip_id")
                if not clip_id_raw:
                    continue
                try:
                    clip_id = _to_uuid(clip_id_raw)
                except Exception:
                    # If clip_id isn't a UUID, skip; DB schema requires UUID.
                    continue

                # Already ingested?
                exists = session.execute(
                    select(Clip.id).where(Clip.id == clip_id)
                ).scalar_one_or_none()
                if exists is not None:
                    continue

                phash = meta.get("perceptual_hash")
                if phash:
                    # Best-effort dedup against existing pHashes.
                    existing = session.execute(
                        select(Clip.perceptual_hash).where(Clip.perceptual_hash.is_not(None))
                    ).scalars().all()
                    is_dup = any(phash_distance(phash, h) <= max_dist for h in existing if h)
                    if is_dup:
                        continue

                clip = Clip(
                    id=clip_id,
                    source=meta["source"],
                    video_id=meta["video_id"],
                    filepath=meta["filepath"],
                    start_time=float(meta["start_time"]),
                    end_time=float(meta["end_time"]),
                    duration=float(meta["duration"]),
                    year=meta.get("year"),
                    perceptual_hash=phash,
                )
                session.add(clip)

                # Embedding + visual tags
                emb_path = embeddings_dir / f"{clip_id}.npy"
                vis_path = visual_tags_dir / f"{clip_id}.json"
                if emb_path.exists():
                    emb = np.load(emb_path).astype(np.float32).reshape(-1)
                    if emb.shape[0] == 512:
                        session.add(ClipEmbedding(clip_id=clip_id, embedding=emb.tolist()))

                if vis_path.exists():
                    vis = read_json(vis_path)
                    tags = vis.get("tags") or {}
                    for tag, score in tags.items():
                        try:
                            session.add(
                                ClipTag(
                                    clip_id=clip_id,
                                    tag=str(tag),
                                    similarity_score=float(score),
                                )
                            )
                        except Exception:
                            continue

                # Signals (required silence_ratio)
                sig_path = signals_dir / f"{clip_id}.json"
                if sig_path.exists():
                    sig = read_json(sig_path)
                    silence_ratio = sig.get("silence_ratio")
                    if silence_ratio is None:
                        # Enforce not-null requirement: skip inserting signals if missing.
                        silence_ratio = 1.0
                    session.add(
                        ClipSignals(
                            clip_id=clip_id,
                            motion_score=sig.get("motion_score"),
                            noise_level=sig.get("noise_level"),
                            silence_ratio=float(silence_ratio),
                            brightness_entropy=sig.get("brightness_entropy"),
                        )
                    )
                else:
                    # Enforce not-null: insert a baseline signals row if absent.
                    session.add(ClipSignals(clip_id=clip_id, silence_ratio=1.0))

                # Audio tag json is currently redundant (signals contain silence_ratio); keep on disk.
                # Could be ingested later into a separate table if desired.


if __name__ == "__main__":
    app()


