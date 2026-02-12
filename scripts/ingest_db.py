from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress
from sqlalchemy import select, text

from src.db.models import Clip, ClipEmbedding, ClipEntity, ClipScene, ClipSignals, ClipTag
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
    entities_dir: Path = typer.Option(None, help="Defaults to DATA_DIR/entities"),
    phash_max_distance: int = typer.Option(None, help="Override env PHASH_MAX_DISTANCE for DB dedup check."),
    overwrite_entities: bool = typer.Option(False, help="If set, replace clip_entities/clip_scenes rows for each clip."),
):
    run_job(
        clips_root=clips_root,
        visual_tags_dir=visual_tags_dir,
        audio_tags_dir=audio_tags_dir,
        signals_dir=signals_dir,
        embeddings_dir=embeddings_dir,
        entities_dir=entities_dir,
        phash_max_distance=phash_max_distance,
        overwrite_entities=overwrite_entities,
    )


def run_job(
    *,
    clips_root: Path | None = None,
    visual_tags_dir: Path | None = None,
    audio_tags_dir: Path | None = None,
    signals_dir: Path | None = None,
    embeddings_dir: Path | None = None,
    entities_dir: Path | None = None,
    phash_max_distance: int | None = None,
    overwrite_entities: bool = False,
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
    entities_dir = entities_dir or (data_dir() / "entities")

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
        # Friendly preflight: make sure core schema exists.
        try:
            reg = session.execute(text("SELECT to_regclass('clips')")).scalar_one()
        except Exception:
            reg = None
        if reg is None:
            raise RuntimeError(
                "Database schema not initialized: table `clips` does not exist. "
                "Run migrations/001_initial_schema.sql (and 002_add_entities.sql) against DATABASE_URL."
            )

        # Load all existing pHashes once for deduplication (much faster than per-clip queries)
        console.print("[dim]Loading existing pHashes for deduplication...[/dim]")
        existing_phashes = set(
            session.execute(
                select(Clip.perceptual_hash).where(Clip.perceptual_hash.is_not(None))
            ).scalars().all()
        )
        console.print(f"[green]Loaded {len(existing_phashes)} existing pHashes.[/green]")

        # Load all existing clip IDs once to avoid per-clip queries
        console.print("[dim]Loading existing clip IDs...[/dim]")
        existing_clip_ids = set(
            session.execute(select(Clip.id)).scalars().all()
        )
        existing_embeddings = set(
            session.execute(select(ClipEmbedding.clip_id)).scalars().all()
        )
        existing_tags = set(
            session.execute(select(ClipTag.clip_id)).scalars().all()
        )
        existing_signals = set(
            session.execute(select(ClipSignals.clip_id)).scalars().all()
        )
        existing_entities = set(
            session.execute(select(ClipEntity.clip_id)).scalars().all()
        )
        existing_scenes = set(
            session.execute(select(ClipScene.clip_id)).scalars().all()
        )
        console.print(f"[green]Loaded existing IDs: {len(existing_clip_ids)} clips, {len(existing_embeddings)} embeddings, {len(existing_tags)} tags, {len(existing_signals)} signals.[/green]")

        # Batch processing: collect inserts and commit in batches
        BATCH_SIZE = 100
        clips_to_insert: list[dict] = []
        pending_clips: dict[uuid.UUID, dict] = {}  # Track pending clips by ID for entity_stats updates
        embeddings_to_insert: list[dict] = []
        tags_to_insert: list[dict] = []
        signals_to_insert: list[dict] = []
        entities_to_insert: list[dict] = []
        scenes_to_insert: list[dict] = []
        clips_to_update: list[dict] = []

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

                # Check if clip already exists
                is_new_clip = clip_id not in existing_clip_ids

                phash = meta.get("perceptual_hash")
                clip_data = {
                    "id": clip_id,
                    "source": meta["source"],
                    "video_id": meta["video_id"],
                    "filepath": meta["filepath"],
                    "start_time": float(meta["start_time"]),
                    "end_time": float(meta["end_time"]),
                    "duration": float(meta["duration"]),
                    "year": meta.get("year"),
                    "perceptual_hash": phash,
                }
                
                if is_new_clip:
                    if phash:
                        # Best-effort dedup against cached pHashes.
                        is_dup = any(phash_distance(phash, h) <= max_dist for h in existing_phashes if h)
                        if is_dup:
                            continue

                    # Track pending clip (may add entity_stats later)
                    pending_clips[clip_id] = clip_data
                    existing_clip_ids.add(clip_id)  # Track in memory
                    if phash:
                        existing_phashes.add(phash)  # Track in memory

                # Embedding + visual tags
                emb_path = embeddings_dir / f"{clip_id}.npy"
                vis_path = visual_tags_dir / f"{clip_id}.json"
                if emb_path.exists() and clip_id not in existing_embeddings:
                    emb = np.load(emb_path).astype(np.float32).reshape(-1)
                    if emb.shape[0] == 512:
                        embeddings_to_insert.append({
                            "clip_id": clip_id,
                            "embedding": emb.tolist(),
                        })
                        existing_embeddings.add(clip_id)

                if vis_path.exists() and clip_id not in existing_tags:
                    vis = read_json(vis_path)
                    tags = vis.get("tags") or {}
                    for tag, score in tags.items():
                        try:
                            tags_to_insert.append({
                                "clip_id": clip_id,
                                "tag": str(tag),
                                "similarity_score": float(score),
                            })
                        except Exception:
                            continue
                    existing_tags.add(clip_id)

                # Signals (required silence_ratio)
                sig_path = signals_dir / f"{clip_id}.json"
                if clip_id not in existing_signals:
                    if sig_path.exists():
                        sig = read_json(sig_path)
                        silence_ratio = sig.get("silence_ratio")
                        if silence_ratio is None:
                            silence_ratio = 1.0
                        signals_to_insert.append({
                            "clip_id": clip_id,
                            "motion_score": sig.get("motion_score"),
                            "noise_level": sig.get("noise_level"),
                            "silence_ratio": float(silence_ratio),
                            "brightness_entropy": sig.get("brightness_entropy"),
                        })
                    else:
                        # Enforce not-null: insert a baseline signals row if absent.
                        signals_to_insert.append({
                            "clip_id": clip_id,
                            "silence_ratio": 1.0,
                        })
                    existing_signals.add(clip_id)

                # Entities/scenes (optional, replayable)
                ent_path = entities_dir / f"{clip_id}.json"
                if ent_path.exists():
                    ent = read_json(ent_path)
                    # Store full richness in clips.entity_stats JSONB
                    if is_new_clip and clip_id in pending_clips:
                        # New clip - add entity_stats to pending clip
                        pending_clips[clip_id]["entity_stats"] = ent
                    elif not is_new_clip:
                        # Existing clip - add to update batch
                        clips_to_update.append({
                            "id": clip_id,
                            "entity_stats": ent,
                        })

                    if overwrite_entities:
                        # Delete existing entities/scenes for this clip (will be done in batch)
                        if clip_id in existing_entities:
                            session.query(ClipEntity).filter(ClipEntity.clip_id == clip_id).delete()
                            existing_entities.discard(clip_id)
                        if clip_id in existing_scenes:
                            session.query(ClipScene).filter(ClipScene.clip_id == clip_id).delete()
                            existing_scenes.discard(clip_id)

                    if clip_id not in existing_entities:
                        objects = ent.get("objects") or {}
                        if isinstance(objects, dict):
                            for label, stats in objects.items():
                                if not isinstance(stats, dict):
                                    continue
                                try:
                                    conf = float(stats.get("max"))
                                except Exception:
                                    continue
                                if conf > 0:
                                    entities_to_insert.append({
                                        "clip_id": clip_id,
                                        "entity": str(label),
                                        "confidence": conf,
                                    })
                        existing_entities.add(clip_id)

                    if clip_id not in existing_scenes:
                        scenes = ent.get("scenes") or {}
                        if isinstance(scenes, dict):
                            for scene, score in scenes.items():
                                try:
                                    conf = float(score)
                                except Exception:
                                    continue
                                if conf > 0:
                                    scenes_to_insert.append({
                                        "clip_id": clip_id,
                                        "scene": str(scene),
                                        "confidence": conf,
                                    })
                        existing_scenes.add(clip_id)

                # Commit in batches for better performance
                # Move pending clips to insert list before checking batch size
                if pending_clips:
                    clips_to_insert.extend(pending_clips.values())
                    pending_clips.clear()
                
                total_pending = (
                    len(clips_to_insert) + len(embeddings_to_insert) + len(tags_to_insert) +
                    len(signals_to_insert) + len(entities_to_insert) + len(scenes_to_insert)
                )
                if total_pending >= BATCH_SIZE:
                    _commit_batch(
                        session,
                        clips_to_insert,
                        embeddings_to_insert,
                        tags_to_insert,
                        signals_to_insert,
                        entities_to_insert,
                        scenes_to_insert,
                        clips_to_update,
                    )
                    # Clear batches
                    clips_to_insert.clear()
                    embeddings_to_insert.clear()
                    tags_to_insert.clear()
                    signals_to_insert.clear()
                    entities_to_insert.clear()
                    scenes_to_insert.clear()
                    clips_to_update.clear()

            # Move any remaining pending clips
            if pending_clips:
                clips_to_insert.extend(pending_clips.values())

            # Commit remaining items
            if (
                clips_to_insert or embeddings_to_insert or tags_to_insert or
                signals_to_insert or entities_to_insert or scenes_to_insert or clips_to_update
            ):
                _commit_batch(
                    session,
                    clips_to_insert,
                    embeddings_to_insert,
                    tags_to_insert,
                    signals_to_insert,
                    entities_to_insert,
                    scenes_to_insert,
                    clips_to_update,
                )


def _commit_batch(
    session,
    clips_to_insert: list[dict],
    embeddings_to_insert: list[dict],
    tags_to_insert: list[dict],
    signals_to_insert: list[dict],
    entities_to_insert: list[dict],
    scenes_to_insert: list[dict],
    clips_to_update: list[dict],
) -> None:
    """Commit a batch of inserts using bulk operations."""
    if clips_to_insert:
        session.bulk_insert_mappings(Clip, clips_to_insert)
    if embeddings_to_insert:
        session.bulk_insert_mappings(ClipEmbedding, embeddings_to_insert)
    if tags_to_insert:
        session.bulk_insert_mappings(ClipTag, tags_to_insert)
    if signals_to_insert:
        session.bulk_insert_mappings(ClipSignals, signals_to_insert)
    if entities_to_insert:
        session.bulk_insert_mappings(ClipEntity, entities_to_insert)
    if scenes_to_insert:
        session.bulk_insert_mappings(ClipScene, scenes_to_insert)
    if clips_to_update:
        # For entity_stats updates, need to update individually
        for update_data in clips_to_update:
            clip_id = update_data.pop("id")
            session.query(Clip).filter(Clip.id == clip_id).update(update_data)
    session.commit()


if __name__ == "__main__":
    app()


