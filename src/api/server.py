from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from src.db.models import Clip, ClipEmbedding, ClipSignals, ClipTag
from src.db.session import get_session_factory

app = FastAPI(title="Aesthetic Indexer API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global CLIP model cache
_model_cache: Optional[tuple] = None


def get_clip_model():
    """Lazy load and cache CLIP model."""
    global _model_cache
    if _model_cache is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model = model.to(dev)
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model_cache = (model, tokenizer, dev)
    return _model_cache


def encode_text_query(query: str) -> np.ndarray:
    """Encode a text query into a CLIP embedding."""
    model, tokenizer, dev = get_clip_model()
    with torch.no_grad():
        text_tokens = tokenizer([query])
        text_tokens = text_tokens.to(dev)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features[0].detach().cpu().numpy().astype(np.float32)


# Pydantic models for API responses
class ClipResponse(BaseModel):
    id: str
    source: str
    video_id: str
    filepath: str
    start_time: float
    end_time: float
    duration: float
    year: Optional[int]
    created_at: str
    tags: list[dict[str, float]]
    signals: Optional[dict]
    similarity_score: Optional[float] = None

    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    clips: list[ClipResponse]
    total: int
    page: int
    page_size: int


@app.get("/")
async def root():
    return {"message": "Aesthetic Indexer API"}


@app.get("/api/clips", response_model=SearchResponse)
async def list_clips(
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
    source: Optional[str] = None,
    year: Optional[int] = None,
):
    """List clips with pagination and optional filters."""
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        query = select(Clip)
        
        if source:
            query = query.where(Clip.source == source)
        if year:
            query = query.where(Clip.year == year)
        
        # Get total count
        count_query = select(func.count()).select_from(Clip)
        if source:
            count_query = count_query.where(Clip.source == source)
        if year:
            count_query = count_query.where(Clip.year == year)
        total = session.scalar(count_query)
        
        # Pagination
        offset = (page - 1) * page_size
        query = query.order_by(Clip.created_at.desc()).offset(offset).limit(page_size)
        
        clips = session.scalars(query).all()
        
        # Load related data
        clip_ids = [c.id for c in clips]
        tags_query = select(ClipTag).where(ClipTag.clip_id.in_(clip_ids))
        tags_map = {}
        for tag in session.scalars(tags_query):
            if tag.clip_id not in tags_map:
                tags_map[tag.clip_id] = []
            tags_map[tag.clip_id].append({tag.tag: tag.similarity_score})
        
        signals_query = select(ClipSignals).where(ClipSignals.clip_id.in_(clip_ids))
        signals_map = {s.clip_id: s for s in session.scalars(signals_query)}
        
        clip_responses = []
        for clip in clips:
            clip_dict = {
                "id": str(clip.id),
                "source": clip.source,
                "video_id": clip.video_id,
                "filepath": clip.filepath,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "duration": clip.duration,
                "year": clip.year,
                "created_at": clip.created_at.isoformat(),
                "tags": tags_map.get(clip.id, []),
                "signals": {
                    "motion_score": signals_map[clip.id].motion_score if clip.id in signals_map else None,
                    "noise_level": signals_map[clip.id].noise_level if clip.id in signals_map else None,
                    "silence_ratio": signals_map[clip.id].silence_ratio if clip.id in signals_map else None,
                    "brightness_entropy": signals_map[clip.id].brightness_entropy if clip.id in signals_map else None,
                } if clip.id in signals_map else None,
            }
            clip_responses.append(ClipResponse(**clip_dict))
        
        return SearchResponse(clips=clip_responses, total=total, page=page, page_size=page_size)


@app.get("/api/clips/search/semantic", response_model=SearchResponse)
async def search_semantic(
    query: str = Query(..., description="Text query for semantic search"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
    limit: int = Query(100, ge=1, le=500, description="Max results to consider before pagination"),
):
    """Semantic search using CLIP embeddings."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Encode query
    query_embedding = encode_text_query(query)
    
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        # Use pgvector distance operator (<->) for cosine distance
        # Since embeddings are normalized, cosine distance = 1 - cosine similarity
        # Format embedding as PostgreSQL array string: '[0.1, 0.2, ...]'
        embedding_list = query_embedding.tolist()
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
        
        # Use string formatting for the vector (safe since we control the format)
        # and bindparam for the limit
        query_sql = text(f"""
            SELECT c.id, c.source, c.video_id, c.filepath, c.start_time, c.end_time, 
                   c.duration, c.year, c.created_at,
                   (1 - (e.embedding <=> '{embedding_str}'::vector)) AS similarity_score
            FROM clip_embeddings e
            JOIN clips c ON c.id = e.clip_id
            ORDER BY e.embedding <=> '{embedding_str}'::vector
            LIMIT :limit
        """)
        
        result = session.execute(
            query_sql,
            {"limit": limit}
        )
        
        all_clips = []
        for row in result:
            clip_dict = {
                "id": str(row.id),
                "source": row.source,
                "video_id": row.video_id,
                "filepath": row.filepath,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "duration": row.duration,
                "year": row.year,
                "created_at": row.created_at.isoformat(),
                "similarity_score": float(row.similarity_score),
            }
            all_clips.append(clip_dict)
        
        # Get total
        total = len(all_clips)
        
        # Pagination
        offset = (page - 1) * page_size
        paginated_clips = all_clips[offset : offset + page_size]
        
        # Load tags and signals for paginated results
        clip_ids = [c["id"] for c in paginated_clips]
        if clip_ids:
            tags_query = select(ClipTag).where(ClipTag.clip_id.in_([uuid.UUID(id) for id in clip_ids]))
            tags_map = {}
            for tag in session.scalars(tags_query):
                if str(tag.clip_id) not in tags_map:
                    tags_map[str(tag.clip_id)] = []
                tags_map[str(tag.clip_id)].append({tag.tag: tag.similarity_score})
            
            signals_query = select(ClipSignals).where(ClipSignals.clip_id.in_([uuid.UUID(id) for id in clip_ids]))
            signals_map = {str(s.clip_id): s for s in session.scalars(signals_query)}
            
            for clip_dict in paginated_clips:
                clip_id = clip_dict["id"]
                clip_dict["tags"] = tags_map.get(clip_id, [])
                signals = signals_map.get(clip_id)
                clip_dict["signals"] = {
                    "motion_score": signals.motion_score if signals else None,
                    "noise_level": signals.noise_level if signals else None,
                    "silence_ratio": signals.silence_ratio if signals else None,
                    "brightness_entropy": signals.brightness_entropy if signals else None,
                } if signals else None
        
        clip_responses = [ClipResponse(**c) for c in paginated_clips]
        return SearchResponse(clips=clip_responses, total=total, page=page, page_size=page_size)


@app.get("/api/clips/search/tags", response_model=SearchResponse)
async def search_by_tags(
    tag: str = Query(..., description="Tag to search for"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity score"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
):
    """Search clips by tag with minimum similarity score."""
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        query = (
            select(Clip, ClipTag.similarity_score)
            .join(ClipTag, Clip.id == ClipTag.clip_id)
            .where(ClipTag.tag == tag)
            .where(ClipTag.similarity_score >= min_score)
            .order_by(ClipTag.similarity_score.desc())
        )
        
        # Get total count
        count_query = (
            select(func.count())
            .select_from(Clip)
            .join(ClipTag, Clip.id == ClipTag.clip_id)
            .where(ClipTag.tag == tag)
            .where(ClipTag.similarity_score >= min_score)
        )
        total = session.scalar(count_query)
        
        # Pagination
        offset = (page - 1) * page_size
        results = session.execute(query.offset(offset).limit(page_size)).all()
        
        clip_responses = []
        clip_ids = []
        for clip, similarity_score in results:
            clip_ids.append(clip.id)
            clip_dict = {
                "id": str(clip.id),
                "source": clip.source,
                "video_id": clip.video_id,
                "filepath": clip.filepath,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "duration": clip.duration,
                "year": clip.year,
                "created_at": clip.created_at.isoformat(),
                "similarity_score": float(similarity_score),
            }
            clip_responses.append(clip_dict)
        
        # Load tags and signals
        if clip_ids:
            tags_query = select(ClipTag).where(ClipTag.clip_id.in_(clip_ids))
            tags_map = {}
            for tag_obj in session.scalars(tags_query):
                if str(tag_obj.clip_id) not in tags_map:
                    tags_map[str(tag_obj.clip_id)] = []
                tags_map[str(tag_obj.clip_id)].append({tag_obj.tag: tag_obj.similarity_score})
            
            signals_query = select(ClipSignals).where(ClipSignals.clip_id.in_(clip_ids))
            signals_map = {str(s.clip_id): s for s in session.scalars(signals_query)}
            
            for clip_dict in clip_responses:
                clip_id = clip_dict["id"]
                clip_dict["tags"] = tags_map.get(clip_id, [])
                signals = signals_map.get(clip_id)
                clip_dict["signals"] = {
                    "motion_score": signals.motion_score if signals else None,
                    "noise_level": signals.noise_level if signals else None,
                    "silence_ratio": signals.silence_ratio if signals else None,
                    "brightness_entropy": signals.brightness_entropy if signals else None,
                } if signals else None
        
        clip_responses_models = [ClipResponse(**c) for c in clip_responses]
        return SearchResponse(clips=clip_responses_models, total=total, page=page, page_size=page_size)


@app.get("/api/clips/search/signals", response_model=SearchResponse)
async def search_by_signals(
    min_motion: Optional[float] = Query(None, description="Minimum motion score"),
    max_motion: Optional[float] = Query(None, description="Maximum motion score"),
    max_silence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Maximum silence ratio"),
    min_brightness: Optional[float] = Query(None, description="Minimum brightness entropy"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
):
    """Search clips by signal filters."""
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        query = select(Clip).join(ClipSignals, Clip.id == ClipSignals.clip_id)
        
        if min_motion is not None:
            query = query.where(ClipSignals.motion_score >= min_motion)
        if max_motion is not None:
            query = query.where(ClipSignals.motion_score <= max_motion)
        if max_silence is not None:
            query = query.where(ClipSignals.silence_ratio <= max_silence)
        if min_brightness is not None:
            query = query.where(ClipSignals.brightness_entropy >= min_brightness)
        
        # Get total count
        count_query = select(func.count()).select_from(Clip).join(ClipSignals, Clip.id == ClipSignals.clip_id)
        if min_motion is not None:
            count_query = count_query.where(ClipSignals.motion_score >= min_motion)
        if max_motion is not None:
            count_query = count_query.where(ClipSignals.motion_score <= max_motion)
        if max_silence is not None:
            count_query = count_query.where(ClipSignals.silence_ratio <= max_silence)
        if min_brightness is not None:
            count_query = count_query.where(ClipSignals.brightness_entropy >= min_brightness)
        total = session.scalar(count_query)
        
        # Pagination
        offset = (page - 1) * page_size
        query = query.order_by(Clip.created_at.desc()).offset(offset).limit(page_size)
        clips = session.scalars(query).all()
        
        # Load related data
        clip_ids = [c.id for c in clips]
        tags_query = select(ClipTag).where(ClipTag.clip_id.in_(clip_ids))
        tags_map = {}
        for tag in session.scalars(tags_query):
            if str(tag.clip_id) not in tags_map:
                tags_map[str(tag.clip_id)] = []
            tags_map[str(tag.clip_id)].append({tag.tag: tag.similarity_score})
        
        signals_query = select(ClipSignals).where(ClipSignals.clip_id.in_(clip_ids))
        signals_map = {str(s.clip_id): s for s in session.scalars(signals_query)}
        
        clip_responses = []
        for clip in clips:
            signals = signals_map.get(str(clip.id))
            clip_dict = {
                "id": str(clip.id),
                "source": clip.source,
                "video_id": clip.video_id,
                "filepath": clip.filepath,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "duration": clip.duration,
                "year": clip.year,
                "created_at": clip.created_at.isoformat(),
                "tags": tags_map.get(str(clip.id), []),
                "signals": {
                    "motion_score": signals.motion_score if signals else None,
                    "noise_level": signals.noise_level if signals else None,
                    "silence_ratio": signals.silence_ratio if signals else None,
                    "brightness_entropy": signals.brightness_entropy if signals else None,
                } if signals else None,
            }
            clip_responses.append(ClipResponse(**clip_dict))
        
        return SearchResponse(clips=clip_responses, total=total, page=page, page_size=page_size)


@app.get("/api/tags")
async def list_tags(
    limit: int = Query(100, ge=1, le=1000),
):
    """List all available tags with their usage counts."""
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        query = (
            select(ClipTag.tag, func.count(ClipTag.clip_id).label("count"))
            .group_by(ClipTag.tag)
            .order_by(func.count(ClipTag.clip_id).desc())
            .limit(limit)
        )
        results = session.execute(query).all()
        return [{"tag": tag, "count": count} for tag, count in results]


@app.get("/api/clips/{clip_id}")
async def get_clip(clip_id: str):
    """Get a single clip by ID."""
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        try:
            clip_uuid = uuid.UUID(clip_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid clip ID")
        
        clip = session.get(Clip, clip_uuid)
        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")
        
        # Load tags
        tags_query = select(ClipTag).where(ClipTag.clip_id == clip_uuid)
        tags = [{tag.tag: tag.similarity_score} for tag in session.scalars(tags_query)]
        
        # Load signals
        signals = session.get(ClipSignals, clip_uuid)
        
        clip_dict = {
            "id": str(clip.id),
            "source": clip.source,
            "video_id": clip.video_id,
            "filepath": clip.filepath,
            "start_time": clip.start_time,
            "end_time": clip.end_time,
            "duration": clip.duration,
            "year": clip.year,
            "created_at": clip.created_at.isoformat(),
            "tags": tags,
            "signals": {
                "motion_score": signals.motion_score if signals else None,
                "noise_level": signals.noise_level if signals else None,
                "silence_ratio": signals.silence_ratio if signals else None,
                "brightness_entropy": signals.brightness_entropy if signals else None,
            } if signals else None,
        }
        
        return ClipResponse(**clip_dict)


@app.get("/api/clips/{clip_id}/video")
async def serve_video(clip_id: str):
    """Serve video file for a clip."""
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        try:
            clip_uuid = uuid.UUID(clip_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid clip ID")
        
        clip = session.get(Clip, clip_uuid)
        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")
        
        # Resolve filepath - could be relative or absolute
        filepath = Path(clip.filepath)
        if not filepath.is_absolute():
            # Try relative to data directory
            data_dir = Path(os.getenv("DATA_DIR", "data"))
            filepath = data_dir / "clips" / filepath
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        return FileResponse(
            str(filepath),
            media_type="video/mp4",
            filename=filepath.name,
        )


@app.get("/api/clips/{clip_id}/thumbnail")
async def serve_thumbnail(clip_id: str):
    """Serve thumbnail (first frame) for a clip."""
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        try:
            clip_uuid = uuid.UUID(clip_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid clip ID")
        
        clip = session.get(Clip, clip_uuid)
        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")
        
        # Find thumbnail in frames directory
        data_dir = Path(os.getenv("DATA_DIR", "data"))
        frames_dir = data_dir / "frames" / clip_id
        thumbnail_path = frames_dir / "0001.jpg"
        
        # Fallback: try other frame numbers if 0001 doesn't exist
        if not thumbnail_path.exists():
            for i in range(1, 10):
                candidate = frames_dir / f"{i:04d}.jpg"
                if candidate.exists():
                    thumbnail_path = candidate
                    break
        
        if not thumbnail_path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        
        return FileResponse(
            str(thumbnail_path),
            media_type="image/jpeg",
            filename=thumbnail_path.name,
        )

