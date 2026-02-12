from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    source: Mapped[str] = mapped_column(Text, nullable=False)  # 'archive' | 'youtube'
    video_id: Mapped[str] = mapped_column(Text, nullable=False)
    filepath: Mapped[str] = mapped_column(Text, nullable=False)

    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    perceptual_hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Rich entity/scene stats (JSONB). Materialized fast-filters live in clip_entities/clip_scenes.
    entity_stats: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    embedding: Mapped["ClipEmbedding"] = relationship(back_populates="clip", cascade="all, delete-orphan", uselist=False)
    signals: Mapped["ClipSignals"] = relationship(back_populates="clip", cascade="all, delete-orphan", uselist=False)
    tags: Mapped[list["ClipTag"]] = relationship(back_populates="clip", cascade="all, delete-orphan")
    entities: Mapped[list["ClipEntity"]] = relationship(back_populates="clip", cascade="all, delete-orphan")
    scenes: Mapped[list["ClipScene"]] = relationship(back_populates="clip", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("source", "video_id", "start_time", "end_time", name="clips_unique_source_segment"),
    )


class ClipEmbedding(Base):
    __tablename__ = "clip_embeddings"

    clip_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clips.id", ondelete="CASCADE"), primary_key=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    clip: Mapped[Clip] = relationship(back_populates="embedding")


class ClipTag(Base):
    __tablename__ = "clip_tags"

    clip_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("clips.id", ondelete="CASCADE"), primary_key=True)
    tag: Mapped[str] = mapped_column(Text, primary_key=True)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    clip: Mapped[Clip] = relationship(back_populates="tags")


class ClipSignals(Base):
    __tablename__ = "clip_signals"

    clip_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clips.id", ondelete="CASCADE"), primary_key=True
    )

    motion_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    noise_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    silence_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    brightness_entropy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    clip: Mapped[Clip] = relationship(back_populates="signals")


class ClipEntity(Base):
    __tablename__ = "clip_entities"

    clip_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clips.id", ondelete="CASCADE"), primary_key=True
    )
    entity: Mapped[str] = mapped_column(Text, primary_key=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    clip: Mapped[Clip] = relationship(back_populates="entities")


class ClipScene(Base):
    __tablename__ = "clip_scenes"

    clip_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clips.id", ondelete="CASCADE"), primary_key=True
    )
    scene: Mapped[str] = mapped_column(Text, primary_key=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    clip: Mapped[Clip] = relationship(back_populates="scenes")

