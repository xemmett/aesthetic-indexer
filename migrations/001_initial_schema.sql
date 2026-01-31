-- 001_initial_schema.sql
-- PostgreSQL 15 + pgvector schema for the Aesthetic Indexer.

BEGIN;

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Core clips table
CREATE TABLE IF NOT EXISTS clips (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source TEXT NOT NULL,                  -- e.g. 'archive' | 'youtube'
  video_id TEXT NOT NULL,                -- source-specific id
  filepath TEXT NOT NULL,                -- relative/absolute path to clip file
  start_time DOUBLE PRECISION NOT NULL,
  end_time DOUBLE PRECISION NOT NULL,
  duration DOUBLE PRECISION NOT NULL,
  year INTEGER NULL,

  -- Deduplication primitive (pHash of first frame as hex string)
  perceptual_hash TEXT NULL,

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT clips_time_valid CHECK (end_time > start_time),
  CONSTRAINT clips_duration_valid CHECK (duration > 0)
);

-- Avoid exact duplicates from re-extraction (deterministic)
CREATE UNIQUE INDEX IF NOT EXISTS clips_unique_source_segment
  ON clips (source, video_id, start_time, end_time);

CREATE INDEX IF NOT EXISTS clips_source_video_idx
  ON clips (source, video_id);

CREATE INDEX IF NOT EXISTS clips_phash_idx
  ON clips (perceptual_hash);

-- Embeddings (ViT-B-32 => 512 dims)
CREATE TABLE IF NOT EXISTS clip_embeddings (
  clip_id UUID PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
  embedding vector(512) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Helpful for ANN later; for now we keep a basic ivfflat index optional.
-- NOTE: requires ANALYZE and appropriate lists value once dataset is larger.
-- CREATE INDEX clip_embeddings_ivfflat_idx
--   ON clip_embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

-- Raw CLIP similarity scores (no booleans)
CREATE TABLE IF NOT EXISTS clip_tags (
  clip_id UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  tag TEXT NOT NULL,
  similarity_score DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (clip_id, tag)
);

CREATE INDEX IF NOT EXISTS clip_tags_tag_idx
  ON clip_tags (tag);

-- Query primitives (signals). silence_ratio is first-class and NOT NULL.
CREATE TABLE IF NOT EXISTS clip_signals (
  clip_id UUID PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
  motion_score DOUBLE PRECISION NULL,
  noise_level DOUBLE PRECISION NULL,
  silence_ratio DOUBLE PRECISION NOT NULL,
  brightness_entropy DOUBLE PRECISION NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMIT;


