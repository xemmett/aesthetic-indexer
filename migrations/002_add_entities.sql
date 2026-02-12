-- 002_add_entities.sql
-- Add entity/scene extraction storage (YOLOv8 + Places365).

BEGIN;

-- Rich aggregated stats (replayable without schema churn)
ALTER TABLE clips
  ADD COLUMN IF NOT EXISTS entity_stats JSONB;

-- Helpful for JSONB containment/path queries
CREATE INDEX IF NOT EXISTS clips_entity_stats_gin_idx
  ON clips USING GIN (entity_stats jsonb_path_ops);

-- Materialized object detections for fast filtering
CREATE TABLE IF NOT EXISTS clip_entities (
  clip_id UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  entity TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (clip_id, entity),
  CONSTRAINT clip_entities_confidence_valid CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE INDEX IF NOT EXISTS clip_entities_entity_confidence_idx
  ON clip_entities (entity, confidence);

CREATE INDEX IF NOT EXISTS clip_entities_clip_id_idx
  ON clip_entities (clip_id);

-- Materialized scene classifications for fast filtering
CREATE TABLE IF NOT EXISTS clip_scenes (
  clip_id UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  scene TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (clip_id, scene),
  CONSTRAINT clip_scenes_confidence_valid CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE INDEX IF NOT EXISTS clip_scenes_scene_confidence_idx
  ON clip_scenes (scene, confidence);

CREATE INDEX IF NOT EXISTS clip_scenes_clip_id_idx
  ON clip_scenes (clip_id);

COMMIT;




