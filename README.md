# Aesthetic Indexer

Automated found-footage pipeline: scrape videos → detect shots → extract clips → tag with CLIP → compute signals → ingest into PostgreSQL + pgvector for semantic search.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  • Internet Archive (collections/keywords)                              │
│  • YouTube (keyword search)                                             │
│  • Local folder (recursive video discovery)                             │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. SCRAPE VIDEOS                                                        │
│    • yt-dlp, internetarchive                                            │
│    → data/raw_videos/{source}/{video_id}/                               │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. DETECT SHOTS                                                         │
│    • PySceneDetect (ContentDetector)                                    │
│    → data/shots/{source}__{video_id}.csv                                │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. EXTRACT CLIPS                                                        │
│    • FFmpeg (1-7s clips, deterministic policy)                          │
│    • pHash deduplication (first frame)                                  │
│    → data/clips/{source}/{video_id}/{clip_id}.mp4                       │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. EXTRACT FRAMES                                                       │
│    • FFmpeg (3-5 evenly spaced frames per clip)                         │
│    → data/frames/{clip_id}/*.jpg                                        │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4.5 EXTRACT ENTITIES + SCENES                                            │
│    • YOLOv8 (objects: person, car, etc.)                                 │
│    • Places365 (scenes: best-effort, optional weights)                   │
│    → data/entities/{clip_id}.json                                        │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. TAG VISUAL                                                           │
│    • open-clip-torch (ViT-B-32/OpenAI)                                  │
│    • Average frame embeddings → 512-dim vector                          │
│    • Cosine similarity vs prompt bank                                   │
│    → data/embeddings/{clip_id}.npy                                      │
│    → data/tags/visual/{clip_id}.json                                    │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. TAG AUDIO                                                            │
│    • librosa (RMS, spectral centroid, silence detection)                │
│    → data/tags/audio/{clip_id}.json                                     │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. COMPUTE SIGNALS                                                      │
│    • OpenCV (optical flow motion, brightness entropy)                   │
│    • librosa (noise level, silence ratio)                               │
│    → data/signals/{clip_id}.json                                        │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 8. INGEST DATABASE                                                      │
│    • PostgreSQL 15 + pgvector                                           │
│    • SQLAlchemy ORM                                                     │
│    → clips, clip_embeddings, clip_tags, clip_signals, clip_entities, clip_scenes │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Libraries

**Video Processing:**
- `scenedetect` - Shot boundary detection
- `opencv-python` - Motion analysis, brightness entropy
- `ffmpeg` (system) - Video/audio extraction

**ML/AI:**
- `open-clip-torch` - CLIP embeddings (ViT-B-32/OpenAI)
- `torch` - Deep learning backend
- `torchvision` - Places365 backbone (ResNet18)
- `ultralytics` - YOLOv8 object detection
- `librosa` - Audio analysis (RMS, spectral features, silence)

**Database:**
- `PostgreSQL 15` + `pgvector` - Vector similarity search
- `SQLAlchemy` - ORM
- `psycopg2-binary` - PostgreSQL driver

**Utilities:**
- `yt-dlp` - YouTube downloads
- `internetarchive` - Internet Archive API
- `ImageHash` - Perceptual hashing (deduplication)
- `FastAPI` + `uvicorn` - Optional API server

## Requirements

- Python **3.12**
- **FFmpeg** on PATH (`ffmpeg`, `ffprobe`)
- PostgreSQL **15** with `pgvector` extension

## Setup

```bash
# Create virtual environment
cd aesthetic-indexer
python -m venv .venv
. .venv/bin/activate  # Windows: .\\.venv\\Scripts\\activate

# Install package
pip install -U pip
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with DATABASE_URL, etc.

# Initialize database
psql "$DATABASE_URL" -f migrations/001_initial_schema.sql
psql "$DATABASE_URL" -f migrations/002_add_entities.sql
```

## Usage

The pipeline is **resumable** — completed stages are skipped automatically.

### Quick Start: Full Pipeline

**Internet Archive (default collections):**
```bash
python -m src.pipeline.run_all run-all
```

**Internet Archive (keyword search):**
```bash
python -m src.pipeline.run_all run-all --archive-keywords-file keywords.txt
```

**Local folder:**
```bash
python -m src.pipeline.run_all run-all --mode local --local-input-dir "D:/footage_dump"
```

### Stage-by-Stage

**1. Scrape videos:**
```bash
# Internet Archive (default collections)
python -m src.pipeline.run_all scrape-archive --archive-limit 50

# Internet Archive (keywords)
python -m src.pipeline.run_all scrape-archive-keywords --keywords-file keywords.txt --archive-limit 50

# YouTube (requires keywords file)
python -m src.pipeline.run_all scrape-youtube --keywords-file keywords.txt
```

**2. Detect shots:**
```bash
python -m src.pipeline.run_all detect-shots
```

**3. Extract clips (1-7s, deterministic):**
```bash
python -m src.pipeline.run_all extract-clips
```

**4. Extract frames (3-5 per clip):**
```bash
python -m src.pipeline.run_all extract-frames
```

**4.5 Extract entities + scenes (YOLOv8 + Places365):**
```bash
python -m src.pipeline.run_all extract-entities
```

**5. Tag visual (CLIP embeddings):**
```bash
python -m src.pipeline.run_all tag-visual
```

**6. Tag audio:**
```bash
python -m src.pipeline.run_all tag-audio
```

**7. Compute signals (motion, noise, silence, brightness):**
```bash
python -m src.pipeline.run_all compute-signals
```

**8. Ingest to database:**
```bash
python -m src.pipeline.run_all ingest-db
```

### Configuration

Environment variables (optional):
- `CLIP_MIN_SECONDS` - Minimum clip duration (default: 1.0)
- `CLIP_MAX_SECONDS` - Maximum clip duration (default: 7.0)
- `CLIP_WINDOW_OVERLAP` - Overlap between windows (default: 0.25)
- `CLIP_MAX_PER_VIDEO` - Max clips per video (default: 500)
- `PHASH_MAX_DISTANCE` - pHash Hamming distance threshold (default: 4)
- `PLACES365_WEIGHTS` - Path to Places365 weights (if set, `extract-entities` will populate `clip_scenes`)

## Database Queries

### Tag-based filtering
```sql
-- Random batch of clips matching a tag
SELECT c.filepath, t.similarity_score
FROM clips c
JOIN clip_tags t ON c.id = t.clip_id
WHERE t.tag = 'night vision'
  AND t.similarity_score > 0.35
ORDER BY RANDOM()
LIMIT 30;
```

### Entity-based filtering (YOLO / Places)
```sql
-- Clips where YOLO saw a person with high confidence
SELECT c.filepath, e.confidence
FROM clips c
JOIN clip_entities e ON c.id = e.clip_id
WHERE e.entity = 'person'
  AND e.confidence > 0.80
ORDER BY e.confidence DESC
LIMIT 50;
```

```sql
-- Best-effort Places365 scene match (requires Places weights configured during extract-entities)
SELECT c.filepath, s.scene, s.confidence
FROM clips c
JOIN clip_scenes s ON c.id = s.clip_id
WHERE s.scene ILIKE '%airport%'
ORDER BY s.confidence DESC
LIMIT 50;
```

### Vector similarity search
```sql
-- Nearest neighbors (requires query embedding)
SELECT c.filepath, (e.embedding <-> :query_embedding) AS distance
FROM clip_embeddings e
JOIN clips c ON c.id = e.clip_id
ORDER BY e.embedding <-> :query_embedding
LIMIT 50;
```

### Signal-based queries
```sql
-- High motion, low silence
SELECT c.filepath, s.motion_score, s.silence_ratio
FROM clips c
JOIN clip_signals s ON c.id = s.clip_id
WHERE s.motion_score > 5.0
  AND s.silence_ratio < 0.2
ORDER BY s.motion_score DESC
LIMIT 20;
```

## Notes

- **Clip extraction is deterministic** — same video produces identical clips (see `CLIP_EXTRACTION_POLICY.md`)
- **Raw similarity scores** — `clip_tags.similarity_score` stores continuous CLIP cosine similarity (no thresholds)
- **Silence ratio is first-class** — always present in `clip_signals.silence_ratio` (NOT NULL)
- **Deduplication** — pHash on first frame with configurable Hamming distance threshold
- **Resumable** — pipeline tracks completion markers in `data/.pipeline/` to skip finished stages
