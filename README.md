# Aesthetic Indexer (Found-Footage Pipeline)

Scrape videos → detect shots → extract short clips → sample frames → tag (CLIP + audio) → compute query signals → ingest into PostgreSQL + pgvector.

## Requirements

- Python **3.12**
- **FFmpeg** available on PATH (`ffmpeg`, `ffprobe`)
- PostgreSQL **15** with `pgvector`

## Setup

Create and activate a virtualenv, then install:

```bash
cd aesthetic-indexer
python -m venv .venv
. .venv/bin/activate  # Windows: .\\.venv\\Scripts\\activate
pip install -U pip
pip install -e .
```

Create your env file:

```bash
cp .env.example .env
```

## Database

Create the database and run the migration:

```bash
psql "$DATABASE_URL" -f migrations/001_initial_schema.sql
```

## Running the pipeline (resumable)

The orchestrator skips completed outputs and only processes new/unprocessed items by default.

```bash
python -m src.pipeline.run_all --help
python -m src.pipeline.run_all scrape-archive
python -m src.pipeline.run_all scrape-archive-keywords --keywords-file keywords.txt
python -m src.pipeline.run_all detect-shots
python -m src.pipeline.run_all extract-clips
python -m src.pipeline.run_all extract-frames
python -m src.pipeline.run_all tag-visual
python -m src.pipeline.run_all tag-audio
python -m src.pipeline.run_all compute-signals
python -m src.pipeline.run_all ingest-db
```

Or run everything:

```bash
python -m src.pipeline.run_all run-all
python -m src.pipeline.run_all run-all --archive-keywords-file keywords.txt
```

### Internet Archive keyword search (instead of collections)

Direct script usage:

```bash
python scripts/scrape_archive.py keywords -k "night vision" -k "religious ritual" --limit 25
python scripts/scrape_archive.py keywords --keywords-file keywords.txt --limit 50
```

Keyword-search downloads are stored under:

- `data/raw_videos/archive/keyword_search/{identifier}/...`

## pgvector retrieval (examples)

### 1) Tag filtering (random montage batch)

```sql
SELECT c.filepath
FROM clips c
JOIN clip_tags t ON c.id = t.clip_id
WHERE t.tag = 'night vision'
  AND t.similarity_score > 0.35
ORDER BY RANDOM()
LIMIT 30;
```

### 2) Similarity search (embedding nearest neighbors)

Assume you have a query embedding (a `vector(512)`) from a prompt like `"religious authority low motion"`.

```sql
-- query_embedding is a vector(512) parameter
SELECT c.filepath, (e.embedding <-> :query_embedding) AS distance
FROM clip_embeddings e
JOIN clips c ON c.id = e.clip_id
ORDER BY e.embedding <-> :query_embedding
LIMIT 50;
```

## Notes

- **Clip extraction is deterministic**. See `CLIP_EXTRACTION_POLICY.md`.
- **Raw CLIP similarity scores** are stored in `clip_tags.similarity_score` (no boolean labels).
- **Silence ratio is first-class** and **NOT NULL** in `clip_signals.silence_ratio`.
- **Deduplication** uses pHash on the clip’s first frame (configurable Hamming distance).


