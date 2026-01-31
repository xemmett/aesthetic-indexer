# API Server Setup

This document describes how to set up and run the FastAPI server for the aesthetic-indexer.

## Installation

Make sure you have the aesthetic-indexer package installed with all dependencies:

```bash
cd aesthetic-indexer
pip install -e .
```

This will install FastAPI and uvicorn along with all other dependencies.

## Configuration

The server uses the same database configuration as the rest of the aesthetic-indexer. Make sure your `.env` file contains:

```
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
DATA_DIR=./data  # Optional, defaults to ./data
```

## Running the Server

### Development Mode

```bash
python -m src.api.run_server
```

This will start the server on `http://localhost:8000` with auto-reload enabled.

### Production Mode

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Or with more workers:

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Clips

- `GET /api/clips` - List clips with pagination
  - Query params: `page`, `page_size`, `source`, `year`
- `GET /api/clips/{clip_id}` - Get a single clip
- `GET /api/clips/{clip_id}/video` - Serve video file

### Search

- `GET /api/clips/search/semantic` - Semantic search using CLIP embeddings
  - Query params: `query` (required), `page`, `page_size`, `limit`
- `GET /api/clips/search/tags` - Search by tag
  - Query params: `tag` (required), `min_score`, `page`, `page_size`
- `GET /api/clips/search/signals` - Filter by signals
  - Query params: `min_motion`, `max_motion`, `max_silence`, `min_brightness`, `page`, `page_size`

### Tags

- `GET /api/tags` - List all available tags
  - Query params: `limit`

## CORS

The server is configured to allow requests from `http://localhost:3000` (Next.js dev server). For production, update the CORS origins in `src/api/server.py`.

## Notes

- The CLIP model is loaded lazily on first semantic search request
- Video files are served directly from the file system
- The server uses the same database session management as the rest of the pipeline

