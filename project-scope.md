# Project: Automated Found-Footage Scraper & Aesthetic Indexer

## Goal

Build a local system that:

1. Scrapes public-domain + online video sources
2. Splits videos into short clips via shot detection
3. Auto-tags clips using vision + audio models
4. Stores clips + embeddings in a searchable database
5. Enables aesthetic-based retrieval for montage editing

---

## Tech Stack

### Core

* **Python 3.12**
* **FFmpeg** (system dependency)
* **PostgreSQL 15**
* **pgvector**

### Python Libraries

```
yt-dlp
internetarchive
PySceneDetect
opencv-python
torch
open-clip-torch
numpy
scipy
librosa
pydub
sqlalchemy
psycopg2-binary
pgvector
tqdm
rich
```

Optional (later):

```
fastapi
streamlit
gradio
```

---

## Repo Structure (authoritative)

```
aesthetic-indexer/
├── README.md
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw_videos/
│   │   ├── archive/
│   │   └── youtube/
│   ├── shots/
│   ├── clips/
│   └── frames/
├── prompts/
│   └── aesthetic_tags.txt
├── scripts/
│   ├── scrape_archive.py
│   ├── scrape_youtube.py
│   ├── detect_shots.py
│   ├── extract_clips.py
│   ├── extract_frames.py
│   ├── tag_visual.py
│   ├── tag_audio.py
│   └── ingest_db.py
├── src/
│   ├── db/
│   │   ├── models.py
│   │   └── session.py
│   ├── vision/
│   │   └── clip_tagger.py
│   ├── audio/
│   │   └── audio_tagger.py
│   ├── pipeline/
│   │   └── run_all.py
│   └── utils/
│       └── ffmpeg.py
└── migrations/
```

---

## Pipeline Definition (must follow this order)

### 1. Scraping

#### `scripts/scrape_archive.py`

* Query Internet Archive collections:

  * `us_military`
  * `prelinger`
  * `religion`
* Download **original video files only**
* Store metadata JSON alongside video

#### `scripts/scrape_youtube.py`

* Use `yt-dlp`
* Input: keyword list
* Output: full-length videos + metadata

---

### 2. Shot Detection

#### `scripts/detect_shots.py`

* Use **PySceneDetect**
* Content-aware detector
* Output:

```
data/shots/{video_id}.csv
(start_time, end_time)
```

---

### 3. Clip Extraction

#### `scripts/extract_clips.py`

Rules:

* Clip length: **1–7 seconds**
* Discard:

  * static slides
  * excessive talking-head shots (optional heuristic)

FFmpeg:

```
ffmpeg -ss START -to END -i input.mp4 -c copy clip.mp4
```

---

### 4. Frame Sampling

#### `scripts/extract_frames.py`

* Extract 3–5 evenly spaced frames per clip
* Store as JPG
* Used for CLIP inference

---

### 5. Visual Tagging (CLIP)

#### `src/vision/clip_tagger.py`

Model:

* `ViT-B-32` via `open-clip`

Process:

* Encode frames
* Average embeddings
* Compare against prompt bank

Prompt bank:

```
prompts/aesthetic_tags.txt
```

Output:

```
clip_id | tag | similarity_score
```

---

### 6. Audio Tagging

#### `src/audio/audio_tagger.py`

Signals:

* RMS energy
* Silence ratio
* Spectral centroid
* Optional pretrained classifier (YAMNet)

Tags:

* silence
* gunfire
* chanting
* crowd
* mechanical

---

### 7. Database Ingest

#### Tables

```sql
clips (
  id UUID PK,
  source TEXT,
  filepath TEXT,
  start_time FLOAT,
  end_time FLOAT,
  year INT,
  duration FLOAT
)

clip_embeddings (
  clip_id UUID,
  embedding VECTOR(512)
)

clip_tags (
  clip_id UUID,
  tag TEXT,
  score FLOAT
)

clip_signals (
  clip_id UUID,
  motion FLOAT,
  noise FLOAT,
  silence_ratio FLOAT
)
```

---

## Query Examples (this is the payoff)

```sql
SELECT c.filepath
FROM clips c
JOIN clip_tags t ON c.id = t.clip_id
WHERE t.tag = 'night vision'
AND t.score > 0.35
ORDER BY RANDOM()
LIMIT 30;
```

Or via pgvector similarity search:

```
religious + authority + low motion
```

---

## Non-Goals (important)

* No automatic editing
* No copyright enforcement
* No UI required in v1

This is an **indexing engine**, not a creative director.

---

## Milestones

1. Archive scraper + shot detection working
2. Clip generation stable
3. CLIP tagging validated
4. DB ingest + basic retrieval
5. (Optional) FastAPI search endpoint

---

## Final note (hard truth)

This system becomes powerful only after:

* **Thousands of clips**
* A **carefully tuned prompt bank**
* Resisting the urge to over-label

Sparse, ambiguous tags win.