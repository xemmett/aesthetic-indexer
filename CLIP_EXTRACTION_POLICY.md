# CLIP_EXTRACTION_POLICY (Deterministic)

This pipeline must produce the **same clips for the same inputs** across runs (no randomness).

## Inputs

- `data/shots/{video_id}.csv` containing `(start_time, end_time)` per detected shot.

## Parameters

- **MIN_SECONDS**: 1.0
- **MAX_SECONDS**: 7.0
- **WINDOW_OVERLAP**: 0.25 (25%)
- **WINDOW_STEP**: \(MAX_SECONDS * (1 - WINDOW_OVERLAP)\) → default 5.25s
- **MAX_CLIPS_PER_VIDEO**: 500

## Rules (in order)

1. **Discard short shots**
   - If `(end - start) < MIN_SECONDS`: discard.

2. **Shots within range**
   - If `MIN_SECONDS <= (end - start) <= MAX_SECONDS`: emit exactly 1 clip `[start, end]`.

3. **Shots longer than MAX_SECONDS (sliding windows)**
   - Emit windows of length `MAX_SECONDS` starting at `start`, incrementing by `WINDOW_STEP` until `window_end <= end`.
   - If the remaining tail `(end - last_window_end) >= MIN_SECONDS`, emit one final clip `[end - MAX_SECONDS, end]` (clamped to shot start).
   - **No random sampling.** Windows are computed deterministically.

4. **Clip count cap**
   - If more than `MAX_CLIPS_PER_VIDEO` clips would be emitted, truncate deterministically by:
     - taking the earliest clips in time order.

5. **Deduplication**
   - Compute pHash of the first extracted frame of the clip.
   - Reject if within `PHASH_MAX_DISTANCE` of an existing clip pHash (configurable).

## Naming / IDs

- `clip_id` is a UUID.
- Clips are written to `data/clips/{source}/{video_id}/{clip_id}.mp4`.
- A sidecar JSON is written alongside each clip at `{clip_id}.json` with:
  - `clip_id`, `video_id`, `source`, `start_time`, `end_time`, `duration`, `year` (if available), `perceptual_hash`.


