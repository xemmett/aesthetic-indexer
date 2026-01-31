from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def project_root() -> Path:
    # When running from repo root: aesthetic-indexer/ is cwd; when imported, this is stable enough.
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    load_dotenv(override=False)
    p = os.getenv("DATA_DIR")
    if p:
        return Path(p).expanduser().resolve()
    return (project_root() / "data").resolve()


def prompts_file() -> Path:
    load_dotenv(override=False)
    p = os.getenv("PROMPTS_FILE")
    if p:
        return Path(p).expanduser().resolve()
    return (project_root() / "prompts" / "aesthetic_tags.txt").resolve()


