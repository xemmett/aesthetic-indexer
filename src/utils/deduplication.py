from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imagehash
from PIL import Image


@dataclass(frozen=True)
class PHashResult:
    hex_hash: str


def compute_phash(image_path: Path) -> PHashResult:
    with Image.open(image_path) as im:
        ph = imagehash.phash(im)
    return PHashResult(hex_hash=str(ph))


def phash_distance(hex_a: str, hex_b: str) -> int:
    a = imagehash.hex_to_hash(hex_a)
    b = imagehash.hex_to_hash(hex_b)
    return int(a - b)


def is_near_duplicate(
    *,
    candidate_hex: str,
    existing_hex: Optional[str],
    max_distance: int,
) -> bool:
    if not existing_hex:
        return False
    return phash_distance(candidate_hex, existing_hex) <= max_distance


