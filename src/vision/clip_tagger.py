from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import open_clip
import torch
from PIL import Image


@dataclass(frozen=True)
class VisualTagResult:
    embedding: np.ndarray  # (512,)
    tags: dict[str, float]  # tag -> cosine similarity


def load_prompt_bank(prompts_file: Path) -> list[str]:
    lines = prompts_file.read_text(encoding="utf-8").splitlines()
    prompts: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        prompts.append(s)
    if not prompts:
        raise ValueError(f"Prompt bank is empty: {prompts_file}")
    return prompts


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPTagger:
    """
    Reusable CLIP tagger that loads the model once and caches text embeddings.
    """
    def __init__(
        self,
        prompts: Sequence[str],
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        batch_size: int = 32,
    ):
        self.prompts = list(prompts)
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.dev = _device()
        
        # Load model once
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.dev)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Compute text embeddings once
        with torch.no_grad():
            text_tokens = self.tokenizer(self.prompts)
            text_tokens = text_tokens.to(self.dev)
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
    
    def tag_frames(
        self,
        frame_paths: Sequence[Path],
    ) -> VisualTagResult:
        """
        Encodes frames, averages embeddings, and computes cosine similarity vs prompt bank.
        Returns raw similarity scores (no thresholding).
        """
        if not frame_paths:
            raise ValueError("No frame paths provided.")

        # Image embeddings (batched)
        image_features_list: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(frame_paths), self.batch_size):
                batch = frame_paths[i : i + self.batch_size]
                images = []
                for p in batch:
                    with Image.open(p) as im:
                        images.append(self.preprocess(im.convert("RGB")))
                image_input = torch.stack(images).to(self.dev)
                feats = self.model.encode_image(image_input)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                image_features_list.append(feats)

            image_features = torch.cat(image_features_list, dim=0)
            avg = image_features.mean(dim=0)
            avg = avg / avg.norm(dim=-1, keepdim=True)

            # Cosine similarity since vectors are normalized
            sims = (self.text_features @ avg).detach().cpu().numpy().astype(np.float32)

        tags = {self.prompts[i]: float(sims[i]) for i in range(len(self.prompts))}
        emb = avg.detach().cpu().numpy().astype(np.float32)
        return VisualTagResult(embedding=emb, tags=tags)


def tag_frames_with_clip(
    frame_paths: Sequence[Path],
    prompts: Sequence[str],
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 32,
) -> VisualTagResult:
    """
    Legacy function for backward compatibility.
    Creates a temporary CLIPTagger instance (inefficient - use CLIPTagger directly for batch processing).
    """
    tagger = CLIPTagger(prompts, model_name=model_name, pretrained=pretrained, batch_size=batch_size)
    return tagger.tag_frames(frame_paths)


