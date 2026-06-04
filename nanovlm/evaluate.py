"""
nanoVLM — Evaluation & Retrieval
==================================
Quantitative metrics  (Recall@K, embedding similarity)  and
qualitative helpers  (image→text / text→image retrieval demos).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import ImageEncoder, TextEncoder
from .dataset import ShapesDataset


# ──────────────────────────────────────────────
#  Build an embedding bank from a DataLoader
# ──────────────────────────────────────────────

@torch.no_grad()
def embed_loader(
    img_enc: ImageEncoder,
    txt_enc: TextEncoder,
    loader: DataLoader,
    dataset: ShapesDataset,
    device: str = "cpu",
) -> dict:
    """
    Encode every sample in *loader* and return a dict with:
      images   – (N, 3, H, W)  raw images
      img_emb  – (N, D)        image embeddings
      txt_emb  – (N, D)        text  embeddings
      captions – list[str]     decoded captions
    """
    img_enc.eval()
    txt_enc.eval()
    all_imgs, all_toks, captions = [], [], []

    for imgs, toks in loader:
        all_imgs.append(imgs)
        all_toks.append(toks)
        for t in toks:
            captions.append(dataset.decode_tokens(t.tolist()))

    images = torch.cat(all_imgs).to(device)
    tokens = torch.cat(all_toks).to(device)

    return {
        "images": images.cpu(),
        "img_emb": img_enc(images),
        "txt_emb": txt_enc(tokens),
        "captions": captions,
    }


# ──────────────────────────────────────────────
#  Recall@K
# ──────────────────────────────────────────────

def recall_at_k(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """
    Compute image→text and text→image Recall@K.

    Returns a dict like:
        {"i2t_R@1": 0.72, "i2t_R@3": 0.93, "t2i_R@1": 0.70, ...}
    """
    sims = img_emb @ txt_emb.T                   # (N, N)
    N = sims.size(0)
    targets = torch.arange(N, device=sims.device)
    results: dict[str, float] = {}

    for direction, mat in [("i2t", sims), ("t2i", sims.T)]:
        ranked = mat.argsort(dim=1, descending=True)
        for k in ks:
            hits = (ranked[:, :k] == targets.unsqueeze(1)).any(dim=1)
            results[f"{direction}_R@{k}"] = hits.float().mean().item()
    return results


# ──────────────────────────────────────────────
#  Top-K retrieval helpers (for demo / notebook)
# ──────────────────────────────────────────────

def topk_texts_for_image(
    bank: dict,
    image_idx: int,
    k: int = 3,
) -> list[tuple[int, str, float]]:
    """Return the k best-matching captions for a given image."""
    sims = (bank["img_emb"] @ bank["txt_emb"].T).softmax(dim=1)
    scores, idxs = sims[image_idx].topk(k)
    return [(j.item(), bank["captions"][j], s.item()) for j, s in zip(idxs, scores)]


def topk_images_for_text(
    bank: dict,
    text_idx: int,
    k: int = 3,
) -> list[tuple[int, float]]:
    """Return the k best-matching image indices for a given caption."""
    sims = (bank["txt_emb"] @ bank["img_emb"].T).softmax(dim=1)
    scores, idxs = sims[text_idx].topk(k)
    return [(j.item(), s.item()) for j, s in zip(idxs, scores)]
