"""
nanoVLM — Synthetic Shapes Dataset
====================================
Procedurally generates tiny RGB images of coloured shapes placed at
various positions, paired with text captions like "red circle top-left".
"""

from typing import Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageDraw

from .config import DataConfig


# ──────────────────────────────────────────────
#  Drawing helpers
# ──────────────────────────────────────────────

def _bbox(position: str, img_size: int, margin: int = 6):
    """Return (x0, y0, x1, y1) for a shape given its position label."""
    w = h = img_size - 2 * margin

    # --- horizontal ---
    if any(tag in position for tag in ("left",)):
        x0, x1 = margin, margin + w // 2
    elif any(tag in position for tag in ("right",)):
        x0, x1 = margin + w // 2, img_size - margin
    else:                                       # center / top / bottom
        x0, x1 = margin + w // 4, margin + 3 * w // 4

    # --- vertical ---
    if any(tag in position for tag in ("top",)):
        y0, y1 = margin, margin + h // 2
    elif any(tag in position for tag in ("bottom",)):
        y0, y1 = margin + h // 2, img_size - margin
    else:                                       # center / left / right
        y0, y1 = margin + h // 4, margin + 3 * h // 4

    return x0, y0, x1, y1


def draw_sample(
    color: str,
    shape: str,
    position: str,
    img_size: int = 32,
) -> Image.Image:
    """Render a single coloured shape on a white canvas."""
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = _bbox(position, img_size)

    if shape == "square":
        draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")
    elif shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=color, outline="black")
    elif shape == "triangle":
        cx = (x0 + x1) // 2
        draw.polygon([(cx, y0), (x0, y1), (x1, y1)], fill=color, outline="black")
    return img


# ──────────────────────────────────────────────
#  PyTorch Dataset
# ──────────────────────────────────────────────

class ShapesDataset(Dataset):
    """
    Cartesian product of (colors × shapes × positions).
    Each item returns (image_tensor [C,H,W], token_ids [L]).
    """

    def __init__(self, cfg: DataConfig | None = None):
        self.cfg = cfg or DataConfig()
        self.images: List[torch.Tensor] = []
        self.captions: List[str] = []

        for c in self.cfg.colors:
            for s in self.cfg.shapes:
                for p in self.cfg.positions:
                    img = draw_sample(c, s, p, self.cfg.img_size)
                    self.images.append(
                        torch.from_numpy(np.asarray(img))
                        .permute(2, 0, 1)
                        .float()
                        / 255.0
                    )
                    self.captions.append(f"{c} {s} {p}")

        self.vocab, self.word2idx = self._build_vocab()

    # ---- vocabulary --------------------------------------------------
    def _build_vocab(self) -> Tuple[List[str], Dict[str, int]]:
        words = sorted({w for cap in self.captions for w in cap.split()})
        vocab = ["[CLS]"] + words
        w2i = {w: i for i, w in enumerate(vocab)}
        return vocab, w2i

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_text(self, text: str) -> torch.Tensor:
        ids = [self.word2idx["[CLS]"]] + [self.word2idx[w] for w in text.split()]
        return torch.tensor(ids, dtype=torch.long)

    def decode_tokens(self, ids) -> str:
        return " ".join(
            self.vocab[i] for i in ids if self.vocab[i] != "[CLS]"
        )

    # ---- Dataset interface -------------------------------------------
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.encode_text(self.captions[idx])


# ──────────────────────────────────────────────
#  Loader factory
# ──────────────────────────────────────────────

def build_loaders(
    dataset: ShapesDataset,
    batch_size: int = 12,
    train_split: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Split *dataset* and return (train_loader, val_loader)."""
    n_train = int(train_split * len(dataset))
    n_val = len(dataset) - n_train
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
