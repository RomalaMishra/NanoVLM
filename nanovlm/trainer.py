"""
nanoVLM — Training Engine
==========================
Self-contained training loop with:
  • train / val split tracking
  • best-model checkpointing
  • loss-curve history returned for plotting
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List

import torch
from torch.utils.data import DataLoader

from .models import ImageEncoder, TextEncoder
from .loss import clip_loss
from .config import TrainConfig


@dataclass
class TrainHistory:
    """Accumulated metrics across epochs."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)


def train(
    img_enc: ImageEncoder,
    txt_enc: TextEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig | None = None,
) -> TrainHistory:
    """
    Train both encoders end-to-end with the CLIP objective.

    Returns a ``TrainHistory`` with per-epoch losses for plotting.
    """
    cfg = cfg or TrainConfig()
    device = torch.device(cfg.device)
    img_enc.to(device)
    txt_enc.to(device)

    params = list(img_enc.parameters()) + list(txt_enc.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_val = float("inf")
    history = TrainHistory()

    for epoch in range(1, cfg.epochs + 1):
        # ── train ──────────────────────────────────────
        img_enc.train()
        txt_enc.train()
        running, n = 0.0, 0

        for imgs, toks in train_loader:
            imgs, toks = imgs.to(device), toks.to(device)
            optimizer.zero_grad(set_to_none=True)

            loss = clip_loss(img_enc(imgs), txt_enc(toks), cfg.temperature)
            loss.backward()
            optimizer.step()

            running += loss.item() * imgs.size(0)
            n += imgs.size(0)

        train_loss = running / n

        # ── validate ───────────────────────────────────
        img_enc.eval()
        txt_enc.eval()
        running, n = 0.0, 0
        with torch.no_grad():
            for imgs, toks in val_loader:
                imgs, toks = imgs.to(device), toks.to(device)
                loss = clip_loss(img_enc(imgs), txt_enc(toks), cfg.temperature)
                running += loss.item() * imgs.size(0)
                n += imgs.size(0)
        val_loss = running / n

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)

        # ── checkpoint ─────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "img_enc": img_enc.state_dict(),
                "txt_enc": txt_enc.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(cfg.checkpoint_dir, "best.pt"))

        # ── logging ────────────────────────────────────
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>3}/{cfg.epochs}"
                f"  │  train {train_loss:.4f}"
                f"  │  val {val_loss:.4f}"
                f"{'  ★' if val_loss <= best_val else ''}"
            )

    print(f"\n  ✓ Training complete — best val loss: {best_val:.4f}")
    return history
