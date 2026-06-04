"""
nanoVLM — Visualisation Helpers
================================
Matplotlib-based plots for README assets and interactive exploration:
  • loss curves
  • embedding heatmaps (pre- vs. post-training)
  • dataset sample grids
  • image→text / text→image retrieval galleries
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from .trainer import TrainHistory
from .evaluate import topk_texts_for_image, topk_images_for_text


# ── style defaults ────────────────────────────────────────
_FIG_DIR = Path("assets")
_DPI = 150

COLORS = {
    "train": "#4361ee",
    "val": "#f72585",
    "accent": "#7209b7",
    "bg": "#f8f9fa",
}


def _save(fig, name: str):
    _FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(_FIG_DIR / name, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {_FIG_DIR / name}")


# ──────────────────────────────────────────────
#  1. Loss curves
# ──────────────────────────────────────────────

def plot_loss(history: TrainHistory, save: bool = True):
    """Plot train vs val loss and optionally save to assets/."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    epochs = range(1, len(history.train_losses) + 1)
    ax.plot(epochs, history.train_losses, color=COLORS["train"], lw=2, label="Train")
    ax.plot(epochs, history.val_losses, color=COLORS["val"], lw=2, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CLIP Loss")
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save:
        _save(fig, "loss_curve.png")
    else:
        plt.show()


# ──────────────────────────────────────────────
#  2. Embedding heatmaps
# ──────────────────────────────────────────────

def plot_embeddings(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    title_prefix: str = "",
    save_name: str | None = None,
):
    """Side-by-side heatmap bars of an image and text embedding vector."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 1.8))
    for ax, emb, label in zip(axes, [img_emb, txt_emb], ["Image Emb", "Text Emb"]):
        ax.imshow(emb.reshape(1, -1), aspect="auto", cmap="viridis")
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{title_prefix} Embedding Vectors", fontweight="bold", fontsize=10)
    fig.tight_layout()
    if save_name:
        _save(fig, save_name)
    else:
        plt.show()


# ──────────────────────────────────────────────
#  3. Dataset sample grid
# ──────────────────────────────────────────────

def plot_dataset_grid(
    dataset,
    n: int = 18,
    seed: int = 0,
    save: bool = True,
):
    """Show a random grid of dataset samples with captions."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    cols = 6
    rows = (len(idxs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.7))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for i, idx in enumerate(idxs):
        img_t, tok_t = dataset[int(idx)]
        img = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cap = dataset.decode_tokens(tok_t.tolist())
        r, c = divmod(i, cols)
        axes[r, c].imshow(img)
        axes[r, c].set_title(cap, fontsize=6, pad=2)

    fig.suptitle("Sample Dataset Entries", fontweight="bold", y=1.01)
    fig.tight_layout()
    if save:
        _save(fig, "dataset_samples.png")
    else:
        plt.show()


# ──────────────────────────────────────────────
#  4. Retrieval gallery
# ──────────────────────────────────────────────

def plot_retrieval_gallery(
    bank: dict,
    n_queries: int = 4,
    k: int = 3,
    seed: int = 7,
    save: bool = True,
):
    """
    Show *n_queries* random query images with their top-k retrieved captions,
    plus *n_queries* random query texts with their top-k retrieved images.
    """
    rng = np.random.default_rng(seed)
    N = len(bank["captions"])

    # ── image → text ─────────────────────────────────
    fig_i2t, axes = plt.subplots(n_queries, 1, figsize=(7, n_queries * 1.5))
    if n_queries == 1:
        axes = [axes]
    idxs = rng.choice(N, size=n_queries, replace=False)
    for ax, qi in zip(axes, idxs):
        img = (bank["images"][qi].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        matches = topk_texts_for_image(bank, int(qi), k)
        txt = "  |  ".join(f"{cap} ({score:.2f})" for _, cap, score in matches)
        ax.imshow(img)
        ax.set_title(f"→  {txt}", fontsize=7, loc="left")
        ax.axis("off")
    fig_i2t.suptitle("Image → Text Retrieval", fontweight="bold")
    fig_i2t.tight_layout()
    if save:
        _save(fig_i2t, "retrieval_i2t.png")
    else:
        plt.show()

    # ── text → image ─────────────────────────────────
    idxs = rng.choice(N, size=n_queries, replace=False)
    fig_t2i = plt.figure(figsize=(7, n_queries * 1.8))
    outer = gridspec.GridSpec(n_queries, 1, figure=fig_t2i, hspace=0.5)

    for row, qi in enumerate(idxs):
        matches = topk_images_for_text(bank, int(qi), k)
        inner = gridspec.GridSpecFromSubplotSpec(1, k + 1, subplot_spec=outer[row], wspace=0.15)
        # query label
        ax_label = fig_t2i.add_subplot(inner[0])
        ax_label.text(
            0.5, 0.5, bank["captions"][qi],
            ha="center", va="center", fontsize=8, fontweight="bold",
            transform=ax_label.transAxes,
        )
        ax_label.axis("off")
        # matched images
        for col, (j, score) in enumerate(matches):
            ax = fig_t2i.add_subplot(inner[col + 1])
            img = (bank["images"][j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            ax.imshow(img)
            ax.set_title(f"{score:.2f}", fontsize=6)
            ax.axis("off")

    fig_t2i.suptitle("Text → Image Retrieval", fontweight="bold", y=1.01)
    if save:
        _save(fig_t2i, "retrieval_t2i.png")
    else:
        plt.show()


# ──────────────────────────────────────────────
#  5. Similarity matrix
# ──────────────────────────────────────────────

def plot_similarity_matrix(
    bank: dict,
    n: int = 12,
    seed: int = 3,
    save: bool = True,
):
    """Heatmap of the image–text cosine similarity matrix for *n* samples."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(bank["captions"]), size=min(n, len(bank["captions"])), replace=False)

    ie = bank["img_emb"][idxs]
    te = bank["txt_emb"][idxs]
    sims = (ie @ te.T).cpu().numpy()
    caps = [bank["captions"][i] for i in idxs]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sims, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(caps)))
    ax.set_yticks(range(len(caps)))
    ax.set_xticklabels(caps, rotation=55, ha="right", fontsize=6)
    ax.set_yticklabels(caps, fontsize=6)
    ax.set_xlabel("Text")
    ax.set_ylabel("Image")
    ax.set_title("Image ↔ Text Cosine Similarity", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if save:
        _save(fig, "similarity_matrix.png")
    else:
        plt.show()
