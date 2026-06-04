#!/usr/bin/env python3
"""
nanoVLM — Interactive Demo
===========================
Load a trained checkpoint and run image↔text retrieval from the terminal.

    python demo.py                          # uses checkpoints/best.pt
    python demo.py --checkpoint my_model.pt
"""

import argparse
import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from nanovlm import (
    DataConfig, ModelConfig,
    ShapesDataset, build_loaders,
    ImageEncoder, TextEncoder,
    embed_loader, recall_at_k,
    topk_texts_for_image, topk_images_for_text,
)


def _show(img_tensor, title=""):
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(img)
    if title:
        plt.title(title, fontsize=8)
    plt.axis("off")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="nanoVLM retrieval demo")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--k", type=int, default=3, help="Top-K results")
    ap.add_argument("--n", type=int, default=5, help="Number of queries to show")
    args = ap.parse_args()

    # ── load ──────────────────────────────────────
    if not os.path.isfile(args.checkpoint):
        print(f"  ✗ Checkpoint not found: {args.checkpoint}")
        print("    Run `python train.py` first.")
        return

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    dataset = ShapesDataset(data_cfg)
    _, val_loader = build_loaders(dataset, seed=42)

    img_enc = ImageEncoder(cfg=model_cfg)
    txt_enc = TextEncoder(vocab_size=dataset.vocab_size, cfg=model_cfg)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    img_enc.load_state_dict(ckpt["img_enc"])
    txt_enc.load_state_dict(ckpt["txt_enc"])
    print(f"  ✓ Loaded checkpoint (epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.4f})")

    # ── embed ─────────────────────────────────────
    bank = embed_loader(img_enc, txt_enc, val_loader, dataset)

    # ── metrics ───────────────────────────────────
    metrics = recall_at_k(bank["img_emb"], bank["txt_emb"])
    print("\n  Recall@K on validation set:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.2%}")

    # ── image → text ──────────────────────────────
    idxs = random.sample(range(len(bank["captions"])), args.n)
    print(f"\n  ── Image → Text (top-{args.k}) ──")
    for qi in idxs:
        matches = topk_texts_for_image(bank, qi, args.k)
        gt = bank["captions"][qi]
        print(f"\n  Query image [{qi}]  (ground-truth: {gt})")
        for rank, (j, cap, score) in enumerate(matches, 1):
            marker = "✓" if cap == gt else " "
            print(f"    {rank}. {cap}  ({score:.3f}) {marker}")
        _show(bank["images"][qi], title=gt)

    # ── text → image ──────────────────────────────
    print(f"\n  ── Text → Image (top-{args.k}) ──")
    for qi in idxs:
        matches = topk_images_for_text(bank, qi, args.k)
        print(f"\n  Query text: \"{bank['captions'][qi]}\"")
        for rank, (j, score) in enumerate(matches, 1):
            _show(bank["images"][j], title=f"#{rank} ({score:.3f}) {bank['captions'][j]}")


if __name__ == "__main__":
    main()
