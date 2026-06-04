#!/usr/bin/env python3
"""
nanoVLM — Train
================
End-to-end training script.  Run from the project root:

    python train.py              # defaults
    python train.py --epochs 100 --lr 1e-3 --batch_size 24
"""

import argparse
import torch

from nanovlm import (
    DataConfig, ModelConfig, TrainConfig,
    ShapesDataset, build_loaders,
    ImageEncoder, TextEncoder,
    train,
    embed_loader, recall_at_k,
    plot_loss, plot_dataset_grid, plot_similarity_matrix, plot_retrieval_gallery,
)


def main():
    ap = argparse.ArgumentParser(description="Train nanoVLM")
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int,   default=12)
    ap.add_argument("--embed_dim",  type=int,   default=64)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--no-plots",   action="store_true", help="Skip generating asset images")
    args = ap.parse_args()

    # ── configs ───────────────────────────────────
    data_cfg  = DataConfig()
    model_cfg = ModelConfig(embed_dim=args.embed_dim)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    torch.manual_seed(train_cfg.seed)

    # ── data ──────────────────────────────────────
    print("\n╔══════════════════════════════════════╗")
    print("║          n a n o V L M               ║")
    print("╚══════════════════════════════════════╝\n")

    dataset = ShapesDataset(data_cfg)
    train_loader, val_loader = build_loaders(
        dataset,
        batch_size=train_cfg.batch_size,
        train_split=data_cfg.train_split,
        seed=train_cfg.seed,
    )
    print(f"  Dataset : {len(dataset)} samples  ({len(dataset.vocab)} vocab)")
    print(f"  Split   : {len(train_loader.dataset)} train / {len(val_loader.dataset)} val")
    print(f"  Device  : {train_cfg.device}\n")

    # ── models ────────────────────────────────────
    img_enc = ImageEncoder(cfg=model_cfg)
    txt_enc = TextEncoder(vocab_size=dataset.vocab_size, cfg=model_cfg)
    n_params = sum(p.numel() for p in list(img_enc.parameters()) + list(txt_enc.parameters()))
    print(f"  Parameters : {n_params:,}\n")

    # ── train ─────────────────────────────────────
    print("  Training …")
    history = train(img_enc, txt_enc, train_loader, val_loader, train_cfg)

    # ── evaluate ──────────────────────────────────
    device = train_cfg.device
    bank = embed_loader(img_enc, txt_enc, val_loader, dataset, device)
    metrics = recall_at_k(bank["img_emb"], bank["txt_emb"])
    print("\n  Retrieval metrics (val set):")
    for k, v in metrics.items():
        print(f"    {k}: {v:.2%}")

    # ── plots ─────────────────────────────────────
    if not args.no_plots:
        print("\n  Generating README assets …")
        plot_loss(history)
        plot_dataset_grid(dataset)
        plot_similarity_matrix(bank)
        plot_retrieval_gallery(bank)
        print("  Done ✓\n")


if __name__ == "__main__":
    main()
