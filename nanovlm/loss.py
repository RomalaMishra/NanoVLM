"""
nanoVLM — Contrastive Loss
===========================
Symmetric cross-entropy over cosine-similarity logits, exactly as
described in *Learning Transferable Visual Models From Natural Language
Supervision* (Radford et al., 2021).
"""

import torch
import torch.nn.functional as F


def clip_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Compute the symmetric CLIP contrastive loss.

    Parameters
    ----------
    img_emb : (B, D) — L2-normalised image embeddings
    txt_emb : (B, D) — L2-normalised text  embeddings
    temperature : learnable or fixed scalar that sharpens the softmax

    Returns
    -------
    Scalar loss (mean of image→text and text→image cross-entropy).
    """
    # cosine similarity matrix scaled by temperature
    logits = img_emb @ txt_emb.T / temperature          # (B, B)

    # diagonal entries are the positive pairs
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)          # image → text
    loss_t2i = F.cross_entropy(logits.T, targets)        # text  → image

    return (loss_i2t + loss_t2i) / 2.0
