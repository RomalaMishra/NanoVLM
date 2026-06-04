"""
nanoVLM — Model Architectures
===============================
Two lightweight encoders that map images and text into a shared
embedding space, exactly as in the original CLIP paper (Radford et al., 2021).

ImageEncoder  — 4-layer CNN  →  global-average-pool  →  linear projection
TextEncoder   — token + position embeddings  →  multi-head self-attention  →  [CLS] projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class ImageEncoder(nn.Module):
    """
    Tiny convolutional vision backbone.

    32×32 RGB  →  conv stack (3→32→64→128→256)  →  GAP  →  Linear(256, embed_dim)
    Output is L2-normalised so dot products equal cosine similarity.
    """

    def __init__(self, embed_dim: int | None = None, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        dim = embed_dim or cfg.embed_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.proj = nn.Linear(256, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, embed_dim)"""
        x = self.backbone(x)
        x = x.mean(dim=[2, 3])         # global average pooling
        x = self.proj(x)
        x = F.normalize(self.norm(x), dim=-1)
        return x


class TextEncoder(nn.Module):
    """
    Minimal transformer text encoder (single attention layer).

    Token ids  →  token + position embeddings  →  MHA self-attention  →  [CLS] vector  →  projection
    Output is L2-normalised to match the image encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int | None = None,
        num_heads: int | None = None,
        context_window: int | None = None,
        cfg: ModelConfig | None = None,
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        dim = embed_dim or cfg.embed_dim
        heads = num_heads or cfg.attention_heads
        ctx = context_window or cfg.context_window

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(ctx, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L) → (B, embed_dim)"""
        B, L = tokens.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)

        x = self.token_emb(tokens) + self.pos_emb(positions)
        x, _ = self.attn(x, x, x)      # self-attention
        cls_vec = x[:, 0]               # [CLS] representation
        out = self.proj(cls_vec)
        return F.normalize(self.norm(out), dim=-1)
