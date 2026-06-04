"""
nanoVLM — Configuration
=======================
All hyperparameters and dataset knobs live here so every other module
stays free of magic numbers.
"""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class DataConfig:
    """Synthetic dataset parameters."""
    img_size: int = 32
    colors: List[str] = field(default_factory=lambda: [
        "red", "green", "blue", "yellow", "purple",
        "orange", "pink", "brown", "gray",
    ])
    shapes: List[str] = field(default_factory=lambda: [
        "square", "circle", "triangle",
    ])
    positions: List[str] = field(default_factory=lambda: [
        "left", "center", "right", "top", "bottom",
        "top-left", "top-right", "bottom-left", "bottom-right",
    ])
    train_split: float = 0.8


@dataclass
class ModelConfig:
    """Architecture parameters."""
    embed_dim: int = 64
    attention_heads: int = 4
    context_window: int = 4          # [CLS] + 3 caption words


@dataclass
class TrainConfig:
    """Training parameters."""
    batch_size: int = 12
    epochs: int = 50
    lr: float = 3e-4
    temperature: float = 0.07
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 5               # print every N epochs
