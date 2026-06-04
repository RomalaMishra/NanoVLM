"""
nanoVLM — A minimal CLIP implementation from scratch
=====================================================
Learn contrastive vision-language alignment on a toy shapes dataset
with ~15 k parameters and zero external pretrained weights.
"""

from .config import DataConfig, ModelConfig, TrainConfig
from .dataset import ShapesDataset, build_loaders, draw_sample
from .models import ImageEncoder, TextEncoder
from .loss import clip_loss
from .trainer import train, TrainHistory
from .evaluate import embed_loader, recall_at_k, topk_texts_for_image, topk_images_for_text
from .visualize import (
    plot_loss,
    plot_embeddings,
    plot_dataset_grid,
    plot_retrieval_gallery,
    plot_similarity_matrix,
)

__all__ = [
    # config
    "DataConfig", "ModelConfig", "TrainConfig",
    # data
    "ShapesDataset", "build_loaders", "draw_sample",
    # models
    "ImageEncoder", "TextEncoder",
    # loss
    "clip_loss",
    # training
    "train", "TrainHistory",
    # evaluation
    "embed_loader", "recall_at_k",
    "topk_texts_for_image", "topk_images_for_text",
    # visualization
    "plot_loss", "plot_embeddings", "plot_dataset_grid",
    "plot_retrieval_gallery", "plot_similarity_matrix",
]
