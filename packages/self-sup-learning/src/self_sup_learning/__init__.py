"""Self-supervised learning package.

This package provides tools for self-supervised learning including:
- DINO (self-distillation with no labels) feature extraction and fine-tuning
- MAE (Masked Autoencoder) feature extraction and fine-tuning
- VAE (Variational Autoencoder) training and clustering
- Various clustering analysis tools
- Iris dataset examples for testing
"""

__version__ = "0.1.0"

# Import submodules
from . import (
    dino,
    dino_finetune,
    iris,
    mae_clustering,
    mae_finetune,
    vae,
    vae_clustering,
)

__all__ = [
    "dino",
    "dino_finetune",
    "mae_finetune",
    "mae_clustering",
    "vae",
    "vae_clustering",
    "iris",
]
