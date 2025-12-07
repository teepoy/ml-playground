# Self-Supervised Learning Package

This package provides a comprehensive suite of tools for self-supervised learning experiments and analysis.

## Package Structure

```
self_sup_learning/
├── dino/                    # DINO feature extraction and analysis
│   ├── dino_feature_extraction.py
│   └── analyze_dino_results.py
├── dino_finetune/           # DINO pre-training and fine-tuning
│   ├── dino_pretrain.py
│   └── dino_finetune.py
├── mae_finetune/            # MAE pre-training and fine-tuning
│   ├── mae_pretrain.py
│   └── mae_finetune.py
├── mae_clustering/          # MAE clustering analysis
│   ├── clustering_analysis.py
│   ├── probe_mae_dimensions.py
│   ├── quick_clustering_test.py
│   ├── visualize_selected_classes.py
│   └── configs/             # Configuration files for clustering
├── vae/                     # VAE models
│   ├── imagenet_vae.py
│   └── compare_vae_results.py
├── vae_clustering/          # VAE clustering analysis
│   ├── imagenet_clustering.py
│   └── analyze_imagenet_clustering.py
└── iris/                    # Iris dataset examples
    ├── iris_vae.py
    ├── iris_clustering.py
    ├── download_iris.py
    ├── use_iris_vae.py
    └── detailed_clustering_analysis.py
```

## Modules

### DINO (Self-Distillation with No Labels)

- **dino/**: Feature extraction and analysis using pre-trained DINOv2 models
- **dino_finetune/**: Pre-training and fine-tuning scripts for DINOv2 models using mmpretrain

### MAE (Masked Autoencoder)

- **mae_finetune/**: Pre-training and fine-tuning scripts for MAE models using mmpretrain
- **mae_clustering/**: Clustering analysis tools for MAE embeddings with support for:
  - K-means clustering
  - Variational Bayesian Gaussian Mixture Models (VBGMM)
  - HDBSCAN
  - Comprehensive evaluation metrics
  - UMAP/t-SNE visualization

### VAE (Variational Autoencoder)

- **vae/**: VAE model implementations for ImageNet
- **vae_clustering/**: Clustering analysis on VAE latent representations

### Iris Examples

- **iris/**: Example implementations using the Iris dataset for:
  - VAE training
  - Clustering analysis
  - Feature extraction

## Usage

```python
# Import the package
import self_sup_learning

# Use specific modules
from self_sup_learning import dino, mae_clustering, vae

# Access submodules
from self_sup_learning.mae_clustering import clustering_analysis
```

## Features

- Pre-training and fine-tuning for DINO and MAE models
- Multiple clustering algorithms (K-means, VBGMM, HDBSCAN)
- Comprehensive evaluation metrics
- Visualization tools (UMAP, t-SNE, PCA)
- Hydra configuration management
- LanceDB integration for embedding storage
- Iris dataset examples for quick testing

## Dependencies

See `pyproject.toml` for the full list of dependencies, which includes:
- PyTorch and torchvision
- mmpretrain, mmcv, mmengine
- scikit-learn
- HDBSCAN, UMAP
- Hydra for configuration management
- And more...

## Configuration

The `mae_clustering` module uses Hydra for configuration management. Configuration files are located in:
- `mae_clustering/configs/base_config.yaml`
- `mae_clustering/configs/clustering_config.yaml`
- `mae_clustering/configs/kmeans/`
- `mae_clustering/configs/vbgmm/`
- `mae_clustering/configs/hdbscan/`

## Development

This package is part of the ml-playground repository and follows the workspace structure defined in the root `pyproject.toml`.
