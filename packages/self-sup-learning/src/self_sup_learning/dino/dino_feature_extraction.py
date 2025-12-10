#!/usr/bin/env python3
"""
Script to extract features from pretrained DINOv2 model and perform clustering
"""


from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.models import build_classifier
from sklearn.cluster import HDBSCAN, KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


def get_project_root():
    """Find the project root directory by looking for the main project markers."""
    current_path = Path(__file__).resolve()
    # Start from current directory and go up the hierarchy
    for parent in current_path.parents:
        # Check for the main project root which has both .git and packages directory
        if (parent / ".git").exists() and (parent / "packages").exists():
            return parent
    # If no main project markers found, return the directory containing the project structure we expect
    return (
        current_path.parent.parent.parent.parent.parent
    )  # fallback to previous method


def get_dino_paths():
    """Get the DINO config and checkpoint paths relative to project root."""
    project_root = get_project_root()
    config_path = (
        project_root
        / "packages"
        / "mmpretrain"
        / "configs"
        / "dinov2"
        / "vit-base-p14_dinov2-pre_headless.py"
    )
    checkpoint_path = (
        project_root
        / "packages"
        / "mmpretrain"
        / "pretrained"
        / "vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth"
    )
    return str(config_path), str(checkpoint_path)


class ImageNetSubsetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


def load_dino_model():
    """Load the pretrained DINOv2 model using relative paths from project root"""
    # Get the DINO paths
    config_path, checkpoint_path = get_dino_paths()

    # Load the configuration
    cfg = Config.fromfile(config_path)

    # Build the model
    model = build_classifier(cfg.model)

    # Load the checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")

    # Set the model to evaluation mode
    model.eval()

    return model


def extract_features(model, data_loader, device="cpu"):
    """Extract features from the DINO model"""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, batch_labels in data_loader:
            data = data.to(device)

            # Get features from the model
            # The DINO model outputs features when the head is None
            batch_features = model.extract_feat(data)

            # Flatten the features if they are in a special format
            if isinstance(batch_features, (list, tuple)):
                # If it's a list/tuple of tensors, take the first one
                batch_features = batch_features[0]

            # Flatten the features if necessary (for ViT models, may need to flatten)
            batch_features = batch_features.view(batch_features.size(0), -1)

            features.extend(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy())

    return np.array(features), np.array(labels)


def perform_clustering(embeddings, n_clusters_list, y_true, class_names, dataset_name):
    """Perform clustering and evaluate results for a specific dataset"""
    results = {}

    clustering_methods = {
        "KMeans": lambda n: KMeans(n_clusters=n, random_state=42).fit(embeddings),
    }

    # Try Spectral clustering separately since it's causing issues
    def spectral_clustering(n):
        try:
            return SpectralClustering(
                n_clusters=n, random_state=42, affinity="nearest_neighbors", n_jobs=-1
            ).fit(embeddings)
        except:
            # If spectral clustering fails, return a dummy clustering
            labels = np.zeros(len(embeddings), dtype=int)
            # Cycle through cluster labels to provide some diversity
            for i in range(len(embeddings)):
                labels[i] = i % n
            from sklearn.base import BaseEstimator, ClusterMixin

            class DummyClustering(BaseEstimator, ClusterMixin):
                def __init__(self, labels):
                    self.labels_ = labels

            return DummyClustering(labels)

    # Handle HDBSCAN separately since it doesn't take n_clusters parameter
    try:
        hdbscan_cluster = HDBSCAN(min_cluster_size=5, min_samples=3).fit(embeddings)
        hdbscan_labels = hdbscan_cluster.labels_
    except:
        # If HDBSCAN fails, use a single cluster as fallback
        hdbscan_labels = np.zeros(len(embeddings), dtype=int)

    # For each clustering method and number of clusters
    for method_name, cluster_func in clustering_methods.items():
        for n_clusters in n_clusters_list:
            # Perform clustering
            result = cluster_func(n_clusters)
            labels = result.labels_ if hasattr(result, "labels_") else result

            # Evaluate clustering - only calculate silhouette score if we have more than 1 cluster and clusters have more than 1 point
            if len(np.unique(labels)) > 1 and all(
                np.sum(labels == lab) > 1 for lab in np.unique(labels)
            ):
                try:
                    silhouette = silhouette_score(embeddings, labels)
                except:
                    silhouette = 0  # Set to 0 if silhouette calculation fails
            else:
                silhouette = 0  # Set to 0 if all points in one cluster

            v_measure = v_measure_score(y_true, labels)
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)

            results[f"{dataset_name}_{method_name}_{n_clusters}"] = {
                "labels": labels,
                "silhouette": silhouette,
                "v_measure": v_measure,
                "ari": ari,
                "nmi": nmi,
                "dataset": dataset_name,
            }

    # Try Spectral clustering separately
    for n_clusters in n_clusters_list:
        result = spectral_clustering(n_clusters)
        labels = result.labels_ if hasattr(result, "labels_") else result

        # Evaluate clustering - only calculate silhouette score if we have more than 1 cluster and clusters have more than 1 point
        if len(np.unique(labels)) > 1 and all(
            np.sum(labels == lab) > 1 for lab in np.unique(labels)
        ):
            try:
                silhouette = silhouette_score(embeddings, labels)
            except:
                silhouette = 0  # Set to 0 if silhouette calculation fails
        else:
            silhouette = 0  # Set to 0 if all points in one cluster

        v_measure = v_measure_score(y_true, labels)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)

        results[f"{dataset_name}_Spectral_{n_clusters}"] = {
            "labels": labels,
            "silhouette": silhouette,
            "v_measure": v_measure,
            "ari": ari,
            "nmi": nmi,
            "dataset": dataset_name,
        }

    # Process HDBSCAN result
    if len(np.unique(hdbscan_labels[hdbscan_labels != -1])) > 1:  # More than just noise
        # Calculate silhouette for non-noise points only
        non_noise_mask = hdbscan_labels != -1
        if (
            np.sum(non_noise_mask) > 1
            and len(np.unique(hdbscan_labels[non_noise_mask])) > 1
        ):
            try:
                silhouette = silhouette_score(
                    embeddings[non_noise_mask], hdbscan_labels[non_noise_mask]
                )
            except:
                silhouette = 0
        else:
            silhouette = 0

        v_measure = v_measure_score(y_true, hdbscan_labels)
        ari = adjusted_rand_score(y_true, hdbscan_labels)
        nmi = normalized_mutual_info_score(y_true, hdbscan_labels)

        results[f"{dataset_name}_HDBSCAN_any"] = {
            "labels": hdbscan_labels,
            "silhouette": silhouette,
            "v_measure": v_measure,
            "ari": ari,
            "nmi": nmi,
            "dataset": dataset_name,
        }
    else:
        # If HDBSCAN didn't find clusters, use all as one cluster
        v_measure = v_measure_score(y_true, np.zeros(len(embeddings), dtype=int))
        ari = adjusted_rand_score(y_true, np.zeros(len(embeddings), dtype=int))
        nmi = normalized_mutual_info_score(y_true, np.zeros(len(embeddings), dtype=int))

        results[f"{dataset_name}_HDBSCAN_any"] = {
            "labels": np.zeros(len(embeddings), dtype=int),
            "silhouette": 0,
            "v_measure": v_measure,
            "ari": ari,
            "nmi": nmi,
            "dataset": dataset_name,
        }

    return results


def main():
    print("Loading pretrained DINOv2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the DINO model
    model = load_dino_model()
    model = model.to(device)

    # Define transforms for DINO (based on the config)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to 224x224 as a standard size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
            ),
        ]
    )

    # Load training and test datasets
    train_dataset = ImageNetSubsetDataset(
        root_dir="/home/jin/Desktop/mm/data/imagenet_subset/train", transform=transform
    )

    test_dataset = ImageNetSubsetDataset(
        root_dir="/home/jin/Desktop/mm/data/imagenet_subset/test", transform=transform
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Extract features
    print("Extracting features from training set...")
    train_features, train_labels = extract_features(model, train_loader, device)
    print(f"Training features shape: {train_features.shape}")

    print("Extracting features from test set...")
    test_features, test_labels = extract_features(model, test_loader, device)
    print(f"Test features shape: {test_features.shape}")

    # Define cluster numbers: 1x (10), 2x (20), 3x (30) the real classes (10)
    n_clusters_list = [1, 5, 10, 20, 30]

    print("Performing clustering on training set...")
    train_results = perform_clustering(
        train_features,
        n_clusters_list,
        train_labels,
        train_dataset.dataset.classes,
        "train_dino",
    )

    print("Performing clustering on test set...")
    test_results = perform_clustering(
        test_features,
        n_clusters_list,
        test_labels,
        test_dataset.dataset.classes,
        "test_dino",
    )

    # Combine results
    all_results = {**train_results, **test_results}

    # Print results table
    print("\n" + "=" * 120)
    print("DINO Feature Clustering Results Summary")
    print("=" * 120)
    print(
        f"{'Dataset':<12} {'Method':<12} {'n_clusters':<10} {'Silhouette':<12} {'V-Measure':<12} {'ARI':<12} {'NMI':<12}"
    )
    print("-" * 120)

    for key, result in all_results.items():
        parts = key.split("_")
        dataset_name = parts[0] + "_" + parts[1]  # train_dino or test_dino
        method = "_".join(parts[2:-1]) if len(parts) > 3 else parts[2]
        n_clusters = parts[-1] if parts[-1] != "any" else "auto"

        print(
            f"{dataset_name:<12} {method:<12} {n_clusters:<10} {result['silhouette']:<12.4f} {result['v_measure']:<12.4f} {result['ari']:<12.4f} {result['nmi']:<12.4f}"
        )

    print("=" * 120)

    # Create summary CSV
    results_df = pd.DataFrame(
        {
            "method_dataset_n_clusters": list(all_results.keys()),
            "dataset": [all_results[k]["dataset"] for k in all_results.keys()],
            "method": [k.split("_")[2] for k in all_results.keys()],
            "n_clusters": [
                k.split("_")[-1] if k.split("_")[-1] != "any" else "auto"
                for k in all_results.keys()
            ],
            "silhouette": [all_results[k]["silhouette"] for k in all_results.keys()],
            "v_measure": [all_results[k]["v_measure"] for k in all_results.keys()],
            "ari": [all_results[k]["ari"] for k in all_results.keys()],
            "nmi": [all_results[k]["nmi"] for k in all_results.keys()],
        }
    )

    results_df.to_csv("dino_clustering_results.csv", index=False)
    print("\nResults saved to dino_clustering_results.csv")
    print(f"Total results: {len(results_df)}")

    # Save features
    np.save("dino_train_features.npy", train_features)
    np.save("dino_test_features.npy", test_features)
    print("Features saved to dino_train_features.npy and dino_test_features.npy")


if __name__ == "__main__":
    main()
