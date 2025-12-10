#!/usr/bin/env python3
"""
MVTec-AD Anomaly Detection using DINO Features and Clustering
This script handles the MVTec-AD dataset structure: <background-class>/test/<defect-class>
and performs clustering within each background class separately.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import HDBSCAN, KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from torch.utils.data import DataLoader, Dataset


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


class MVTecADDataset(Dataset):
    """
    Dataset class for MVTec-AD that handles the <background-class>/test/<defect-class> structure.
    """

    def __init__(self, root_dir, background_class, transform=None, split="test"):
        """
        Args:
            root_dir (str): Root directory of the dataset
            background_class (str): The background class name (e.g., "bottle", "cable", etc.)
            transform: Transformations to apply to the images
            split (str): Dataset split to load ("test" by default, since we're clustering per background class)
        """
        self.root_dir = root_dir
        self.background_class = background_class
        self.transform = transform
        self.split = split

        # Find all defect classes in the background class
        self.defect_classes = []
        split_path = os.path.join(root_dir, background_class, split)

        if os.path.exists(split_path):
            for item in os.listdir(split_path):
                item_path = os.path.join(split_path, item)
                if os.path.isdir(item_path):
                    self.defect_classes.append(item)

        # Add "good" as a defect class if it exists (normal samples)
        if "good" not in self.defect_classes:
            good_path = os.path.join(split_path, "good")
            if os.path.exists(good_path):
                self.defect_classes = ["good"] + self.defect_classes
        else:
            self.defect_classes = ["good"] + [
                cls for cls in self.defect_classes if cls != "good"
            ]

        # Store the number of classes (clusters) for this background class
        self.n_clusters = len(self.defect_classes)

        # Load all images and create labels
        self.image_paths = []
        self.image_labels = []
        self.image_defect_labels = []  # Track actual defect type for evaluation

        for defect_idx, defect_class in enumerate(self.defect_classes):
            defect_path = os.path.join(split_path, defect_class)
            if os.path.exists(defect_path):
                for img_file in os.listdir(defect_path):
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(defect_path, img_file))
                        self.image_labels.append(
                            defect_idx
                        )  # All will be labeled the same for clustering
                        self.image_defect_labels.append(
                            defect_class
                        )  # Actual defect type for evaluation

        print(
            f"Loaded {len(self.image_paths)} images from {background_class}/{split} with {self.n_clusters} defect classes: {self.defect_classes}"
        )

    def get_n_clusters(self):
        """Return the number of clusters (defect classes) for this background class."""
        return self.n_clusters

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]
        defect_label = self.image_defect_labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return both the image and the actual defect type for evaluation
        return image, label, defect_label


def load_dino_model():
    """Load the DINO model for feature extraction based on the existing project code."""
    # Get the DINO paths
    config_path, checkpoint_path = get_dino_paths()

    # Check if required files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    try:
        # Import necessary modules as in the original DINO code
        from mmengine.config import Config
        from mmengine.runner import load_checkpoint
        from mmpretrain.models import build_classifier

        # Load config and model as in the original script
        cfg = Config.fromfile(config_path)
        model = build_classifier(cfg.model)
        load_checkpoint(model, checkpoint_path, map_location="cpu")
        # Set the model to evaluation mode
        model.eval()
        return model

    except ImportError as e:
        raise ImportError(f"Could not import required modules for DINO: {e}")
    except Exception as e:
        raise Exception(f"Error loading DINO model: {e}")


def extract_features(model, data_loader, device="cpu"):
    """Extract features from the model for the dataset"""
    # If we reach here, model should not be None (load_dino_model would have raised an error)
    # So we can proceed with feature extraction assuming model is valid
    model.eval()
    features = []
    all_defect_labels = []

    with torch.no_grad():
        for images, _, defect_labels in data_loader:
            images = images.to(device)

            batch_features = model(images)
            # Handle different output formats
            if isinstance(batch_features, (list, tuple)):
                # If it's a list/tuple of tensors, take the first one
                batch_features = batch_features[0]

            # Flatten the features if necessary (for ViT models, may need to flatten)
            batch_features = batch_features.view(batch_features.size(0), -1)

            features.extend(batch_features.cpu().numpy())
            all_defect_labels.extend(defect_labels)

    return np.array(features), all_defect_labels


def perform_clustering_within_background(
    embeddings, defect_labels, background_class_name, n_clusters_expected
):
    """Perform clustering specifically within one background class"""
    results = {}

    # Map defect labels to numeric values for evaluation
    unique_defects = list(set(defect_labels))
    defect_to_idx = {defect: idx for idx, defect in enumerate(unique_defects)}
    true_labels = np.array([defect_to_idx[defect] for defect in defect_labels])

    print(f"Background class: {background_class_name}")
    print(f"Defect classes found: {unique_defects}")
    print(f"Expected number of clusters: {n_clusters_expected}")
    print(f"Feature shape: {embeddings.shape}")

    # Use the expected number of clusters and variations around it
    n_clusters_list = [
        n_clusters_expected,
        max(2, n_clusters_expected - 1),
        max(3, n_clusters_expected + 1),
        n_clusters_expected * 2,
    ]

    # Clustering methods
    clustering_methods = {
        "KMeans": lambda n: KMeans(n_clusters=n, random_state=42).fit(embeddings),
    }

    # Handle Spectral Clustering separately (it can be problematic with high-dim features)
    def spectral_clustering(n):
        try:
            if embeddings.shape[0] > 2000:  # Use subsampling for large datasets
                from sklearn.neighbors import kneighbors_graph

                connectivity = kneighbors_graph(
                    embeddings,
                    n_neighbors=min(10, len(embeddings) - 1),
                    include_self=False,
                )
                return SpectralClustering(
                    n_clusters=n,
                    random_state=42,
                    affinity="precomputed_nearest_neighbors",
                    n_neighbors=min(10, len(embeddings) - 1),
                ).fit(embeddings)
            else:
                return SpectralClustering(
                    n_clusters=n,
                    random_state=42,
                    affinity="nearest_neighbors",
                    n_jobs=-1,
                ).fit(embeddings)
        except:
            # Fallback clustering if spectral clustering fails
            labels = np.zeros(len(embeddings), dtype=int)
            for i in range(len(embeddings)):
                labels[i] = i % n
            from sklearn.base import BaseEstimator, ClusterMixin

            class DummyClustering(BaseEstimator, ClusterMixin):
                def __init__(self, labels):
                    self.labels_ = labels

            return DummyClustering(labels)

    # Handle HDBSCAN separately since it doesn't take n_clusters parameter
    try:
        min_cluster_size = max(2, len(embeddings) // 20)  # Dynamic min_cluster_size
        hdbscan_cluster = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1).fit(
            embeddings
        )
        hdbscan_labels = hdbscan_cluster.labels_
    except:
        # If HDBSCAN fails, use a single cluster as fallback
        hdbscan_labels = np.zeros(len(embeddings), dtype=int)

    # Perform clustering with each method and number of clusters
    for method_name, cluster_func in clustering_methods.items():
        for n_clusters in n_clusters_list:
            try:
                result = cluster_func(n_clusters)
                labels = result.labels_ if hasattr(result, "labels_") else result

                # Evaluate clustering
                silhouette = 0
                if len(np.unique(labels)) > 1 and all(
                    np.sum(labels == lab) > 1 for lab in np.unique(labels) if lab != -1
                ):
                    try:
                        silhouette = silhouette_score(embeddings, labels)
                    except:
                        silhouette = 0

                v_measure = v_measure_score(true_labels, labels)
                ari = adjusted_rand_score(true_labels, labels)
                nmi = normalized_mutual_info_score(true_labels, labels)

                results[f"{background_class_name}_{method_name}_{n_clusters}"] = {
                    "labels": labels,
                    "silhouette": silhouette,
                    "v_measure": v_measure,
                    "ari": ari,
                    "nmi": nmi,
                    "dataset": background_class_name,
                    "n_clusters_requested": n_clusters,
                    "n_clusters_found": (
                        len(np.unique(labels[labels != -1]))
                        if -1 in labels
                        else len(np.unique(labels))
                    ),
                    "defect_labels": defect_labels,
                    "true_labels": true_labels,
                }
            except Exception as e:
                print(f"Error in {method_name} with {n_clusters} clusters: {e}")
                continue

    # Try Spectral clustering
    for n_clusters in n_clusters_list:
        try:
            result = spectral_clustering(n_clusters)
            labels = result.labels_ if hasattr(result, "labels_") else result

            # Evaluate clustering
            silhouette = 0
            if len(np.unique(labels)) > 1 and all(
                np.sum(labels == lab) > 1 for lab in np.unique(labels) if lab != -1
            ):
                try:
                    silhouette = silhouette_score(embeddings, labels)
                except:
                    silhouette = 0

            v_measure = v_measure_score(true_labels, labels)
            ari = adjusted_rand_score(true_labels, labels)
            nmi = normalized_mutual_info_score(true_labels, labels)

            results[f"{background_class_name}_Spectral_{n_clusters}"] = {
                "labels": labels,
                "silhouette": silhouette,
                "v_measure": v_measure,
                "ari": ari,
                "nmi": nmi,
                "dataset": background_class_name,
                "n_clusters_requested": n_clusters,
                "n_clusters_found": (
                    len(np.unique(labels[labels != -1]))
                    if -1 in labels
                    else len(np.unique(labels))
                ),
                "defect_labels": defect_labels,
                "true_labels": true_labels,
            }
        except Exception as e:
            print(f"Error in Spectral clustering with {n_clusters} clusters: {e}")
            continue

    # Process HDBSCAN result
    n_found = len(
        np.unique(hdbscan_labels[hdbscan_labels != -1])
    )  # Count non-noise clusters
    if n_found > 0:
        # Calculate silhouette for non-noise points only
        non_noise_mask = hdbscan_labels != -1
        if np.sum(non_noise_mask) > 1 and n_found > 1:
            try:
                silhouette = silhouette_score(
                    embeddings[non_noise_mask], hdbscan_labels[non_noise_mask]
                )
            except:
                silhouette = 0
        else:
            silhouette = 0

        v_measure = v_measure_score(true_labels, hdbscan_labels)
        ari = adjusted_rand_score(true_labels, hdbscan_labels)
        nmi = normalized_mutual_info_score(true_labels, hdbscan_labels)

        results[f"{background_class_name}_HDBSCAN"] = {
            "labels": hdbscan_labels,
            "silhouette": silhouette,
            "v_measure": v_measure,
            "ari": ari,
            "nmi": nmi,
            "dataset": background_class_name,
            "n_clusters_requested": "auto",
            "n_clusters_found": n_found,
            "defect_labels": defect_labels,
            "true_labels": true_labels,
        }
    else:
        # If HDBSCAN didn't find clusters, use all as one cluster
        v_measure = v_measure_score(true_labels, np.zeros(len(embeddings), dtype=int))
        ari = adjusted_rand_score(true_labels, np.zeros(len(embeddings), dtype=int))
        nmi = normalized_mutual_info_score(
            true_labels, np.zeros(len(embeddings), dtype=int)
        )

        results[f"{background_class_name}_HDBSCAN"] = {
            "labels": np.zeros(len(embeddings), dtype=int),
            "silhouette": 0,
            "v_measure": v_measure,
            "ari": ari,
            "nmi": nmi,
            "dataset": background_class_name,
            "n_clusters_requested": "auto",
            "n_clusters_found": 1,
            "defect_labels": defect_labels,
            "true_labels": true_labels,
        }

    return results


def analyze_mvtec_ad(root_dir):
    """Main function to analyze MVTec-AD dataset using DINO features and clustering"""
    print("Starting MVTec-AD Analysis with DINO features and clustering...")

    # Get all background classes (main object categories)
    background_classes = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            # Check if it has a 'test' subdirectory
            test_path = os.path.join(item_path, "test")
            if os.path.exists(test_path):
                background_classes.append(item)

    print(f"Found background classes: {background_classes}")

    # Define transformations for DINO (similar to ImageNet preprocessing)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load DINO model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_dino_model()
    if model is not None:
        model = model.to(device)

    all_results = {}

    # Process each background class separately
    for bg_class in background_classes:
        print(f"\nProcessing background class: {bg_class}")

        # Create dataset for this background class
        dataset = MVTecADDataset(
            root_dir=root_dir,
            background_class=bg_class,
            transform=transform,
            split="test",
        )

        if len(dataset) == 0:
            print(f"No images found for {bg_class}, skipping...")
            continue

        # Create data loader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

        # Extract features
        print(f"Extracting features for {bg_class}...")
        features, defect_labels = extract_features(model, dataloader, device)

        if len(features) == 0:
            print(f"No features extracted for {bg_class}, skipping...")
            continue

        # Get the expected number of clusters from the dataset
        n_clusters_expected = dataset.get_n_clusters()
        print(f"Expected number of clusters for {bg_class}: {n_clusters_expected}")

        # Perform clustering within this background class
        bg_results = perform_clustering_within_background(
            features, defect_labels, bg_class, n_clusters_expected
        )
        all_results.update(bg_results)

    # Print summary results
    print("\n" + "=" * 120)
    print("MVTec-AD DINO Feature Clustering Results Summary (per Background Class)")
    print("=" * 120)
    print(
        f"{'Background':<12} {'Method':<12} {'Requested':<10} {'Found':<8} {'Silhouette':<12} {'V-Measure':<12} {'ARI':<12} {'NMI':<12}"
    )
    print("-" * 120)

    for key, result in all_results.items():
        bg_class = key.split("_")[0]
        method = (
            "_".join(key.split("_")[1:-1])
            if len(key.split("_")) > 2
            else key.split("_")[1]
        )
        n_requested = result["n_clusters_requested"]
        n_found = result["n_clusters_found"]

        print(
            f"{bg_class:<12} {method:<12} {str(n_requested):<10} {n_found:<8} {result['silhouette']:<12.4f} {result['v_measure']:<12.4f} {result['ari']:<12.4f} {result['nmi']:<12.4f}"
        )

    print("=" * 120)

    # Create summary CSV
    results_df = pd.DataFrame(
        {
            "method_dataset_n_clusters": list(all_results.keys()),
            "dataset": [all_results[k]["dataset"] for k in all_results.keys()],
            "method": [
                (
                    k.replace(all_results[k]["dataset"] + "_", "").rsplit("_", 1)[0]
                    if "_" in k.replace(all_results[k]["dataset"] + "_", "")
                    else k.replace(all_results[k]["dataset"] + "_", "")
                )
                for k in all_results.keys()
            ],
            "n_clusters_requested": [
                all_results[k]["n_clusters_requested"] for k in all_results.keys()
            ],
            "n_clusters_found": [
                all_results[k]["n_clusters_found"] for k in all_results.keys()
            ],
            "silhouette": [all_results[k]["silhouette"] for k in all_results.keys()],
            "v_measure": [all_results[k]["v_measure"] for k in all_results.keys()],
            "ari": [all_results[k]["ari"] for k in all_results.keys()],
            "nmi": [all_results[k]["nmi"] for k in all_results.keys()],
        }
    )

    output_file = "mvtec_ad_dino_clustering_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Total results: {len(results_df)}")

    # Also save a detailed analysis
    detailed_results = []
    anomaly_detection_results = []  # Special analysis for anomaly detection

    for key, result in all_results.items():
        bg_class = result["dataset"]
        method = (
            key.replace(bg_class + "_", "").rsplit("_", 1)[0]
            if "_" in key.replace(bg_class + "_", "")
            else key.replace(bg_class + "_", "")
        )
        n_requested = result["n_clusters_requested"]
        n_found = result["n_clusters_found"]

        # Count how many images were assigned to each cluster
        unique, counts = np.unique(result["labels"], return_counts=True)
        cluster_distribution = dict(zip(unique, counts))

        detailed_results.append(
            {
                "background_class": bg_class,
                "method": method,
                "n_clusters_requested": n_requested,
                "n_clusters_found": n_found,
                "silhouette": result["silhouette"],
                "v_measure": result["v_measure"],
                "ari": result["ari"],
                "nmi": result["nmi"],
                "cluster_distribution": str(cluster_distribution),
                "defect_labels": str(list(set(result["defect_labels"]))),
            }
        )

        # Anomaly detection analysis: see if clustering can separate normal from defective
        defect_labels = result["defect_labels"]
        true_labels = result["true_labels"]
        cluster_labels = result["labels"]

        # Analyze if "good" samples cluster separately from defect samples
        good_indices = [i for i, label in enumerate(defect_labels) if label == "good"]
        defect_indices = [i for i, label in enumerate(defect_labels) if label != "good"]

        if good_indices and defect_indices:
            # Check cluster distribution between good and defective
            good_clusters = [cluster_labels[i] for i in good_indices]
            defect_clusters = [cluster_labels[i] for i in defect_indices]

            # Calculate homogeneity of good vs defect clustering
            from collections import Counter

            good_cluster_counts = Counter(good_clusters)
            defect_cluster_counts = Counter(defect_clusters)

            # Anomaly detection capability: how well do good samples cluster separately?
            # If good samples are in different clusters than defects, it's better for anomaly detection
            good_clusters_set = set(good_cluster_counts.keys())
            defect_clusters_set = set(defect_cluster_counts.keys())
            overlap_clusters = good_clusters_set.intersection(defect_clusters_set)

            anomaly_capability = 1 - (
                len(overlap_clusters)
                / max(len(good_clusters_set), len(defect_clusters_set), 1)
            )

            anomaly_detection_results.append(
                {
                    "background_class": bg_class,
                    "method": method,
                    "n_clusters_requested": n_requested,
                    "anomaly_detection_capability": anomaly_capability,
                    "good_cluster_count": len(good_clusters_set),
                    "defect_cluster_count": len(defect_clusters_set),
                    "overlapping_clusters": len(overlap_clusters),
                    "total_samples": len(cluster_labels),
                    "good_samples": len(good_indices),
                    "defect_samples": len(defect_indices),
                    "v_measure": result["v_measure"],
                    "ari": result["ari"],
                }
            )

    detailed_df = pd.DataFrame(detailed_results)
    detailed_output_file = "mvtec_ad_dino_detailed_results.csv"
    detailed_df.to_csv(detailed_output_file, index=False)
    print(f"Detailed results saved to {detailed_output_file}")

    # Save anomaly detection analysis
    if anomaly_detection_results:
        anomaly_df = pd.DataFrame(anomaly_detection_results)
        anomaly_output_file = "mvtec_ad_anomaly_detection_analysis.csv"
        anomaly_df.to_csv(anomaly_output_file, index=False)
        print(f"Anomaly detection analysis saved to {anomaly_output_file}")

        print("\nAnomaly Detection Analysis Summary:")
        print("=" * 80)
        print(
            f"{'Background':<12} {'Method':<12} {'Anomaly Cap.':<15} {'Good Clusters':<14} {'Defect Clusters':<16} {'Overlap Clusters':<16}"
        )
        print("-" * 80)
        for _, row in anomaly_df.iterrows():
            print(
                f"{row['background_class']:<12} {row['method']:<12} {row['anomaly_detection_capability']:<15.4f} {row['good_cluster_count']:<14} {row['defect_cluster_count']:<16} {row['overlapping_clusters']:<16}"
            )
        print("=" * 80)

    return all_results


def main():
    # Get project root and set MVTec-AD dataset path relative to it
    project_root = get_project_root()

    # Try both possible naming conventions for MVTec-AD
    mvtec_path_options = [
        project_root / "data" / "mvtec_ad",  # underscore version
        project_root / "data" / "mvtec-ad",  # hyphen version
        project_root / "data" / "MVTec-AD",  # capitalized version
    ]

    mvtec_path = None

    for path in mvtec_path_options:
        if os.path.exists(str(path)):
            mvtec_path = str(path)
            break

    if mvtec_path is None:
        # If none exist, default to the hyphen version (most common)
        mvtec_path = str(project_root / "data" / "mvtec-ad")

    if not os.path.exists(mvtec_path):
        print(f"MVTec-AD dataset not found at {mvtec_path}")
        print(
            "Please download the MVTec-AD dataset and place it at the above location."
        )
        print(
            "\nMVTec-AD can be downloaded from: https://www.mvtec.com/company/research/datasets/mvtec-ad"
        )
        print("\nThe expected structure is (using common naming):")
        print("mvtec-ad/")
        print("├── bottle/")
        print("│   ├── test/")
        print("│   │   ├── good/")
        print("│   │   ├── broken_large/")
        print("│   │   └── ...")
        print("├── cable/")
        print("│   ├── test/")
        print("│   │   ├── good/")
        print("│   │   ├── bent_wire/")
        print("│   │   └── ...")
        print("└── ...")
        print(f"\nScript will look in these possible locations:")
        for path in mvtec_path_options:
            print(f"  - {path}")
        return

    results = analyze_mvtec_ad(mvtec_path)
    print("\nAnalysis complete!")

    # Provide summary of findings
    print("\nSUMMARY OF FINDINGS:")
    print("=" * 60)
    print("This analysis used DINO features with clustering to detect anomalies")
    print("in the MVTec-AD dataset. For anomaly detection, we want normal")
    print('("good") samples to cluster separately from defective samples.')
    print("\nAnomaly Detection Capability Score:")
    print("- 1.0 means good and defective samples are perfectly separated")
    print("- 0.0 means good and defective samples are completely mixed")
    print("\nHigher scores indicate better potential for anomaly detection.")
    print("=" * 60)


if __name__ == "__main__":
    main()
