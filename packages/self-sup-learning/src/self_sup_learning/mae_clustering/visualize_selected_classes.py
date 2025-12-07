"""
Visualize selected ground truth classes using UMAP dimension reduction
"""

import sys
from pathlib import Path
from typing import List, Tuple

import lancedb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import OPTICS, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

try:
    import hdbscan
except ImportError:
    print("Warning: hdbscan not installed. Install with: pip install hdbscan")
    hdbscan = None

try:
    import umap
except ImportError:
    print("Error: umap-learn not installed. Install with: pip install umap-learn")
    sys.exit(1)

sns.set_style("whitegrid")


def load_embeddings_from_lancedb(
    db_path: str, table_name: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata from LanceDB"""
    print(f"Loading embeddings from {db_path}/{table_name}")
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    # Convert to pandas DataFrame
    df = table.to_pandas()

    # Extract embeddings and metadata
    embeddings = np.array(df["vector"].tolist())
    metadata = df[["filename", "class_id", "class", "path"]].copy()

    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    return embeddings, metadata


def select_random_classes(
    metadata: pd.DataFrame, n_classes: int = 10, random_state: int = 42
) -> List[int]:
    """Select random classes from the dataset"""
    unique_classes = metadata["class_id"].unique()
    np.random.seed(random_state)
    selected_classes = np.random.choice(
        unique_classes, size=min(n_classes, len(unique_classes)), replace=False
    )
    return sorted(selected_classes.tolist())


def filter_data_by_classes(
    embeddings: np.ndarray, metadata: pd.DataFrame, selected_classes: List[int]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Filter embeddings and metadata by selected classes"""
    mask = metadata["class_id"].isin(selected_classes)
    filtered_embeddings = embeddings[mask]
    filtered_metadata = metadata[mask].copy().reset_index(drop=True)
    return filtered_embeddings, filtered_metadata


def run_clustering_methods(embeddings: np.ndarray, n_clusters: int = 10):
    """Run multiple clustering methods: K-means, Spectral Clustering, HDBSCAN, OPTICS"""
    print("\nRunning unsupervised clustering methods...")
    results = {}

    # K-means with Euclidean distance
    print("  K-means (Euclidean)...")
    kmeans_l2 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    results["kmeans_euclidean"] = kmeans_l2.fit_predict(embeddings)

    # K-means with Cosine distance (normalize first)
    print("  K-means (Cosine)...")
    embeddings_normalized = normalize(embeddings, norm="l2")
    kmeans_cosine = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300
    )
    results["kmeans_cosine"] = kmeans_cosine.fit_predict(embeddings_normalized)

    # Spectral Clustering with RBF kernel
    print("  Spectral Clustering (RBF)...")
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="rbf",
        gamma=1.0,
        random_state=42,
        n_init=10,
        assign_labels="kmeans",
    )
    results["spectral_rbf"] = spectral.fit_predict(embeddings)

    # Spectral Clustering with nearest neighbors
    print("  Spectral Clustering (Nearest Neighbors)...")
    spectral_nn = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=10,
        random_state=42,
        n_init=10,
        assign_labels="kmeans",
    )
    results["spectral_nn"] = spectral_nn.fit_predict(embeddings)

    # HDBSCAN
    if hdbscan is not None:
        print("  HDBSCAN...")
        hdbscan_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=30,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        results["hdbscan"] = hdbscan_clusterer.fit_predict(embeddings)
    else:
        print("  HDBSCAN... skipped (not installed)")

    # OPTICS
    print("  OPTICS...")
    optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1, metric="euclidean")
    results["optics"] = optics.fit_predict(embeddings)

    return results


def plot_umap_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    output_path: str,
    title: str = None,
):
    """Apply UMAP and plot the 2D projection"""
    print(f"  Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")

    # Apply UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
        verbose=False,
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get unique classes for color mapping
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    # Plot each class
    for idx, class_id in enumerate(unique_labels):
        mask = labels == class_id
        class_name = class_names[mask][0]  # Get class name
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[idx]],
            label=f"{class_name} (ID: {class_id})",
            s=30,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
        )

    ax.set_xlabel("UMAP Component 1", fontsize=12)
    ax.set_ylabel("UMAP Component 2", fontsize=12)

    if title is None:
        title = f"UMAP Projection (n_neighbors={n_neighbors}, min_dist={min_dist})"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Place legend outside plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved plot to {output_path}")
    plt.close()


def main():
    # Configuration
    DB_PATH = "/home/jin/Desktop/mm/lancedb"
    TABLE_NAME = "imagenet_mae_embeddings"
    OUTPUT_DIR = Path("/home/jin/Desktop/mm/playground/clustering/umap_visualizations")
    N_CLASSES = 10
    RANDOM_STATE = 42

    # UMAP parameter variations
    umap_configs = [
        {"n_neighbors": 15, "min_dist": 0.1},  # Default UMAP settings
        {"n_neighbors": 5, "min_dist": 0.1},  # Smaller neighborhood (local structure)
        {"n_neighbors": 30, "min_dist": 0.1},  # Larger neighborhood (global structure)
        {"n_neighbors": 15, "min_dist": 0.01},  # Tighter clusters
        {"n_neighbors": 15, "min_dist": 0.5},  # Looser clusters
        {"n_neighbors": 50, "min_dist": 0.1},  # Very large neighborhood
        {"n_neighbors": 5, "min_dist": 0.01},  # Local + tight
        {"n_neighbors": 30, "min_dist": 0.5},  # Global + loose
    ]

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("UMAP Visualization of Selected Classes")
    print("=" * 80)

    # Load data
    embeddings, metadata = load_embeddings_from_lancedb(DB_PATH, TABLE_NAME)

    # Select random classes
    print(f"\nSelecting {N_CLASSES} random classes...")
    selected_classes = select_random_classes(
        metadata, n_classes=N_CLASSES, random_state=RANDOM_STATE
    )
    print(f"Selected class IDs: {selected_classes}")

    # Filter data
    filtered_embeddings, filtered_metadata = filter_data_by_classes(
        embeddings, metadata, selected_classes
    )

    print("\nFiltered dataset:")
    print(f"  Total samples: {len(filtered_embeddings)}")
    print(f"  Classes: {N_CLASSES}")
    print("  Samples per class:")
    for class_id in selected_classes:
        class_name = filtered_metadata[filtered_metadata["class_id"] == class_id][
            "class"
        ].iloc[0]
        count = (filtered_metadata["class_id"] == class_id).sum()
        print(f"    {class_name} (ID: {class_id}): {count} samples")

    # Get labels and class names
    labels = filtered_metadata["class_id"].values
    class_names = filtered_metadata["class"].values

    # Run unsupervised clustering methods
    clustering_results = run_clustering_methods(
        filtered_embeddings, n_clusters=N_CLASSES
    )

    # Calculate clustering metrics vs ground truth
    print("\nClustering performance vs ground truth:")
    for method_name, cluster_labels in clustering_results.items():
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        print(f"  {method_name:20s} - ARI: {ari:.4f}, NMI: {nmi:.4f}")

    # Generate visualizations with different UMAP parameters
    print(f"\n{'=' * 80}")
    print("Generating UMAP visualizations with different parameters")
    print(f"{'=' * 80}\n")

    for idx, config in enumerate(umap_configs, 1):
        n_neighbors = config["n_neighbors"]
        min_dist = config["min_dist"]

        print(f"Configuration {idx}/{len(umap_configs)}:")
        output_file = OUTPUT_DIR / f"umap_n{n_neighbors}_md{min_dist:.2f}.png"

        plot_umap_projection(
            filtered_embeddings,
            labels,
            class_names,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            output_path=output_file,
        )

    # Create a summary plot with 4 key configurations
    print(f"\n{'=' * 80}")
    print("Creating summary comparison plot")
    print(f"{'=' * 80}\n")

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    summary_configs = [
        {"n_neighbors": 15, "min_dist": 0.1, "title": "Default (n=15, d=0.1)"},
        {"n_neighbors": 5, "min_dist": 0.1, "title": "Local Structure (n=5, d=0.1)"},
        {"n_neighbors": 50, "min_dist": 0.1, "title": "Global Structure (n=50, d=0.1)"},
        {"n_neighbors": 15, "min_dist": 0.01, "title": "Tight Clusters (n=15, d=0.01)"},
    ]

    for idx, (ax, config) in enumerate(zip(axes, summary_configs)):
        n_neighbors = config["n_neighbors"]
        min_dist = config["min_dist"]

        print(f"  Generating subplot {idx + 1}/4: {config['title']}")

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
            verbose=False,
        )
        embedding_2d = reducer.fit_transform(filtered_embeddings)

        # Plot
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for label_idx, class_id in enumerate(unique_labels):
            mask = labels == class_id
            class_name = class_names[mask][0]
            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[colors[label_idx]],
                label=f"{class_name} ({class_id})",
                s=20,
                alpha=0.7,
                edgecolors="k",
                linewidths=0.3,
            )

        ax.set_xlabel("UMAP Component 1", fontsize=10)
        ax.set_ylabel("UMAP Component 2", fontsize=10)
        ax.set_title(config["title"], fontsize=12, fontweight="bold")

        if idx == 1:  # Only show legend on one subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.suptitle(
        f"UMAP Parameter Comparison ({N_CLASSES} Classes)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    summary_file = OUTPUT_DIR / "umap_comparison_summary.png"
    plt.savefig(summary_file, dpi=300, bbox_inches="tight")
    print(f"\n  Saved summary comparison to {summary_file}")
    plt.close()

    # Create comprehensive clustering comparison plot (Ground Truth + Unsupervised Methods)
    print(f"\n{'=' * 80}")
    print("Creating clustering methods comparison plot")
    print(f"{'=' * 80}\n")

    # Use default UMAP parameters for comparison
    umap_reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, verbose=False
    )
    embedding_2d = umap_reducer.fit_transform(filtered_embeddings)

    # Prepare all label sets
    all_labels = {
        "Ground Truth": labels,
        "K-Means (Euclidean)": clustering_results["kmeans_euclidean"],
        "K-Means (Cosine)": clustering_results["kmeans_cosine"],
        "Spectral (RBF)": clustering_results["spectral_rbf"],
        "Spectral (NN)": clustering_results["spectral_nn"],
    }

    # Add HDBSCAN if available
    if "hdbscan" in clustering_results:
        all_labels["HDBSCAN"] = clustering_results["hdbscan"]

    # Add OPTICS if it found clusters
    if "optics" in clustering_results:
        all_labels["OPTICS"] = clustering_results["optics"]

    # Create figure with appropriate number of subplots
    n_methods = len(all_labels)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 8 * n_rows))
    axes = axes.flatten()

    for idx, (method_name, method_labels) in enumerate(all_labels.items()):
        ax = axes[idx]
        print(f"  Plotting {method_name}...")

        unique_method_labels = np.unique(method_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_method_labels)))

        for label_idx, cluster_id in enumerate(unique_method_labels):
            mask = method_labels == cluster_id

            # For ground truth, show class names
            if method_name == "Ground Truth":
                class_name = class_names[mask][0]
                label_text = f"{class_name} ({cluster_id})"
            else:
                label_text = f"Cluster {cluster_id}"

            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[colors[label_idx]],
                label=label_text if method_name == "Ground Truth" else None,
                s=25,
                alpha=0.7,
                edgecolors="k",
                linewidths=0.3,
            )

        ax.set_xlabel("UMAP Component 1", fontsize=11)
        ax.set_ylabel("UMAP Component 2", fontsize=11)

        # Add metrics to title for clustering methods
        if method_name != "Ground Truth":
            # Map display name to result key
            method_mapping = {
                "K-Means (Euclidean)": "kmeans_euclidean",
                "K-Means (Cosine)": "kmeans_cosine",
                "Spectral (RBF)": "spectral_rbf",
                "Spectral (NN)": "spectral_nn",
                "HDBSCAN": "hdbscan",
                "OPTICS": "optics",
            }

            if method_name in method_mapping:
                key = method_mapping[method_name]
                ari = adjusted_rand_score(labels, clustering_results[key])
                nmi = normalized_mutual_info_score(labels, clustering_results[key])
                n_clusters_found = len(
                    np.unique(clustering_results[key][clustering_results[key] >= 0])
                )
                n_noise = np.sum(clustering_results[key] == -1)

                if n_noise > 0:
                    title_text = f"{method_name}\nARI: {ari:.4f}, NMI: {nmi:.4f}\nClusters: {n_clusters_found}, Noise: {n_noise}"
                else:
                    title_text = f"{method_name}\nARI: {ari:.4f}, NMI: {nmi:.4f}"

                ax.set_title(title_text, fontsize=12, fontweight="bold")
        else:
            ax.set_title(method_name, fontsize=12, fontweight="bold")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # Hide unused subplots
    for idx in range(len(all_labels), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Clustering Methods Comparison - UMAP Projection ({N_CLASSES} Classes)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    clustering_comparison_file = OUTPUT_DIR / "clustering_methods_comparison.png"
    plt.savefig(clustering_comparison_file, dpi=300, bbox_inches="tight")
    print(f"\n  Saved clustering comparison to {clustering_comparison_file}")
    plt.close()

    print(f"\n{'=' * 80}")
    print("Visualization complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"  - {len(umap_configs)} individual UMAP parameter plots")
    print("  - 1 UMAP parameter comparison plot")
    print("  - 1 clustering methods comparison plot")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
