"""
Clustering analysis on MAE ViT embeddings using k-means, VBGMM, and HDBSCAN
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import hydra
import lancedb
from omegaconf import DictConfig, OmegaConf


class TeeLogger:
    """Logger that writes to both file and console"""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# Clustering algorithms
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture

try:
    import hdbscan
except ImportError:
    print("Warning: hdbscan not installed. Install with: pip install hdbscan")
    hdbscan = None

# Evaluation metrics
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
except ImportError:
    plt = None
    sns = None

try:
    import umap
except ImportError:
    umap = None

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 10,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, KMeans]:
    """Run k-means clustering

    Args:
        metric: Distance metric. 'euclidean' or 'cosine'
    """
    # For cosine similarity, normalize embeddings (cosine distance = 1 - cosine similarity)
    # K-means with normalized vectors = cosine distance clustering
    if metric == "cosine":
        print(f"  Running k-means with n_clusters={n_clusters} (cosine distance)")
        from sklearn.preprocessing import normalize

        embeddings_normalized = normalize(embeddings, norm="l2")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
            verbose=0,
        )
        labels = kmeans.fit_predict(embeddings_normalized)
    else:
        print(f"  Running k-means with n_clusters={n_clusters} (euclidean distance)")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
            verbose=0,
        )
        labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def run_vbgmm(
    embeddings: np.ndarray,
    n_components: int,
    random_state: int = 42,
    covariance_type: str = "diag",
    max_iter: int = 200,
    n_init: int = 1,
) -> Tuple[np.ndarray, BayesianGaussianMixture]:
    """Run Variational Bayesian Gaussian Mixture Model"""
    print(f"  Running VBGMM with n_components={n_components}")
    vbgmm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
        verbose=0,
    )
    vbgmm.fit(embeddings)
    labels = vbgmm.predict(embeddings)
    return labels, vbgmm


def run_hdbscan_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    metric: str = "euclidean",
) -> Tuple[np.ndarray, Any]:
    """Run HDBSCAN clustering

    Args:
        metric: Distance metric. 'euclidean' or 'cosine'
    """
    if hdbscan is None:
        raise ImportError("hdbscan not installed")

    print(
        f"  Running HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples} ({metric} distance)"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings)
    return labels, clusterer


def evaluate_clustering(
    embeddings: np.ndarray, labels: np.ndarray, true_labels: np.ndarray = None
) -> Dict[str, float]:
    """Evaluate clustering results"""
    metrics = {}

    # Filter out noise points for HDBSCAN (label -1)
    valid_mask = labels != -1
    n_noise = np.sum(~valid_mask)
    n_clusters = len(np.unique(labels[valid_mask]))

    metrics["n_clusters"] = n_clusters
    metrics["n_noise_points"] = int(n_noise)
    metrics["noise_ratio"] = float(n_noise / len(labels))

    # Need at least 2 clusters for most metrics
    if n_clusters < 2:
        print("    Warning: Less than 2 clusters found, skipping some metrics")
        return metrics

    # Use only valid points for internal metrics
    valid_embeddings = embeddings[valid_mask]
    valid_labels = labels[valid_mask]

    # Internal metrics (don't require true labels)
    try:
        metrics["silhouette"] = float(silhouette_score(valid_embeddings, valid_labels))
    except Exception as e:
        print(f"    Warning: Could not compute silhouette score: {e}")
        metrics["silhouette"] = None

    try:
        metrics["calinski_harabasz"] = float(
            calinski_harabasz_score(valid_embeddings, valid_labels)
        )
    except Exception as e:
        print(f"    Warning: Could not compute Calinski-Harabasz score: {e}")
        metrics["calinski_harabasz"] = None

    try:
        metrics["davies_bouldin"] = float(
            davies_bouldin_score(valid_embeddings, valid_labels)
        )
    except Exception as e:
        print(f"    Warning: Could not compute Davies-Bouldin score: {e}")
        metrics["davies_bouldin"] = None

    # External metrics (require true labels)
    if true_labels is not None:
        valid_true_labels = true_labels[valid_mask]

        try:
            metrics["adjusted_rand_index"] = float(
                adjusted_rand_score(valid_true_labels, valid_labels)
            )
        except Exception as e:
            print(f"    Warning: Could not compute ARI: {e}")
            metrics["adjusted_rand_index"] = None

        try:
            metrics["normalized_mutual_info"] = float(
                normalized_mutual_info_score(valid_true_labels, valid_labels)
            )
        except Exception as e:
            print(f"    Warning: Could not compute NMI: {e}")
            metrics["normalized_mutual_info"] = None

        try:
            metrics["v_measure"] = float(
                v_measure_score(valid_true_labels, valid_labels)
            )
        except Exception as e:
            print(f"    Warning: Could not compute V-measure: {e}")
            metrics["v_measure"] = None

    return metrics


def visualize_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    max_samples: int = 5000,
    output_path: str = None,
    title: str = "Cluster Visualization",
):
    """Visualize clusters in 2D using dimensionality reduction"""
    if plt is None:
        print("Warning: matplotlib not available, skipping visualization")
        return

    # Subsample if too many points
    if len(embeddings) > max_samples:
        print(f"  Subsampling {max_samples} points for visualization")
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Dimensionality reduction
    print(f"  Reducing dimensions using {method}")
    if method == "umap" and umap is not None:
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
    else:  # PCA
        reducer = PCA(n_components=n_components, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Separate noise points (label -1) from clusters
    noise_mask = labels == -1
    cluster_mask = ~noise_mask

    # Plot clusters
    if np.any(cluster_mask):
        scatter = ax.scatter(
            embedding_2d[cluster_mask, 0],
            embedding_2d[cluster_mask, 1],
            c=labels[cluster_mask],
            cmap="tab20",
            s=5,
            alpha=0.6,
            edgecolors="none",
        )
        plt.colorbar(scatter, ax=ax, label="Cluster ID")

    # Plot noise points in gray
    if np.any(noise_mask):
        ax.scatter(
            embedding_2d[noise_mask, 0],
            embedding_2d[noise_mask, 1],
            c="gray",
            s=3,
            alpha=0.3,
            label="Noise",
            edgecolors="none",
        )
        ax.legend()

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title(title)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visualization to {output_path}")

    plt.close()


@hydra.main(version_base=None, config_path="configs", config_name="clustering_config")
def main(cfg: DictConfig):
    # Create output directory first
    output_dir = Path(cfg.output.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to both file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"clustering_log_{timestamp}.txt"
    tee = TeeLogger(log_file)
    sys.stdout = tee

    print("=" * 80)
    print("MAE ViT Embedding Clustering Analysis")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))

    # Load embeddings
    embeddings, metadata = load_embeddings_from_lancedb(
        cfg.database.path, cfg.database.table_name
    )

    # Get true labels if available
    true_labels = None
    if cfg.evaluation.use_true_labels and "class_id" in metadata.columns:
        true_labels = metadata["class_id"].values
        print(
            f"Using true labels for evaluation ({len(np.unique(true_labels))} classes)"
        )

    # Store all results
    all_results = []

    # Run k-means clustering
    if cfg.methods.kmeans.enabled:
        print("\n" + "=" * 80)
        print("Running k-means clustering")
        print("=" * 80)

        for n_clusters in cfg.methods.kmeans.n_clusters:
            print(f"\nk-means with k={n_clusters}")

            start_time = time.time()
            labels, model = run_kmeans(
                embeddings,
                n_clusters=n_clusters,
                random_state=cfg.methods.kmeans.random_state,
                max_iter=cfg.methods.kmeans.max_iter,
                n_init=cfg.methods.kmeans.n_init,
                metric=cfg.methods.kmeans.get("metric", "euclidean"),
            )
            elapsed_time = time.time() - start_time

            # Evaluate
            metrics = evaluate_clustering(embeddings, labels, true_labels)
            metrics["method"] = "kmeans"
            metrics["n_clusters_param"] = n_clusters
            metrics["distance_metric"] = cfg.methods.kmeans.get("metric", "euclidean")
            metrics["clustering_time_seconds"] = float(elapsed_time)

            print("  Results:")
            print(f"    Clustering Time: {elapsed_time:.2f} seconds")
            print("    Internal Metrics:")
            print(f"      Silhouette Score: {metrics.get('silhouette', 'N/A')}")
            print(
                f"      Calinski-Harabasz Score: {metrics.get('calinski_harabasz', 'N/A')}"
            )
            print(f"      Davies-Bouldin Score: {metrics.get('davies_bouldin', 'N/A')}")
            if true_labels is not None:
                print("    External Metrics (vs Ground Truth):")
                print(
                    f"      Adjusted Rand Index: {metrics.get('adjusted_rand_index', 'N/A')}"
                )
                print(
                    f"      Normalized Mutual Info: {metrics.get('normalized_mutual_info', 'N/A')}"
                )
                print(f"      V-Measure: {metrics.get('v_measure', 'N/A')}")
            all_results.append(metrics)

            # Save predictions
            if cfg.output.save_predictions:
                pred_df = metadata.copy()
                pred_df["cluster_label"] = labels
                output_file = output_dir / f"kmeans_k{n_clusters}_predictions.csv"
                pred_df.to_csv(output_file, index=False)
                print(f"  Saved predictions to {output_file}")

            # Visualize
            if cfg.visualization.enabled and cfg.visualization.save_plots:
                vis_file = output_dir / f"kmeans_k{n_clusters}_visualization.png"
                visualize_clusters(
                    embeddings,
                    labels,
                    method=cfg.visualization.method,
                    max_samples=cfg.visualization.max_samples_plot,
                    output_path=vis_file,
                    title=f"k-means (k={n_clusters})",
                )

    # Run VBGMM clustering
    if cfg.methods.vbgmm.enabled:
        print("\n" + "=" * 80)
        print("Running Variational Bayesian GMM clustering")
        print("=" * 80)

        for n_components in cfg.methods.vbgmm.n_components:
            print(f"\nVBGMM with n_components={n_components}")

            start_time = time.time()
            labels, model = run_vbgmm(
                embeddings,
                n_components=n_components,
                random_state=cfg.methods.vbgmm.random_state,
                covariance_type=cfg.methods.vbgmm.covariance_type,
                max_iter=cfg.methods.vbgmm.max_iter,
                n_init=cfg.methods.vbgmm.n_init,
            )
            elapsed_time = time.time() - start_time

            # Evaluate
            metrics = evaluate_clustering(embeddings, labels, true_labels)
            metrics["method"] = "vbgmm"
            metrics["n_components_param"] = n_components
            metrics["covariance_type"] = cfg.methods.vbgmm.covariance_type
            metrics["clustering_time_seconds"] = float(elapsed_time)

            print("  Results:")
            print(f"    Clustering Time: {elapsed_time:.2f} seconds")
            print("    Internal Metrics:")
            print(f"      Silhouette Score: {metrics.get('silhouette', 'N/A')}")
            print(
                f"      Calinski-Harabasz Score: {metrics.get('calinski_harabasz', 'N/A')}"
            )
            print(f"      Davies-Bouldin Score: {metrics.get('davies_bouldin', 'N/A')}")
            if true_labels is not None:
                print("    External Metrics (vs Ground Truth):")
                print(
                    f"      Adjusted Rand Index: {metrics.get('adjusted_rand_index', 'N/A')}"
                )
                print(
                    f"      Normalized Mutual Info: {metrics.get('normalized_mutual_info', 'N/A')}"
                )
                print(f"      V-Measure: {metrics.get('v_measure', 'N/A')}")
            all_results.append(metrics)

            # Save predictions
            if cfg.output.save_predictions:
                pred_df = metadata.copy()
                pred_df["cluster_label"] = labels
                output_file = output_dir / f"vbgmm_n{n_components}_predictions.csv"
                pred_df.to_csv(output_file, index=False)
                print(f"  Saved predictions to {output_file}")

            # Visualize
            if cfg.visualization.enabled and cfg.visualization.save_plots:
                vis_file = output_dir / f"vbgmm_n{n_components}_visualization.png"
                visualize_clusters(
                    embeddings,
                    labels,
                    method=cfg.visualization.method,
                    max_samples=cfg.visualization.max_samples_plot,
                    output_path=vis_file,
                    title=f"VBGMM (n={n_components})",
                )

    # Run HDBSCAN clustering
    if cfg.methods.hdbscan.enabled:
        print("\n" + "=" * 80)
        print("Running HDBSCAN clustering")
        print("=" * 80)

        if hdbscan is None:
            print("Warning: hdbscan not installed, skipping")
        else:
            for min_cluster_size in cfg.methods.hdbscan.min_cluster_size:
                for min_samples in cfg.methods.hdbscan.min_samples:
                    print(
                        f"\nHDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}"
                    )

                    start_time = time.time()
                    labels, model = run_hdbscan_clustering(
                        embeddings,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=cfg.methods.hdbscan.cluster_selection_epsilon,
                        cluster_selection_method=cfg.methods.hdbscan.cluster_selection_method,
                        metric=cfg.methods.hdbscan.get("metric", "euclidean"),
                    )
                    elapsed_time = time.time() - start_time

                    # Evaluate
                    metrics = evaluate_clustering(embeddings, labels, true_labels)
                    metrics["method"] = "hdbscan"
                    metrics["min_cluster_size_param"] = min_cluster_size
                    metrics["min_samples_param"] = min_samples
                    metrics["distance_metric"] = cfg.methods.hdbscan.get(
                        "metric", "euclidean"
                    )
                    metrics["clustering_time_seconds"] = float(elapsed_time)

                    print("  Results:")
                    print(f"    Clustering Time: {elapsed_time:.2f} seconds")
                    print(f"    Clusters found: {metrics.get('n_clusters', 0)}")
                    print(
                        f"    Noise points: {metrics.get('n_noise_points', 0)} ({metrics.get('noise_ratio', 0) * 100:.1f}%)"
                    )
                    print("    Internal Metrics:")
                    print(f"      Silhouette Score: {metrics.get('silhouette', 'N/A')}")
                    print(
                        f"      Calinski-Harabasz Score: {metrics.get('calinski_harabasz', 'N/A')}"
                    )
                    print(
                        f"      Davies-Bouldin Score: {metrics.get('davies_bouldin', 'N/A')}"
                    )
                    if true_labels is not None:
                        print("    External Metrics (vs Ground Truth):")
                        print(
                            f"      Adjusted Rand Index: {metrics.get('adjusted_rand_index', 'N/A')}"
                        )
                        print(
                            f"      Normalized Mutual Info: {metrics.get('normalized_mutual_info', 'N/A')}"
                        )
                        print(f"      V-Measure: {metrics.get('v_measure', 'N/A')}")
                    all_results.append(metrics)

                    # Save predictions
                    if cfg.output.save_predictions:
                        pred_df = metadata.copy()
                        pred_df["cluster_label"] = labels
                        output_file = (
                            output_dir
                            / f"hdbscan_mcs{min_cluster_size}_ms{min_samples}_predictions.csv"
                        )
                        pred_df.to_csv(output_file, index=False)
                        print(f"  Saved predictions to {output_file}")

                    # Visualize
                    if cfg.visualization.enabled and cfg.visualization.save_plots:
                        vis_file = (
                            output_dir
                            / f"hdbscan_mcs{min_cluster_size}_ms{min_samples}_visualization.png"
                        )
                        visualize_clusters(
                            embeddings,
                            labels,
                            method=cfg.visualization.method,
                            max_samples=cfg.visualization.max_samples_plot,
                            output_path=vis_file,
                            title=f"HDBSCAN (mcs={min_cluster_size}, ms={min_samples})",
                        )

    # Save summary of all results
    if cfg.output.save_metrics:
        results_df = pd.DataFrame(all_results)
        summary_file = output_dir / "clustering_summary.csv"
        results_df.to_csv(summary_file, index=False)
        print(f"\n{'=' * 80}")
        print(f"Saved clustering summary to {summary_file}")

        # Print summary table
        print(f"\n{'=' * 80}")
        print("Clustering Results Summary")
        print(f"{'=' * 80}")
        print(results_df.to_string(index=False))

        # Save detailed JSON
        json_file = output_dir / "clustering_summary.json"
        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved detailed results to {json_file}")

    print(f"\n{'=' * 80}")
    print("Clustering analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Log saved to: {log_file}")
    print(f"{'=' * 80}")

    # Close logger
    tee.close()
    sys.stdout = tee.terminal


if __name__ == "__main__":
    main()
