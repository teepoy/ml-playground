"""
Quick test script to verify clustering pipeline works with a small subset
"""

import time
from pathlib import Path

import lancedb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed")

from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def test_clustering():
    print("=" * 80)
    print("Quick Clustering Test")
    print("=" * 80)

    # Load embeddings from LanceDB
    db_path = "/home/jin/Desktop/mm/lancedb"
    table_name = "imagenet_mae_embeddings"

    print(f"\nLoading embeddings from {db_path}/{table_name}")
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    df = table.to_pandas()

    # Use subset for quick test
    n_samples = min(5000, len(df))
    print(f"Using {n_samples} samples for testing")
    df_subset = df.sample(n=n_samples, random_state=42)

    embeddings = np.array(df_subset["vector"].tolist())
    true_labels = df_subset["class_id"].values

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of true classes: {len(np.unique(true_labels))}")

    # Create output directory
    output_dir = Path("./clustering_test_results")
    output_dir.mkdir(exist_ok=True)

    results = []

    # Test 1: k-means with k=10
    print("\n" + "-" * 80)
    print("Test 1: k-means (k=10)")
    print("-" * 80)
    start_time = time.time()
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    elapsed_time = time.time() - start_time

    print(f"  Clustering Time: {elapsed_time:.2f} seconds")
    silhouette = silhouette_score(embeddings, labels)
    calinski = calinski_harabasz_score(embeddings, labels)
    davies = davies_bouldin_score(embeddings, labels)
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)

    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz Score: {calinski:.2f}")
    print(f"  Davies-Bouldin Score: {davies:.4f}")
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  Normalized Mutual Info: {nmi:.4f}")

    results.append(
        {
            "method": "kmeans",
            "params": "k=10",
            "n_clusters": 10,
            "clustering_time_seconds": elapsed_time,
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies,
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
        }
    )

    # Save predictions
    pred_df = df_subset[["filename", "class_id", "class"]].copy()
    pred_df["cluster_label"] = labels
    pred_df.to_csv(output_dir / "kmeans_k10_test.csv", index=False)
    print(f"  Saved to {output_dir / 'kmeans_k10_test.csv'}")

    # Test 2: VBGMM with n=10
    print("\n" + "-" * 80)
    print("Test 2: VBGMM (n_components=10)")
    print("-" * 80)
    start_time = time.time()
    vbgmm = BayesianGaussianMixture(
        n_components=10, random_state=42, covariance_type="diag"
    )
    vbgmm.fit(embeddings)
    labels = vbgmm.predict(embeddings)
    elapsed_time = time.time() - start_time

    n_clusters = len(np.unique(labels))
    print(f"  Clustering Time: {elapsed_time:.2f} seconds")
    print(f"  Number of clusters found: {n_clusters}")

    if n_clusters >= 2:
        silhouette = silhouette_score(embeddings, labels)
        calinski = calinski_harabasz_score(embeddings, labels)
        davies = davies_bouldin_score(embeddings, labels)
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)

        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski:.2f}")
        print(f"  Davies-Bouldin Score: {davies:.4f}")
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"  Normalized Mutual Info: {nmi:.4f}")

        results.append(
            {
                "method": "vbgmm",
                "params": "n=10",
                "n_clusters": n_clusters,
                "clustering_time_seconds": elapsed_time,
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies,
                "adjusted_rand_index": ari,
                "normalized_mutual_info": nmi,
            }
        )
    else:
        print(f"  Warning: Only {n_clusters} cluster(s) found, skipping metrics")
        results.append(
            {
                "method": "vbgmm",
                "params": "n=10",
                "n_clusters": n_clusters,
                "clustering_time_seconds": elapsed_time,
                "silhouette": None,
                "calinski_harabasz": None,
                "davies_bouldin": None,
                "adjusted_rand_index": None,
                "normalized_mutual_info": None,
            }
        )

    pred_df = df_subset[["filename", "class_id", "class"]].copy()
    pred_df["cluster_label"] = labels
    pred_df.to_csv(output_dir / "vbgmm_n10_test.csv", index=False)
    print(f"  Saved to {output_dir / 'vbgmm_n10_test.csv'}")

    # Test 3: HDBSCAN
    if HDBSCAN_AVAILABLE:
        print("\n" + "-" * 80)
        print("Test 3: HDBSCAN (min_cluster_size=50)")
        print("-" * 80)
        start_time = time.time()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
        labels = clusterer.fit_predict(embeddings)
        elapsed_time = time.time() - start_time

        n_clusters = len(np.unique(labels[labels != -1]))
        print(f"  Clustering Time: {elapsed_time:.2f} seconds")
        n_noise = np.sum(labels == -1)

        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points: {n_noise} ({100 * n_noise / len(labels):.1f}%)")

        # Compute metrics only on non-noise points
        if n_clusters >= 2:
            valid_mask = labels != -1
            if np.sum(valid_mask) > 0:
                silhouette = silhouette_score(
                    embeddings[valid_mask], labels[valid_mask]
                )
                calinski = calinski_harabasz_score(
                    embeddings[valid_mask], labels[valid_mask]
                )
                davies = davies_bouldin_score(
                    embeddings[valid_mask], labels[valid_mask]
                )
                ari = adjusted_rand_score(true_labels[valid_mask], labels[valid_mask])
                nmi = normalized_mutual_info_score(
                    true_labels[valid_mask], labels[valid_mask]
                )

                print(f"  Silhouette Score: {silhouette:.4f}")
                print(f"  Calinski-Harabasz Score: {calinski:.2f}")
                print(f"  Davies-Bouldin Score: {davies:.4f}")
                print(f"  Adjusted Rand Index: {ari:.4f}")
                print(f"  Normalized Mutual Info: {nmi:.4f}")

                results.append(
                    {
                        "method": "hdbscan",
                        "params": "mcs=50, ms=10",
                        "n_clusters": n_clusters,
                        "n_noise": n_noise,
                        "clustering_time_seconds": elapsed_time,
                        "silhouette": silhouette,
                        "calinski_harabasz": calinski,
                        "davies_bouldin": davies,
                        "adjusted_rand_index": ari,
                        "normalized_mutual_info": nmi,
                    }
                )

        pred_df = df_subset[["filename", "class_id", "class"]].copy()
        pred_df["cluster_label"] = labels
        pred_df.to_csv(output_dir / "hdbscan_test.csv", index=False)
        print(f"  Saved to {output_dir / 'hdbscan_test.csv'}")

    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "test_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {output_dir}")
    print("=" * 80)

    print("\n✓ All tests passed! You can now run the full clustering analysis.")
    print("  Run: python playground/clustering_analysis.py")


if __name__ == "__main__":
    test_clustering()
