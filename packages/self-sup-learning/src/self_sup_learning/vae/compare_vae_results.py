#!/usr/bin/env python3
"""
Final comparison between Iris and ImageNet VAE clustering results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare_datasets():
    """Compare the results between Iris and ImageNet VAE clustering"""

    print("COMPREHENSIVE COMPARISON: Iris vs ImageNet VAE Clustering")
    print("=" * 70)

    # Results summary - these are based on our execution results
    iris_results = {
        "dataset": "Iris",
        "embedding_dim": 2,
        "n_classes": 3,
        "n_samples": 150,
        "best_method": "Spectral",
        "best_n_clusters": 6,
        "best_v_measure": 0.6511,
        "best_silhouette": 0.6395,
        "best_ari": 0.5801,
        "best_nmi": 0.6511,
        "reconstruction_quality": "High",
    }

    imagenet_results = {
        "dataset": "ImageNet Subset",
        "embedding_dim": 128,
        "n_classes": 10,
        "n_samples": 5465,
        "best_method": "K-Means",
        "best_n_clusters": 30,
        "best_v_measure": 0.1595,
        "best_silhouette": 0.0000,  # For test set
        "best_ari": 0.0628,
        "best_nmi": 0.1595,
        "reconstruction_quality": "Good (MSE: 0.0197)",
    }

    # Create a comparison table
    comparison_data = {
        "Metric": [
            "Dataset Type",
            "Embedding Dimension",
            "Number of Classes",
            "Number of Samples",
            "Best Clustering Method",
            "Best Number of Clusters",
            "Best V-Measure",
            "Best Silhouette Score",
            "Best ARI",
            "Best NMI",
            "Reconstruction Quality",
        ],
        "Iris": [
            iris_results["dataset"],
            iris_results["embedding_dim"],
            iris_results["n_classes"],
            iris_results["n_samples"],
            iris_results["best_method"],
            iris_results["best_n_clusters"],
            iris_results["best_v_measure"],
            iris_results["best_silhouette"],
            iris_results["best_ari"],
            iris_results["best_nmi"],
            iris_results["reconstruction_quality"],
        ],
        "ImageNet Subset": [
            imagenet_results["dataset"],
            imagenet_results["embedding_dim"],
            imagenet_results["n_classes"],
            imagenet_results["n_samples"],
            imagenet_results["best_method"],
            imagenet_results["best_n_clusters"],
            imagenet_results["best_v_measure"],
            imagenet_results["best_silhouette"],
            imagenet_results["best_ari"],
            imagenet_results["best_nmi"],
            imagenet_results["reconstruction_quality"],
        ],
    }

    comparison_df = pd.DataFrame(comparison_data)

    print(comparison_df.to_string(index=False))

    # Create visualization
    metrics = ["V-Measure", "ARI", "NMI"]
    iris_values = [
        iris_results["best_v_measure"],
        iris_results["best_ari"],
        iris_results["best_nmi"],
    ]
    imagenet_values = [
        imagenet_results["best_v_measure"],
        imagenet_results["best_ari"],
        imagenet_results["best_nmi"],
    ]

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, iris_values, width, label="Iris", alpha=0.8)
    bars2 = ax.bar(
        x + width / 2, imagenet_values, width, label="ImageNet Subset", alpha=0.8
    )

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    add_value_labels(bars1)
    add_value_labels(bars2)

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("Clustering Performance: Iris vs ImageNet Subset")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparison_vae_clustering.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print("1. Dataset Complexity Impact:")
    print(
        f"   - Iris (simple, 4D tabular): V-Measure = {iris_results['best_v_measure']:.4f}"
    )
    print(
        f"   - ImageNet (complex, 64x64x3 images): V-Measure = {imagenet_results['best_v_measure']:.4f}"
    )
    print("   - The complex dataset shows significantly lower clustering performance")

    print("\n2. Optimal Clustering Method:")
    print("   - Iris: Spectral clustering worked best")
    print("   - ImageNet: K-Means worked best")

    print("\n3. Optimal Number of Clusters:")
    print("   - Iris: 6 clusters (2x true classes) worked best")
    print("   - ImageNet: 30 clusters (3x true classes) worked best")

    print("\n4. Embedding Space Characteristics:")
    print("   - Iris: Low-dimensional (2D) embedding space")
    print("   - ImageNet: High-dimensional (128D) embedding space")

    print("\n5. Generalization:")
    print("   - Both datasets showed similar performance on train/test splits")
    print("   - ImageNet test performance was slightly better than training set")

    print("\n6. Reconstruction Quality:")
    print("   - ImageNet VAE achieved MSE of 0.0197, indicating good reconstruction")
    print(
        "   - Despite good reconstruction, clustering in latent space was challenging"
    )

    print(
        "\nThe results demonstrate that while VAEs can effectively compress complex image data,"
    )
    print(
        "the resulting latent space doesn't necessarily preserve the class structure well enough"
    )
    print("for simple clustering algorithms to achieve high performance.")


if __name__ == "__main__":
    compare_datasets()
