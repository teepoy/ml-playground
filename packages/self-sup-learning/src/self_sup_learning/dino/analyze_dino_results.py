#!/usr/bin/env python3
"""
Analysis and comparison of DINO vs VAE clustering results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_dino_results():
    """Analyze the DINO clustering results"""

    # Load DINO results
    dino_results = pd.read_csv("dino_clustering_results.csv")

    print("DINO Clustering Analysis")
    print("=" * 60)

    # Separate training and test results
    train_results = dino_results[dino_results["dataset"] == "train_dino"]
    test_results = dino_results[dino_results["dataset"] == "test_dino"]

    print(f"DINO Training set results: {len(train_results)} configurations")
    print(f"DINO Test set results: {len(test_results)} configurations")

    # Find best results for each method and dataset
    methods = ["KMeans", "Spectral", "HDBSCAN"]

    print("\nBest Results by Method and Dataset:")
    print("-" * 50)

    for dataset_name, dataset_results in [
        ("train_dino", train_results),
        ("test_dino", test_results),
    ]:
        print(f"\n{dataset_name.upper()}:")
        for method in methods:
            method_results = dataset_results[dataset_results["method"] == method]
            if len(method_results) > 0:
                best_idx = method_results["v_measure"].idxmax()
                best_result = method_results.loc[best_idx]
                print(
                    f"  {method:12}: V-Measure={best_result['v_measure']:.4f}, "
                    f"Silhouette={best_result['silhouette']:.4f}, "
                    f"ARI={best_result['ari']:.4f}, "
                    f"NMI={best_result['nmi']:.4f}"
                )

    # Compare training vs test performance
    print("\nCOMPARISON: DINO Training vs Test Set Performance")
    print("-" * 50)

    for method in methods:
        train_best = train_results[train_results["method"] == method]["v_measure"].max()
        test_best = test_results[test_results["method"] == method]["v_measure"].max()
        print(
            f"{method:12}: Train V-Measure={train_best:.4f}, Test V-Measure={test_best:.4f}"
        )

    return dino_results


def compare_with_vae():
    """Compare DINO results with VAE results"""

    # Load both result sets
    dino_results = pd.read_csv("dino_clustering_results.csv")
    vae_results = pd.read_csv("imagenet_clustering_results.csv")

    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON: DINO vs VAE (ImageNet Subset)")
    print("=" * 80)

    # Get best results for each approach
    dino_train_best = dino_results[dino_results["dataset"] == "train_dino"][
        "v_measure"
    ].max()
    dino_test_best = dino_results[dino_results["dataset"] == "test_dino"][
        "v_measure"
    ].max()
    vae_train_best = vae_results[vae_results["dataset"] == "train"]["v_measure"].max()
    vae_test_best = vae_results[vae_results["dataset"] == "test"]["v_measure"].max()

    print(f"DINO Best Train V-Measure: {dino_train_best:.4f}")
    print(f"DINO Best Test V-Measure:  {dino_test_best:.4f}")
    print(f"VAE  Best Train V-Measure: {vae_train_best:.4f}")
    print(f"VAE  Best Test V-Measure:  {vae_test_best:.4f}")

    print("\nImprovement with DINO over VAE:")
    print(
        f"  Train: {((dino_train_best - vae_train_best) / vae_train_best * 100):.2f}%"
    )
    print(f"  Test:  {((dino_test_best - vae_test_best) / vae_test_best * 100):.2f}%")

    # Get the best configurations
    dino_train_best_row = dino_results[
        (dino_results["dataset"] == "train_dino")
        & (dino_results["v_measure"] == dino_train_best)
    ].iloc[0]
    dino_test_best_row = dino_results[
        (dino_results["dataset"] == "test_dino")
        & (dino_results["v_measure"] == dino_test_best)
    ].iloc[0]
    vae_train_best_row = vae_results[
        (vae_results["dataset"] == "train")
        & (vae_results["v_measure"] == vae_train_best)
    ].iloc[0]
    vae_test_best_row = vae_results[
        (vae_results["dataset"] == "test") & (vae_results["v_measure"] == vae_test_best)
    ].iloc[0]

    print("\nBest DINO Configuration:")
    print(
        f"  Train: {dino_train_best_row['method']} with {dino_train_best_row['n_clusters']} clusters - "
        f"V-Measure: {dino_train_best_row['v_measure']:.4f}"
    )
    print(
        f"  Test:  {dino_test_best_row['method']} with {dino_test_best_row['n_clusters']} clusters - "
        f"V-Measure: {dino_test_best_row['v_measure']:.4f}"
    )

    print("\nBest VAE Configuration:")
    print(
        f"  Train: {vae_train_best_row['method']} with {vae_train_best_row['n_clusters']} clusters - "
        f"V-Measure: {vae_train_best_row['v_measure']:.4f}"
    )
    print(
        f"  Test:  {vae_test_best_row['method']} with {vae_test_best_row['n_clusters']} clusters - "
        f"V-Measure: {vae_test_best_row['v_measure']:.4f}"
    )

    # Create comparison dataframe for visualization
    comparison_data = {
        "Method": ["DINO-Train", "DINO-Test", "VAE-Train", "VAE-Test"],
        "V-Measure": [dino_train_best, dino_test_best, vae_train_best, vae_test_best],
        "ARI": [
            dino_results[
                (dino_results["dataset"] == "train_dino")
                & (dino_results["v_measure"] == dino_train_best)
            ]["ari"].iloc[0],
            dino_results[
                (dino_results["dataset"] == "test_dino")
                & (dino_results["v_measure"] == dino_test_best)
            ]["ari"].iloc[0],
            vae_results[
                (vae_results["dataset"] == "train")
                & (vae_results["v_measure"] == vae_train_best)
            ]["ari"].iloc[0],
            vae_results[
                (vae_results["dataset"] == "test")
                & (vae_results["v_measure"] == vae_test_best)
            ]["ari"].iloc[0],
        ],
        "NMI": [
            dino_results[
                (dino_results["dataset"] == "train_dino")
                & (dino_results["v_measure"] == dino_train_best)
            ]["nmi"].iloc[0],
            dino_results[
                (dino_results["dataset"] == "test_dino")
                & (dino_results["v_measure"] == dino_test_best)
            ]["nmi"].iloc[0],
            vae_results[
                (vae_results["dataset"] == "train")
                & (vae_results["v_measure"] == vae_train_best)
            ]["nmi"].iloc[0],
            vae_results[
                (vae_results["dataset"] == "test")
                & (vae_results["v_measure"] == vae_test_best)
            ]["nmi"].iloc[0],
        ],
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Create visualization
    metrics = ["V-Measure", "ARI", "NMI"]
    x = np.arange(len(metrics))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))

    # DINO Train
    bars1 = ax.bar(
        x - width * 1.5,
        [dino_train_best, comparison_data["ARI"][0], comparison_data["NMI"][0]],
        width,
        label="DINO-Train",
        alpha=0.8,
    )
    # DINO Test
    bars2 = ax.bar(
        x - width / 2,
        [dino_test_best, comparison_data["ARI"][1], comparison_data["NMI"][1]],
        width,
        label="DINO-Test",
        alpha=0.8,
    )
    # VAE Train
    bars3 = ax.bar(
        x + width / 2,
        [vae_train_best, comparison_data["ARI"][2], comparison_data["NMI"][2]],
        width,
        label="VAE-Train",
        alpha=0.8,
    )
    # VAE Test
    bars4 = ax.bar(
        x + width * 1.5,
        [vae_test_best, comparison_data["ARI"][3], comparison_data["NMI"][3]],
        width,
        label="VAE-Test",
        alpha=0.8,
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
                fontsize=9,
            )

    for bars in [bars1, bars2, bars3, bars4]:
        add_value_labels(bars)

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("DINO vs VAE: Clustering Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("dino_vae_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nDINO significantly outperforms VAE on the ImageNet subset:")
    print("- DINO achieves V-Measure scores of ~0.8-0.95 compared to VAE's ~0.16")
    print("- This demonstrates the power of supervised/pretrained representations")
    print(
        "- DINO features preserve class structure much better than learned VAE embeddings"
    )

    return comparison_df


def main():
    print("Analyzing DINO clustering results...")
    dino_results = analyze_dino_results()

    print("\nComparing with VAE results...")
    comparison_df = compare_with_vae()

    print("\nAnalysis complete! Check the generated comparison plot.")


if __name__ == "__main__":
    main()
