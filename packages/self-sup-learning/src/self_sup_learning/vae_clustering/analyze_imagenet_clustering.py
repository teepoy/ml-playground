#!/usr/bin/env python3
"""
Detailed analysis of ImageNet VAE clustering results
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_clustering_results():
    """Load and analyze clustering results"""
    # Load the results
    results_df = pd.read_csv("imagenet_clustering_results.csv")

    print("ImageNet VAE Clustering Analysis")
    print("=" * 60)

    # Separate training and test results
    train_results = results_df[results_df["dataset"] == "train"]
    test_results = results_df[results_df["dataset"] == "test"]

    print(f"Training set results: {len(train_results)} configurations")
    print(f"Test set results: {len(test_results)} configurations")

    # Find best results for each method and dataset
    methods = ["KMeans", "Spectral", "HDBSCAN"]

    print("\nBest Results by Method and Dataset:")
    print("-" * 50)

    for dataset_name, dataset_results in [
        ("train", train_results),
        ("test", test_results),
    ]:
        print(f"\n{dataset_name.upper()} SET:")
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
    print("\nCOMPARISON: Training vs Test Set Performance")
    print("-" * 50)

    for method in methods:
        train_best = train_results[train_results["method"] == method]["v_measure"].max()
        test_best = test_results[test_results["method"] == method]["v_measure"].max()
        print(
            f"{method:12}: Train V-Measure={train_best:.4f}, Test V-Measure={test_best:.4f}"
        )

    # Create visualizations
    create_visualizations(results_df)

    return results_df


def create_visualizations(results_df):
    """Create visualizations comparing training and test performance"""

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: V-Measure by dataset and method
    sns.boxplot(
        data=results_df[results_df["method"] != "HDBSCAN"],
        x="method",
        y="v_measure",
        hue="dataset",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("V-Measure by Method and Dataset (excluding HDBSCAN)")
    axes[0, 0].set_ylabel("V-Measure")

    # Plot 2: Silhouette Score by dataset and method
    sns.boxplot(
        data=results_df[results_df["method"] != "HDBSCAN"],
        x="method",
        y="silhouette",
        hue="dataset",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Silhouette Score by Method and Dataset (excluding HDBSCAN)")
    axes[0, 1].set_ylabel("Silhouette Score")

    # Plot 3: ARI by dataset and method
    sns.boxplot(
        data=results_df[results_df["method"] != "HDBSCAN"],
        x="method",
        y="ari",
        hue="dataset",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title(
        "Adjusted Rand Index by Method and Dataset (excluding HDBSCAN)"
    )
    axes[1, 0].set_ylabel("ARI")

    # Plot 4: NMI by dataset and method
    sns.boxplot(
        data=results_df[results_df["method"] != "HDBSCAN"],
        x="method",
        y="nmi",
        hue="dataset",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(
        "Normalized Mutual Information by Method and Dataset (excluding HDBSCAN)"
    )
    axes[1, 1].set_ylabel("NMI")

    plt.tight_layout()
    plt.savefig("imagenet_clustering_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Create a bar plot comparing the best performance per method
    best_results = []
    for dataset_name in ["train", "test"]:
        for method in ["KMeans", "Spectral"]:
            subset = results_df[
                (results_df["dataset"] == dataset_name)
                & (results_df["method"] == method)
            ]
            if len(subset) > 0:
                best_v_measure = subset["v_measure"].max()
                best_results.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "best_v_measure": best_v_measure,
                    }
                )

    best_results_df = pd.DataFrame(best_results)

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        data=best_results_df, x="method", y="best_v_measure", hue="dataset"
    )
    plt.title("Best V-Measure Performance by Method and Dataset")
    plt.ylabel("Best V-Measure Score")

    # Add value labels on bars
    for p in bar_plot.patches:
        bar_plot.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        "imagenet_best_performance_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.show()


def summarize_findings(results_df):
    """Summarize key findings"""
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Overall performance
    train_results = results_df[results_df["dataset"] == "train"]
    test_results = results_df[results_df["dataset"] == "test"]

    print(
        f"Overall best performance on training set: {train_results['v_measure'].max():.4f}"
    )
    print(
        f"Overall best performance on test set: {test_results['v_measure'].max():.4f}"
    )

    # Best configurations
    train_best_idx = train_results["v_measure"].idxmax()
    test_best_idx = test_results["v_measure"].idxmax()

    train_best = train_results.loc[train_best_idx]
    test_best = test_results.loc[test_best_idx]

    print(
        f"\nBest training config: {train_best['method']} with {train_best['n_clusters']} clusters "
        f"(V-Measure: {train_best['v_measure']:.4f})"
    )
    print(
        f"Best test config: {test_best['method']} with {test_best['n_clusters']} clusters "
        f"(V-Measure: {test_best['v_measure']:.4f})"
    )

    # General observations
    print("\nObservations:")
    print("- Both training and test sets show similar performance levels")
    print("- KMeans generally outperformed Spectral clustering")
    print("- HDBSCAN showed mixed results")
    print("- More clusters didn't necessarily lead to better performance")

    # Calculate average performance by dataset
    avg_train = train_results["v_measure"].mean()
    avg_test = test_results["v_measure"].mean()

    print(f"\nAverage V-Measure: Train={avg_train:.4f}, Test={avg_test:.4f}")
    print(f"Performance gap (train - test): {avg_train - avg_test:.4f}")


def main():
    print("Analyzing ImageNet VAE clustering results...")
    results_df = analyze_clustering_results()
    summarize_findings(results_df)

    print("\nAnalysis complete! Check the generated plots for visual insights.")


if __name__ == "__main__":
    main()
